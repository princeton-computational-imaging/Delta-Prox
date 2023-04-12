import torch
from .cg import conjugate_gradient2, conjugate_gradient3
from functools import partial


class CustomLinearSolve(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, linop_args):
        # ctx.solver_options = solver_options
        ctx.A = A
        # lin_solver_type = solver_options.pop(lin_solver_type)
        ctx.lin_solver = partial(conjugate_gradient2, **linop_args)
        x = ctx.lin_solver(A, B)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_B = ctx.lin_solver(ctx.A, grad_x)
        return None, grad_B, None


def custom_conjugate_gradient(A, b, x_init=None, tol=1e-5, 
                              num_iters=100, verbose=False, **kwargs):
    out = CustomLinearSolve.apply(A, b, dict(x_init=x_init, tol=tol, num_iters=num_iters, verbose=verbose))
    return out



class CustomLinearSolve2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, height_map, linop_args, *Aparams):
        ctx.A = A
        ctx.lin_solver = partial(conjugate_gradient2, **linop_args)
        ctx.height_map = height_map 
        x,_ = ctx.lin_solver(A, B)
        ctx.save_for_backward(x, *Aparams)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        # print('########## compute gradient')
        v, res = ctx.lin_solver(ctx.A, grad_x)
        grad_B = v
        
        x = ctx.saved_tensors[0]
        x = x.detach().clone()
        Aparams = ctx.height_map.height_map_sqrt.clone().requires_grad_()
        backup = ctx.height_map.height_map_sqrt
        params = torch.nn.Parameter(Aparams)
        ctx.height_map.height_map_sqrt = params
        with torch.enable_grad():  
            loss = -ctx.A(x)
        # print('backward', id(ctx.height_map), id(ctx.height_map.height_map_sqrt), ctx.height_map.height_map_sqrt.requires_grad)
        # print(type(loss), type(Aparams), type(v))
        grad_params = torch.autograd.grad((loss,), [ctx.height_map.height_map_sqrt], grad_outputs=(v,),
                                          create_graph=torch.is_grad_enabled(),
                                          allow_unused=True)
        
        ctx.height_map.height_map_sqrt = backup
        try:
            if res > 1e-3:
                grad_params = (torch.zeros_like(grad_params[0]) if grad_params[0] is not None else None,)
                grad_B = torch.zeros_like(grad_B)
        except:
            import ipdb; ipdb.set_trace()
            
        return (None, grad_B, None, None, *grad_params)


def custom_conjugate_gradient2(A, b, height_map=None, x_init=None, tol=1e-5, num_iters=100, verbose=False):

    # b = b.detach().clone().requires_grad_()
    # with torch.enable_grad():
    #     out = conjugate_gradient3(A,b,tol=1e-5,num_iters=100,verbose=True)
    # out.mean().backward()
    # print(b.grad)
    
    # import ipdb; ipdb.set_trace()
    
    out = CustomLinearSolve2.apply(A, b, height_map, 
                                    dict(x_init=x_init, tol=tol, num_iters=num_iters, verbose=verbose), 
                                    height_map.height_map_sqrt)

    # out.mean().backward()
    # print(b.grad)
    # import ipdb; ipdb.set_trace()
    return out