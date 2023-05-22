# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import torch
from typing import NamedTuple


class MINRESSettings(NamedTuple):
    max_cg_iterations: int = 1000  # The maximum number of conjugate gradient iterations to perform (when computing
    # matrix solves). A higher value rarely results in more accurate solves -- instead, lower the CG tolerance.
    minres_tolerance: float = 1e-4  # Relative update term tolerance to use for terminating MINRES.
    verbose_linalg: bool = False  # Print out information whenever running an expensive linear algebra routine


def _pad_with_singletons(obj, num_singletons_before=0, num_singletons_after=0):
    """
    Pad obj with singleton dimensions on the left and right
    Example:
        >>> x = torch.randn(10, 5)
        >>> _pad_width_singletons(x, 2, 3).shape
        >>> # [1, 1, 10, 5, 1, 1, 1]
    """
    new_shape = [1] * num_singletons_before + list(obj.shape) + [1] * num_singletons_after
    return obj.view(*new_shape)


def MINRES(
    A,
    b,
    eps=1e-25,
    shifts=None,
    value=None,
    max_iters=None,
    preconditioner=None,
    settings: MINRESSettings = MINRESSettings(),
):
    r"""
    Perform MINRES to find solutions to :math:`(\mathbf K + \alpha \sigma \mathbf I) \mathbf x = \mathbf b`.
    Will find solutions for multiple shifts :math:`\sigma` at the same time.

    :param callable A: Function to perform matmul with.
    :param torch.Tensor b: The vector :math:`\mathbf b` to solve against.
    :param torch.Tensor shifts: (default None) The shift :math:`\sigma` values. If set to None,
        then :math:`\sigma=0`.
    :param float value: (default None) The multiplicative constant :math:`\alpha`. If set to None,
        then :math:`\alpha=0`.
    :param int max_iters: (default None) The maximum number of minres iterations. If set to None, then
        uses the constant stored in :obj:`linear_operator.settings.max_cg_iterations`.
    :rtype: torch.Tensor
    :return: The solves :math:`\mathbf x`. The shape will correspond to the size of `b` and `shifts`.
    """
    # Default values
    if torch.is_tensor(A):
        A = A.matmul
    mm_ = A
    if preconditioner is None:
        preconditioner = lambda x: x.clone()

    if shifts is None:
        shifts = torch.tensor(0.0, dtype=b.dtype, device=b.device)

    # Scale the b
    squeeze = False
    if b.dim() == 1:
        b = b.unsqueeze(-1)
        squeeze = True

    rhs_norm = b.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(1e-10)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    b = b.div(rhs_norm)

    # Use the right number of iterations
    if max_iters is None:
        max_iters = settings.max_cg_iterations
    max_iters = min(max_iters, b.size(-2) + 1)

    # Epsilon (to prevent nans)
    eps = torch.tensor(eps, dtype=b.dtype, device=b.device)

    # Create space for matmul product, solution
    prod = mm_(b)
    if value is not None:
        prod.mul_(value)

    # Resize shifts
    shifts = _pad_with_singletons(shifts, 0, prod.dim() - shifts.dim() + 1)
    solution = torch.zeros(shifts.shape[:1] + prod.shape, dtype=b.dtype, device=b.device)

    # Variables for Lanczos terms
    zvec_prev2 = torch.zeros_like(prod)
    zvec_prev1 = b.clone().expand_as(prod).contiguous()
    qvec_prev1 = preconditioner(zvec_prev1)
    alpha_curr = torch.empty(prod.shape[:-2] + (1, prod.size(-1)), dtype=b.dtype, device=b.device)
    alpha_shifted_curr = torch.empty(solution.shape[:-2] + (1, prod.size(-1)), dtype=b.dtype, device=b.device)
    beta_prev = (zvec_prev1 * qvec_prev1).sum(dim=-2, keepdim=True).sqrt_()
    beta_curr = torch.empty_like(beta_prev)
    tmpvec = torch.empty_like(qvec_prev1)

    # Divide by beta_prev
    zvec_prev1.div_(beta_prev)
    qvec_prev1.div_(beta_prev)

    # Variables for the QR rotation
    # 1) Components of the Givens rotations
    cos_prev2 = torch.ones(solution.shape[:-2] + (1, b.size(-1)), dtype=b.dtype, device=b.device)
    sin_prev2 = torch.zeros(solution.shape[:-2] + (1, b.size(-1)), dtype=b.dtype, device=b.device)
    cos_prev1 = torch.ones_like(cos_prev2)
    sin_prev1 = torch.zeros_like(sin_prev2)
    radius_curr = torch.empty_like(cos_prev1)
    cos_curr = torch.empty_like(cos_prev1)
    sin_curr = torch.empty_like(cos_prev1)
    # 2) Terms QR decomposition of T
    subsub_diag_term = torch.empty_like(alpha_shifted_curr)
    sub_diag_term = torch.empty_like(alpha_shifted_curr)
    diag_term = torch.empty_like(alpha_shifted_curr)

    # Variables for the solution updates
    # 1) The "search" vectors of the solution
    # Equivalent to the vectors of Q R^{-1}, where Q is the matrix of Lanczos vectors and
    # R is the QR factor of the tridiagonal Lanczos matrix.
    search_prev2 = torch.zeros_like(solution)
    search_prev1 = torch.zeros_like(solution)
    search_curr = torch.empty_like(search_prev1)
    search_update = torch.empty_like(search_prev1)
    # 2) The "scaling" terms of the search vectors
    # Equivalent to the terms of V^T Q^T b, where Q is the matrix of Lanczos vectors and
    # V is the QR orthonormal of the tridiagonal Lanczos matrix.
    scale_prev = beta_prev.repeat(shifts.size(0), *([1] * beta_prev.dim()))
    scale_curr = torch.empty_like(scale_prev)

    # Terms for checking for convergence
    solution_norm = torch.zeros(*solution.shape[:-2], solution.size(-1), dtype=solution.dtype, device=solution.device)
    search_update_norm = torch.zeros_like(solution_norm)

    # Maybe log
    if settings.verbose_linalg:
        # settings.verbose_linalg.logger.debug(
        print(
            f"Running MINRES on a {b.shape} RHS for {max_iters} iterations (tol={settings.minres_tolerance.value()}). "
            f"Output: {solution.shape}."
        )

    bnorm = torch.linalg.norm(b)
    # Perform iterations
    for i in range(max_iters + 2):
        # Perform matmul
        prod = mm_(qvec_prev1)
        if value is not None:
            prod.mul_(value)

        # Get next Lanczos terms
        # --> alpha_curr, beta_curr, qvec_curr
        torch.mul(prod, qvec_prev1, out=tmpvec)
        torch.sum(tmpvec, -2, keepdim=True, out=alpha_curr)

        zvec_curr = prod.addcmul_(alpha_curr, zvec_prev1, value=-1).addcmul_(beta_prev, zvec_prev2, value=-1)

        qvec_curr = preconditioner(zvec_curr)
        torch.mul(zvec_curr, qvec_curr, out=tmpvec)
        torch.sum(tmpvec, -2, keepdim=True, out=beta_curr)
        beta_curr.sqrt_()
        beta_curr.clamp_min_(eps)

        zvec_curr.div_(beta_curr)
        qvec_curr.div_(beta_curr)

        # Perform JIT-ted update
        conv = _jit_minres_updates(
            solution,
            shifts,
            eps,
            qvec_prev1,
            alpha_curr,
            alpha_shifted_curr,
            beta_prev,
            beta_curr,
            cos_prev2,
            cos_prev1,
            cos_curr,
            sin_prev2,
            sin_prev1,
            sin_curr,
            radius_curr,
            subsub_diag_term,
            sub_diag_term,
            diag_term,
            search_prev2,
            search_prev1,
            search_curr,
            search_update,
            scale_prev,
            scale_curr,
            search_update_norm,
            solution_norm,
        )

        # Check convergence criterion
        if (i + 1) % 10 == 0:
            # torch.norm(search_update, dim=-2, out=search_update_norm)
            # torch.norm(solution, dim=-2, out=solution_norm)
            # conv = search_update_norm.div_(solution_norm).mean().item()
            # if conv < settings.minres_tolerance:
            #     break

            r = mm_(solution[0]) - b
            rnorm = torch.linalg.norm(r)
            if rnorm <= settings.minres_tolerance * bnorm:
                break

        # Update terms for next iteration
        # Lanczos terms
        zvec_prev2, zvec_prev1 = zvec_prev1, prod
        qvec_prev1 = qvec_curr
        beta_prev, beta_curr = beta_curr, beta_prev
        # Givens rotations terms
        cos_prev2, cos_prev1, cos_curr = cos_prev1, cos_curr, cos_prev2
        sin_prev2, sin_prev1, sin_curr = sin_prev1, sin_curr, sin_prev2
        # Search vector terms)
        search_prev2, search_prev1, search_curr = search_prev1, search_curr, search_prev2
        scale_prev, scale_curr = scale_curr, scale_prev

    # For b-s that are close to zero, set them to zero
    solution.masked_fill_(rhs_is_zero, 0)

    if squeeze:
        solution = solution.squeeze(-1)
        b = b.squeeze(-1)
        rhs_norm = rhs_norm.squeeze(-1)

    if shifts.numel() == 1:
        # If we weren't shifting we shouldn't return a batch output
        solution = solution.squeeze(0)

    return solution.mul_(rhs_norm)


def _jit_minres_updates(
    solution,
    shifts,
    eps,
    qvec_prev1,
    alpha_curr,
    alpha_shifted_curr,
    beta_prev,
    beta_curr,
    cos_prev2,
    cos_prev1,
    cos_curr,
    sin_prev2,
    sin_prev1,
    sin_curr,
    radius_curr,
    subsub_diag_term,
    sub_diag_term,
    diag_term,
    search_prev2,
    search_prev1,
    search_curr,
    search_update,
    scale_prev,
    scale_curr,
    search_update_norm,
    solution_norm,
):
    # Start givens rotation
    # Givens rotation from 2 steps ago
    torch.mul(sin_prev2, beta_prev, out=subsub_diag_term)
    torch.mul(cos_prev2, beta_prev, out=sub_diag_term)

    # Compute shifted alpha
    torch.add(alpha_curr, shifts, out=alpha_shifted_curr)

    # Givens rotation from 1 step ago
    torch.mul(alpha_shifted_curr, cos_prev1, out=diag_term).addcmul_(sin_prev1, sub_diag_term, value=-1)
    sub_diag_term.mul_(cos_prev1).addcmul_(sin_prev1, alpha_shifted_curr)

    # 3) Compute next Givens terms
    torch.mul(diag_term, diag_term, out=radius_curr).addcmul_(beta_curr, beta_curr).sqrt_()
    cos_curr = torch.div(diag_term, radius_curr, out=cos_curr)
    sin_curr = torch.div(beta_curr, radius_curr, out=sin_curr)
    # 4) Apply current Givens rotation
    diag_term.mul_(cos_curr).addcmul_(sin_curr, beta_curr)

    # Update the solution
    # --> search_curr, scale_curr solution
    # 1) Apply the latest Givens rotation to the Lanczos-b ( ||b|| e_1 )
    # This is getting the scale terms for the "search" vectors
    torch.mul(scale_prev, sin_curr, out=scale_curr).mul_(-1)
    scale_prev.mul_(cos_curr)
    # 2) Get the new search vector
    torch.addcmul(qvec_prev1, sub_diag_term, search_prev1, value=-1, out=search_curr)
    search_curr.addcmul_(subsub_diag_term, search_prev2, value=-1)
    search_curr.div_(diag_term)

    # 3) Update the solution
    torch.mul(search_curr, scale_prev, out=search_update)
    solution.add_(search_update)
