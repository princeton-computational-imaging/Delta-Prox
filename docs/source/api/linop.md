# Linear Operator


```{eval-rst}
.. autoclass:: dprox.linop.base.LinOp
   :members: forward, adoint, is_gram_diag, is_diag, get_diag, variables, constants, is_constant, value, offset, norm_bound, T, gram, clone
.. autoclass:: dprox.linop.conv.conv
   :members: forward, adjoint
.. autoclass:: dprox.linop.conv.conv_doe
   :members: forward, adjoint
.. autoclass:: dprox.linop.sum.sum
   :members: forward, adjoint
.. autoclass:: dprox.linop.sum.copy
   :members: forward, adjoint
.. autoclass:: dprox.linop.vstack.vstack
   :members: forward, adjoint
.. autoclass:: dprox.linop.vstack.split
   :members: forward, adjoint
.. autoclass:: dprox.linop.variable.Variable
   :members: forward, adjoint
.. autoclass:: dprox.linop.subsample.mosaic
   :members: forward, adjoint
.. autoclass:: dprox.linop.scale.scale
   :members: forward, adjoint
.. autoclass:: dprox.linop.placeholder.PlaceHolder
   :members: forward, adjoint
.. autoclass:: dprox.linop.grad.grad
   :members: forward, adjoint
```