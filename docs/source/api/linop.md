# Linear Operator

Defining new linear operators mainly involves the definition of the forward and adjoint routines. The following code shows the template for defining them. Similar to a [proxable function](), the operator is defined as a class inheriting from the base class `LinOp`.

```python
class new_linop(LinOp): 
   def __init__(...):
      """ Custom initialization code. 
      """
      
   def forward(self, inputs):
        """The forward operator. Compute x -> Kx
        """

   def adjoint(self, inputs):
        """The adjoint operator. Compute x -> K^Tx
        """

   def is_diag(self, freq):
      """(Optional) Check if the linear operator is diagonalizable or
         diagonalizable in the frequency domain.
      """

   def get_diag(self, x, freq):
      """(Optional) Return the diagonal/frequency diagonal matrix that 
         matches the shape of input x.
      """
```

By default, the linear operator is not diagonal. To introduce a diagonal linear operator, one must implement the `is_diag` and `get_diag` for checking the diagonalizability and acquiring the diagonal matrix. These methods facilitate ∇-Prox to construct more efficient solvers, e.g., ADMM with closed-form matrix inverse for the least-square update.

## Sanity Check

Typically, it is not always easy to correctly implement the forward and adjoint operations of the linear operator. To facilitate the testing of these operators, ∇-Prox provides an implementation of the **dot-product test** for verifying that the `forward` and `adjoint` are adjoint to each other.

Basically, the idea of the dot-product test comes from the associative property of linear algebra, which gives the following equation,

$$
y^T(Ax) = (A^Ty)^Tx
$$

where $x$ and $y$ are randomly generated data, and $A$ and $A^T$ denote the forward and adjoint of the linear operator. ∇-Prox makes use of this property and generates a large number of random data to check if this equation always holds with respect to a given precision. 

To use this utility, users can call the `validate(linop, tol=1e-6)` and specify the tolerance of the difference between two sides of the equation. 

```python
import dprox as dp
from dprox.utils.examples import fspecial_gaussian

x = dp.Variable()
psf = fspecial_gaussian(15, 5)
op = dp.conv(x, psf)
assert dp.validate(op)
```


## Indices

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