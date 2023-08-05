# Proximal Functions

The following code shows a template for defining a new proxable function. As previously mentioned, we define the function as a class inheriting from the base class `ProxFn`, and implement all the required methods. Then, ∇-Prox will properly handle all other necessary steps so that the new proxable function can work with operators, algorithms, and training utilities of the existing system.

```python
class new_func(ProxFn):
    def __init__(...):
        """ Custom initialization code.
        """

    def _prox(self, tau, v):
        """ Code to compute the function's proximal operator.
        """
        return ...

    def _eval(self, v):
        """ (Optional) Code to evaluate the function.
        """
        return ...
        
    def _grad(self, v):
        """ (Optional) Code to compute the analytic gradient.
        """
        return ...
```

More specifically, defining a new function only requires a method `_prox` to be implemented, which evaluates the proximal operator of the given function. 

Users can optionally implement the `_grad` function to provide a routine for computing the analytic gradient of the proxable function. This facilitates the algorithms that partially rely on the gradient evaluation, e.g., proximal gradient descent. 

Alternatively, users can also implement the `_eval` method that computes the forwarding results of the proxable function if it is possible. ∇-Prox takes the `_eval` routine and computes the gradient with auto-diff if `_grad` is not implemented.


```{eval-rst}
.. autoclass:: dprox.proxfn.base.ProxFn
```