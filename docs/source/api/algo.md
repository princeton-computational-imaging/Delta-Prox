# Proximal Algorithms

Extending ∇-Prox for more proximal algorithms is also easy and straightforward. We again define a new algorithm class that inherits from the base class `Algorithm`. For this case, the required methods to be implemented are `partition` and `_iter`, which represent the problem partition and a single algorithm iteration. 

The `partition` takes a list of proxable functions and returns their splits as a list of `psi_fn` and `omega_fn`. For `_iter`, it is a single iteration of the proximal algorithm that takes an input of state and two parameters, `rho` for penalty strength on multipliers and `lam` for proximal operators. 

The state is generally a list of variables, including the auxiliary ones that an algorithm creates. ∇-Prox simply provides a state as what returns in the previous execution of `_iter` or the initial state provided by the `initialize` method. 

```python
class new_algorithm(Algorithm):
    def partition(cls, prox_fns: List[ProxFn]):
        # Perform problem partition on algorithm's need

    def __init__(...):
        # Custom initialization code.

    def _iter(self, state, rho, lam):
        # Code to compute the function's proximal operator.
        return ...
        
    def initialize(self, x0):
        # Return the initial state
        return ...
        
    def nparams(self):
        # (Optional) Return the number of hyperparameters of 
        # this algorithm 
        return ...
        
    def state_split(self):
        # (Optional) Return the split size of the packed state.
        # Useful for deep equilibrium/reinforcement learning.
        return ...
```

The implementations of `partition`, `initialize`, and `_iter` are generally enough to evaluate the proximal algorithm for a given problem. 

To integrate it with deep equilibrium learning (DEQ) and deep reinforcement learning (RL), users have to implement two additional helper methods, i.e., `params` for counting the number of hyperparameters and `state_split` for the structures of the state that returns by `_iter`. 

For example, assuming `_iter` returns the state as nested arrays like `[x,[v1,v2],[u1,u2]]`, the output of `state_split` should be `[1,[2],[2]]`. ∇-Prox exploits these properties to perform necessary packing and unpacking for the iteration states to achieve a unified interface for the internal DEQ and RL implementations.
