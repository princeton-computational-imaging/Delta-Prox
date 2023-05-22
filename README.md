
<a href="https://deltaprox.readthedocs.io/"><img src="https://user-images.githubusercontent.com/26198430/229363778-a546ffce-b43d-4f4a-9869-629208f050cc.svg" alt="" width="30%"></a> &ensp; 


Differentiable Proximal Algorithm Modeling for Large-Scale Optimization

<a href="https://deltaprox.readthedocs.io/">Docs</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx/tree/master/tutorials">Tutorials</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx/tree/master/examples">Examples</a> |
<a href="#">Paper</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx#citation">Citation</a> 

<a href="https://pypi.org/project/dprox/">![Version](https://img.shields.io/pypi/v/dprox)</a>
  <a href="https://arxiv.org/abs/2207.02849">![arXiv](https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg)</a>


```bash
pip install dprox
```

> $\nabla$-Prox is a domain-specific language (DSL) and compiler that transforms optimization problems into differentiable proximal solvers. Departing from handwriting these solvers and differentiating via autograd, $\nabla$-Prox requires only a few lines of code to define a solver that can be *specialized based on user requirements w.r.t memory constraints or training budget* by optimized algorithm unrolling, deep equilibrium learning, and deep reinforcement learning. $\nabla$-Prox makes it easier to prototype different learning-based bi-level optimization problems for a diverse range of applications. We compare our framework against existing methods with naive implementations. $\nabla$-Prox is significantly more compact in terms of lines of code and compares favorably in memory consumption in applications across domains.

## News
 
**[Jan 21 2023]**  Release preview code.

> 🚧 The code is still under construction, more features would be migrated from our dev code.

## Getting Started

### Solver construction

Consider a simple image deconvlution problem, where we seek to find a clean image $x$ given the blurred observation $y$ that minimizes the following objective function:

$$
\arg \min_x { \frac{1}{2} |Dx - y|^2_2 + g(x) }
$$

where $g(x)$ denotes an implicit plug-and-play denoiser prior. We could solve this problem in ∇-Prox with the following code: 

```python
from dprox import *
from dprox.utils import *
from dprox.utils.examples import *

img = sample()
psf = point_spread_function(15, 5)
b = blurring(img, psf)

x = Variable()
data_term = sum_squares(conv(x, psf) - b)
reg_term = deep_prior(x, denoiser='ffdnet_color')
prob = Problem(data_term + reg_term)

out = prob.solve(method='admm', x0=b)
```

Here is what we got,

<img src="docs/source/_static/example_deconv.png" width="500" />

### Solver Specialization

```python

```


Please refer to the [documentation]() site for more instructions on the efficient differentiation of proximal algorithm with ∇-Prox.

## Citation

```bibtex
@article{deltaprox2023,
  title={∇-Prox: Differentiable Proximal Algorithm Modeling for Large-Scale Optimization},
  author={Lai, Zeqiang and Wei, Kaixuan and Fu, Ying and Härtel, Philipp and Heide, Felix},
  journal={ACM Transactions on Graphics},
  year={2023},
}
```
