<p align="center">
<a href="https://light.princeton.edu/publication/delta_prox/">
    <img src="docs/source/_static/logo.svg" alt="Delta Prox" width="16.5%">
    </a> &ensp; 
</p>


<p align="center">
Differentiable Proximal Algorithm Modeling for Large-Scale Optimization
</p>

<p align="center">
<a href="https://light.princeton.edu/publication/delta_prox/">Paper</a> |
<a href="https://github.com/princeton-computational-imaging/Delta-Prox/tree/main/notebooks">Tutorials</a> |
<a href="https://github.com/princeton-computational-imaging/Delta-Prox/tree/main/examples">Examples</a> |
<a href="https://github.com/princeton-computational-imaging/Delta-Prox#citation">Citation</a> 
</p>

<p align="center">
    <a href="https://pypi.org/project/dprox/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/dprox">
    </a>
    <a href="https://dl.acm.org/doi/abs/10.1145/3592144">
        <img alt="arXiv" src="https://img.shields.io/badge/doi-10.1145/3592144-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/delta-prox">
        <img alt="huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue">
    </a>

</p>


> $\nabla$-Prox is a domain-specific language (DSL) and compiler that transforms optimization problems into differentiable proximal solvers. Departing from handwriting these solvers and differentiating via autograd, $\nabla$-Prox requires only a few lines of code to define a solver that can be *specialized based on user requirements w.r.t memory constraints or training budget* by optimized algorithm unrolling, deep equilibrium learning, and deep reinforcement learning. $\nabla$-Prox makes it easier to prototype different learning-based bi-level optimization problems for a diverse range of applications. We compare our framework against existing methods with naive implementations. $\nabla$-Prox is significantly more compact in terms of lines of code and compares favorably in memory consumption in applications across domains.

## News


- **August 2023** : $\nabla$-Prox is presented at SIGGRAPH 2023 and its code base is now public.
- **May 2023** : $\nabla$-Prox is accepted as a journal paper at SIGGRAPH 2023.

## Installation

We recommend installing $\nabla$-Prox in a virtual environment from PyPI.

```bash
pip install dprox
```

<!-- Please refer to the [Installation](https://deltaprox.readthedocs.io/started/install) guide for other options. -->

## Quickstart
![pipeline2](docs/source/_static/pipeline_dprox.gif)


Consider a simple image deconvolution problem, where we want to find a clean image $x$ given the blurred observation $y$ that minimizes the following objective function:

$$
\arg \min_x { \frac{1}{2} |Dx - y|^2_2 + g(x) },
$$

where $g(x)$ denotes an implicit plug-and-play denoiser prior. We can solve this problem in ∇-Prox with the following code: 

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

We can also specialize the solver via bi-level optimization.
For example, we can specialize the solver into a reinforcement learning (RL) solver for automatic parameter tuning.

```python
solver = compile(data_term + reg_term, method='admm')
rl_solver = specialize(solver, method='rl')
rl_solver = train(rl_solver, dataset)
```

Alternatively, we can specialize the solver into an unrolled solver for end-to-end optics optimization.

```python
x = Variable()
y = Placeholder()
PSF = Placeholder()
data_term = sum_squares(conv_doe(x, PSF, circular=True) - y)
reg_term = deep_prior(x, denoiser='ffdnet_color')
solver = compile(data_term + reg_term, method='admm')
unrolled_solver = specialize(solver, step=10)
train(unrolled_solver, dataset)
```

<!-- Want to learn more? Check out the [documentation](https://deltaprox.readthedocs.io/) or have a look at our [tutorials](https://github.com/princeton-computational-imaging/Delta-Prox/tree/main/notebooks).-->
Want to learn more? Check out the step-by-step [tutorials](https://github.com/princeton-computational-imaging/Delta-Prox/tree/main/notebooks) for the framework and its applications.

## Citation

```bibtex
@article{deltaprox2023,
  title = {∇-Prox: Differentiable Proximal Algorithm Modeling for Large-Scale Optimization},
  author = {Lai, Zeqiang and Wei, Kaixuan and Fu, Ying and H\"{a}rtel, Philipp and Heide, Felix},
  journal={ACM Transactions on Graphics (TOG)},
  volume = {42},
  number = {4},
  articleno = {105},
  pages = {1--19},
  year={2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3592144},
}
```

## Acknowledgement

[ProxImaL](https://github.com/comp-imaging/ProxImaL) &ensp; [ODL](https://github.com/odlgroup/odl) &ensp; [DPIR](https://github.com/cszn/DPIR) &ensp; [DPHSIR](https://github.com/Zeqiang-Lai/DPHSIR) &ensp; [DGUNet](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration)
