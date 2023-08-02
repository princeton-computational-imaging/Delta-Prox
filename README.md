<p align="center">
<a href="https://light.princeton.edu/publication/delta_prox/">
    <img src="https://github.com/princeton-computational-imaging/Delta-Prox/assets/26198430/8e81717a-977c-4dd6-bc8b-7cba975690b8" alt="" width="33%">
<!--     <img src="https://github.com/princeton-computational-imaging/Delta-Prox/assets/26198430/9c4d10a2-7d15-4f25-83b7-c94b442a8347" alt="" width="30%"> -->
    </a> &ensp; 
</p>



<p align="center">
Differentiable Proximal Algorithm Modeling for Large-Scale Optimization
</p>

<p align="center">
<!-- <a href="">Project Page</a> | -->
<a href="#">Paper</a> |
<a href="https://deltaprox.readthedocs.io/">Docs</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx/tree/master/tutorials">Tutorials</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx/tree/master/examples">Examples</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx#citation">Citation</a> 
</p>

<p align="center">
    <a href="[https://pypi.org/project/dprox/](https://pypi.org/project/dprox/)">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/dprox">
    </a>
    <a href="[https://pypi.org/project/auto-gptq/](https://arxiv.org/abs/2207.02849)">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg">
    </a>
</p>

<!-- <a href="https://pypi.org/project/dprox/">![Version](https://img.shields.io/pypi/v/dprox)</a> -->
<!-- <a href="https://arxiv.org/abs/2207.02849">![arXiv](https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg)</a> -->



> $\nabla$-Prox is a domain-specific language (DSL) and compiler that transforms optimization problems into differentiable proximal solvers. Departing from handwriting these solvers and differentiating via autograd, $\nabla$-Prox requires only a few lines of code to define a solver that can be *specialized based on user requirements w.r.t memory constraints or training budget* by optimized algorithm unrolling, deep equilibrium learning, and deep reinforcement learning. $\nabla$-Prox makes it easier to prototype different learning-based bi-level optimization problems for a diverse range of applications. We compare our framework against existing methods with naive implementations. $\nabla$-Prox is significantly more compact in terms of lines of code and compares favorably in memory consumption in applications across domains.

## News


- **Jun 2023** :  Release preview code.
- **May 2023** : 🎉 $\nabla$-Prox is accepted by the journal track of SIGGRAPH 2023.

## Installtion

We recommend installing 🍕 $\nabla$-Prox in a virtual environment from PyPI.

```bash
pip install dprox
```

Please refer to the [Installtion]() guide for other options.

## Quickstart
<!-- ![pipeline](https://github.com/princeton-computational-imaging/Delta-Prox/assets/26198430/544a0972-f911-4976-8228-a5aa6de319c8) -->
![pipeline2](https://github.com/princeton-computational-imaging/Delta-Prox/assets/26198430/6cef8951-7671-4d7a-88dd-e4e6d508fe09)


Consider a simple image deconvolution problem, where we seek to find a clean image $x$ given the blurred observation $y$ that minimizes the following objective function:

$$
\arg \min_x { \frac{1}{2} |Dx - y|^2_2 + g(x) }\,,
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

We can also specialize the solver that adapts for learning-based bi-level optimization. 
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

Want to learn more? Check out the [documentation]() or have a look at our [tutorials]().

## Citation

```bibtex
@article{deltaprox2023,
  title={∇-Prox: Differentiable Proximal Algorithm Modeling for Large-Scale Optimization},
  author={Lai, Zeqiang and Wei, Kaixuan and Fu, Ying and Härtel, Philipp and Heide, Felix},
  journal={ACM Transactions on Graphics},
  year={2023},
}
```

## Acknowledgement

[ProxImaL](https://github.com/comp-imaging/ProxImaL) &ensp; [ODL](https://github.com/odlgroup/odl) &ensp; [DPIR](https://github.com/cszn/DPIR) &ensp; [DPHSIR](https://github.com/Zeqiang-Lai/DPHSIR) &ensp; [DGUNet](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration)
