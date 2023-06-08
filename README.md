<p align="center">
<a href="https://deltaprox.readthedocs.io/"><img src="https://user-images.githubusercontent.com/26198430/229363778-a546ffce-b43d-4f4a-9869-629208f050cc.svg" alt="" width="30%"></a> &ensp; 
</p>

<p align="center">
Differentiable Proximal Algorithm Modeling for Large-Scale Optimization
</p>

<p align="center">
<a href="https://deltaprox.readthedocs.io/">Docs</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx/tree/master/tutorials">Tutorials</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx/tree/master/examples">Examples</a> |
<a href="#">Paper</a> |
<a href="https://github.com/Zeqiang-Lai/DeltaProx#citation">Citation</a> 
</p>

<p align="center">
    <a href="[https://github.com/PanQiWei/AutoGPTQ/releases](https://pypi.org/project/dprox/)">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/dprox">
    </a>
    <a href="[https://pypi.org/project/auto-gptq/](https://arxiv.org/abs/2207.02849)">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg">
    </a>
</p>

<!-- <a href="https://pypi.org/project/dprox/">![Version](https://img.shields.io/pypi/v/dprox)</a> -->
<!-- <a href="https://arxiv.org/abs/2207.02849">![arXiv](https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg)</a> -->

```bash
pip install dprox
```

> $\nabla$-Prox is a domain-specific language (DSL) and compiler that transforms optimization problems into differentiable proximal solvers. Departing from handwriting these solvers and differentiating via autograd, $\nabla$-Prox requires only a few lines of code to define a solver that can be *specialized based on user requirements w.r.t memory constraints or training budget* by optimized algorithm unrolling, deep equilibrium learning, and deep reinforcement learning. $\nabla$-Prox makes it easier to prototype different learning-based bi-level optimization problems for a diverse range of applications. We compare our framework against existing methods with naive implementations. $\nabla$-Prox is significantly more compact in terms of lines of code and compares favorably in memory consumption in applications across domains.

## News


- **[Jan 21 2023]** :  Release preview code.

## Getting Started

♨️ Here is an example for using  $\nabla$-Prox. For more comprehensive usage, please refer to the [Documentation]().

### Solver Compliation

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

Conventional imaging systems employ compound refractive lens systems that are typically hand-engineered for image quality in isolation of the downstream camera task. Departing from this design paradigm, a growing body of work in computational photography [Haim et al. 2018; Horstmeyer et al. 2017] has explored the design of specialized lens system with diffractive optical elements (DOEs). 


As a specific example, we consider end-to-end computational optics that jointly optimize a diffractive optical element (DOE) and an image reconstruction algorithm,  where the observation $y$ is obtained by convolving a clear image $x$ by the point spread function (PSF) of DOE as,

$$
    y =  D\left(x;\, \theta_{DOE} \right) + \epsilon, 
$$

where  $D(\cdot; \theta_{DOE})$ indicates a shift-invariant convolution process with an optical kernel, i.e., PSF, derived from a DOE image formation model parameterized by $\theta_{DOE}$, and $\epsilon$ is measurement noise, e.g., Poissionian-Gaussian noise. 
To reconstruct target image $x$ from noise-contaminated measurements $y$, we minimize the sum of a data-fidelity $f$ and regularizer term $r$ as

$$
    min_{x \in R^n} ~ f \left( D\left(x;\, \theta_{DOE} \right) \right) + r \left(x ; \, \theta_r \right).
$$

```python
# generate input
gt = gt.to(device).float()
psf = rgb_collim_model.get_psf()
inp = img_psf_conv(gt, psf, circular=True)
inp_dprox = inp + torch.randn(*inp.shape, device=inp.device) * sigma

# build solver
x = Variable()
y = Placeholder()
PSF = Placeholder()
data_term = sum_squares(conv_doe(x, PSF, circular=True), y)
reg_term = deep_prior(x, denoiser='ffdnet_color')
solver = compile(data_term + reg_term, method='admm')

# solve the problem
y.value = inp_dprox
PSF.value = psf
with torch.no_grad():
    out_dprox = solver.solve(x0=inp_dprox,
                        rhos=rgb_collim_model.rhos,
                        lams={reg_term: rgb_collim_model.sigmas},
                        max_iter=max_iter)

```

<img width="500" alt="image" src="https://github.com/princeton-computational-imaging/Delta-Prox/assets/26198430/384d422c-c984-4209-8358-51f4594f431c">
</br>
<img width="500" alt="image" src="https://github.com/princeton-computational-imaging/Delta-Prox/assets/26198430/628665d0-b2ac-48c7-9c9f-0216db4807ca">


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

## Acknowledgement

[ProxImaL]() 
