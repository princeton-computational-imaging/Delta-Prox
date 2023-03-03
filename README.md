# 🎉  Delta Prox

A differentiable modeling language for proximal algorithms on large-scale optimization.


<a href="#/">Docs</a> |
<a href="#">Tutorials</a> |
<a href="#">Examples</a> |
<a href="#">Paper</a> |
<a href="#">Citation</a> 


<a href="https://pypi.org/project/dprox/">![Version](https://img.shields.io/pypi/v/dprox)</a>
![GitHub](https://img.shields.io/github/license/princeton-computational-imaging/Delta-Prox)
  <a href="https://arxiv.org/abs/2207.02849">![arXiv](https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg)</a>


```bash
pip install dprox
```

**Features**

- ∇-Prox allows users to specify optimization objective functions of unknowns concisely at a high level, and intelligently compiles the problem into compute and memory efficient differentiable solvers.


> 🚧 The code is still under construction, more features would be migrated from our dev code.

## News
 
- **[Jan 21 2023]**  Release preview code.

## Getting Started

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


Please refer to the [documentation]() site for more instructions on the efficient differentiation of proximal algorithm with ∇-Prox.

## Citation

```bibtex
@article{deltaprox2023,
  title={∇-Prox: Differentiable Proximal Algorithm Modeling for Large-Scale Optimization},
  author={Lai, Zeqiang and Wei, Kaixuan and Fu, Ying and HÄRTEL, PHILIPP and HEIDE, FELIX},
  journal={arXiv preprint arXiv:2301.11525},
  year={2023},
}
```

## License

Delta Prox is licensed under the MIT License.
