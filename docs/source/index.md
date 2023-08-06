---
hide-toc: true
---

<!-- # Welcome to ∇-Prox -->
<br/>
<p align="center">
<a href="https://light.princeton.edu/publication/delta_prox/">
    <img src="_static/logo.svg" alt="" width="22%">
</a> 
</p>


```{toctree}
:maxdepth: 3
:hidden: 

Get Started <started/index>
tutorials/index
api/index
citation
```


```{toctree}
:caption: Useful Links
:hidden:
PyPI Page <https://pypi.org/project/dprox/>
GitHub Repository <https://github.com/princeton-computational-imaging/Delta-Prox>
Project Page <https://light.princeton.edu/publication/delta_prox/>
Paper <https://dl.acm.org/doi/abs/10.1145/3592144>
```

<br/>

🎉  ∇-Prox is a domain-specific language (DSL) and compiler that transforms optimization problems into differentiable proximal solvers. 
<br/>
🎉  ∇-Prox allows for rapid prototyping of learning-based bi-level optimization problems for a diverse range of applications, by [optimized algorithm unrolling](https://pypi.org/project/dprox/), [deep equilibrium learning](https://pypi.org/project/dprox/), and [deep reinforcement learning](https://pypi.org/project/dprox/). 

The library includes the following major components:

- A library of differentiable [proximal algorithms](api/algo), [proximal operators](api/proxfn), and [linear operators](api/linop).
- Interchangeable [specialization](api/primitive) strategies for balancing trade-offs between speed and memory.
- Out-of-the-box [training utilities](api/primitive) for learning-based bi-level optimization with a few lines of code.

```{nbgallery}
```

<div class="toctree-wrapper compound">
<div class="nbsphinx-gallery">
<a class="reference internal" href="started/quicktour.html">
  <b>Quicktour</b>
  <p style="color:var(--color-content-foreground)">Learn the fundamentals of using ∇-Prox. We recommend starting here if you are using 🎉 ∇-Prox for the first time! </p>
</a>
<a class="reference internal" href="tutorials/index.html">
  <b>Tutorials</b>
  <p style="color:var(--color-content-foreground)">Understand the design of the library and the mathematics behind the code. </p>
</a>
<a class="reference internal" href="api/index.html">
  <b>API Documentation</b>
  <p style="color:var(--color-content-foreground)">Explore the complete reference guide. Useful if you want to develop programs with ∇-Prox. </p>
</a>
</div>
</div>