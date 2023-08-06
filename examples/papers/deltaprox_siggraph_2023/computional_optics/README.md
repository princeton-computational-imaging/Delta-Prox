# End-to-End Computational Optics

The application of Delta-Prox for end-to-end computational optics. 

## Prepare Data and Checkpoints

- Download [BSD500](https://huggingface.co/datasets/delta-prox/BSD500) for training, and [CBSD68](https://huggingface.co/datasets/delta-prox/CBSD68) and [Urban100](https://huggingface.co/datasets/delta-prox/Urban100) for evaluation.
- The pretrained models are hosted at [Huggingface](https://huggingface.co/delta-prox/computational_optics).

## Training 

- Training Delta-Prox

```bash
python e2e_optics_dprox.py train
```

- Training DeepOpticsUnet

```bash
python e2e_optics_unet.py train
```

## Evaluation

- Evaluate Delta-Prox

```bash
python e2e_optics_dprox.py test
```

- Evaluate DeepOpticsUnet

```bash
python e2e_optics_unet.py test
```

- Evaluate DPIR

```bash
python pnp_optics.py
```