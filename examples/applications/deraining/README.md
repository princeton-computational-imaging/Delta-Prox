# Image Deraining

The application of Delta-Prox for image-deraining.

## Prepare Data and Checkpoints

- Download test data from [Google drive](https://drive.google.com/file/d/1P_-RAvltEoEhfT-9GrWRdpEi6NSswTs8/view?usp=sharing
). Put it into the `datasets/test` folder. For example, use `datasets/test/Rain100H`.

- Download the checkpoints from [Huggingface]().

## Evaluation

- Unrolled Proximal Gradient Descent with shared parameters and restormer as initializer.

```bash
python test_unroll_share.py
python 
```

- Unrolled Proximal Gradient Descent with unshared parameters

```bash
python test_unroll.py
python evaluate_PSNR_SSIM.py 
```

> To obtain the paper results, please use the matlab script `evaluate_PSNR_SSIM.m`.

## Acknowledgement

- [Restormer](https://github.com/swz30/Restormer) 
- [DGUNet](https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration)