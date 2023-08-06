# Compressed-MRI

The application of Delta-Prox for end-to-end compressed-MRI.

## Prepare Data and Checkpoints

You download the testing datasests from huggingface, [MICCAI_2020](https://huggingface.co/datasets/delta-prox/MICCAI_2020), [Medical_7_2020](https://huggingface.co/datasets/delta-prox/Medical_7_2020), [examples](https://huggingface.co/datasets/delta-prox/examples)

To train the rl solver, you need to download the training dataset, [Image128](https://huggingface.co/datasets/delta-prox/Image128), as well.

The checkpoints is hosted at [aaronb/CSMRI](https://huggingface.co/aaronb/CSMRI).

## Evaluation & training

All scripts can be directly executed, e.g.,

```bash
python deq_tfpnp.py
```

For more configuration args. Use 
```bash
python deq_tfpnp.py --help
```

