import dprox.utils.huggingface as hf

hf.download_dataset('Medical_7_2020', local_dir='data/Medical7_2020')
hf.download_dataset('MICCAI_2020', local_dir='data/MICCAI_2020')
