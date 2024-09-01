# HandFormer

PyTorch implementation of "On the Utility of 3D Hand Poses for Action Recognition", ECCV 2024.

[[arXiv](https://arxiv.org/pdf/2403.09805)][[Project Page](https://s-shamil.github.io/HandFormer/)]

<img src="docs/images/teaser.png" width="90%">

# Requirements

### Environment

Create a conda environment with the necessary packages.
```
$ conda create -n handformer_env --file package-list.txt
```

### Data Preprocessing

Set the parameters and data directories in `Preprocess/config.py` and run the following:

```
$ python Preprocess/1_get_clip_data.py
$ python Preprocess/2_get_final_data.py
```
$${\color{red}TODO}$$	: Upload and refer to preprocessed and additional source data.
