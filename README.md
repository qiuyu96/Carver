# Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase

![timeline.jpg](figures/3D_benchmark.jpg)
Figure: Overview of our modularized pipeline for 3D-aware image synthesis, which modularizes the
generation process in a universal way. Each module can be improved independently,
facilitating algorithm development. Note that the discriminator is omitted for simplicity.

> **Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase** <br>
> Qiuyu Wang, Zifan Shi, Kecheng Zheng, Yinghao Xu, Sida Peng, Yujun Shen <br>
> *arXiv: 2306.12423* <br>

[[Paper](https://arxiv.org/abs/2306.12423)]

## Overview of methods supported by our codebase:

<details open>
<summary><b>Supported Methods (9)</b></summary>

> - [x] [![](https://img.shields.io/badge/NeurIPS'2020-GRAF-f4d5b3?style=for-the-badge)](https://github.com/autonomousvision/graf)
> - [x] [![](https://img.shields.io/badge/NeurIPS'2022-EpiGRAF-d0e9ff?style=for-the-badge)](https://github.com/universome/epigraf)
> - [x] [![](https://img.shields.io/badge/CVPR'2021-π&#8211;GAN-yellowgreen?style=for-the-badge)](https://github.com/marcoamonteiro/pi-GAN)
> - [x] [![](https://img.shields.io/badge/CVPR'2021-GIRAFFE-D14836?style=for-the-badge)](https://github.com/autonomousvision/giraffe)
> - [x] [![](https://img.shields.io/badge/CVPR'2022-EG3D-c2e2de?style=for-the-badge)](https://github.com/NVlabs/eg3d)
> - [x] [![](https://img.shields.io/badge/CVPR'2022-GRAM-854?style=for-the-badge)](https://github.com/microsoft/GRAM)
> - [x] [![](https://img.shields.io/badge/CVPR'2022-StyleSDF-123456?style=for-the-badge)](https://github.com/royorel/StyleSDF)
> - [x] [![](https://img.shields.io/badge/CVPR'2022-VolumeGAN-535?style=for-the-badge)](https://github.com/genforce/volumegan)
> - [x] [![](https://img.shields.io/badge/ICLR'2022-StyleNeRF-1223?style=for-the-badge)](https://github.com/facebookresearch/StyleNeRF)
</details>
<details open>
<summary><b>Supported Modules (8)</b></summary>

> - [x] ![](https://img.shields.io/badge/pose_sampler-f4d5b3?style=for-the-badge) Deterministic Pose Sampling, Uncertainty Pose Sampling, ...
> - [x] ![](https://img.shields.io/badge/point_sampler-d0e9ff?style=for-the-badge) Uniform, Normal, Fixed, ...
> - [x] ![](https://img.shields.io/badge/point_embedder-854?style=for-the-badge) Tri-plane, Volume, MLP, MPI, ...
> - [x] ![](https://img.shields.io/badge/feature_decoder-D14836?style=for-the-badge) LeakyReLU, Softplus, SIREN, ReLU, ...
> - [x] ![](https://img.shields.io/badge/volume_renderer-535?style=for-the-badge) Occupancy, Density, Feature, Color, SDF, ...
> - [x] ![](https://img.shields.io/badge/stochasticity_mapper-123456?style=for-the-badge)
> - [x] ![](https://img.shields.io/badge/upsampler-c2e2de?style=for-the-badge)
> - [x] ![](https://img.shields.io/badge/visualizer-1223?style=for-the-badge) color, geometry, ...
> - [x] ![](https://img.shields.io/badge/evaluator-552?style=for-the-badge) FID, ID, RE, DE, PE, ...

</details>

## Installation

Our code is tested with Python 3.8, CUDA 11.3 and PyTorch 1.11.0.

1. Install package requirements via `conda`:

    ```shell
    conda create -n <ENV_NAME> python=3.8  # create virtual environment with Python 3.8
    conda activate <ENV_NAME>
    conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch # install PyTorch 1.11.0
    pip install -r requirements.txt # install dependencies
    ```
2. Our code requires [nvdiffrast](https://nvlabs.github.io/nvdiffrast), so please refer to the [documentation](https://nvlabs.github.io/nvdiffrast/#linux) for instructions on how to install it.

3. Our code also depends on the [face reconstruction model](https://arxiv.org/abs/1903.08527) to evaluate metrics. Please refer to [this guide](https://github.com/sicxu/Deep3DFaceRecon_pytorch#prepare-prerequisite-models) to prepare prerequisite models.

4. To use video visualizer (optional), please also install `ffmpeg`.

    - Ubuntu: `sudo apt-get install ffmpeg`.
    - MacOS: `brew install ffmpeg`.

5. To reduce memory footprint (optional), you can switch to either `jemalloc` (recommended) or `tcmalloc` rather than your default memory allocator.

    - jemalloc (recommended):
        - Ubuntu: `sudo apt-get install libjemalloc`
    - tcmalloc:
        - Ubuntu: `sudo apt-get install google-perftools`

## Preparing datasets

**FFHQ** and **ShapeNet Cars**: Please refer to [this guide](https://github.com/NVlabs/eg3d#preparing-datasets) to prepare the datasets.

**Cats**: Please refer to [this guide](https://github.com/microsoft/GRAM#data-preparation) to prepare the dataset.

## Quick Demo

### Train [EG3D](https://nvlabs.github.io/eg3d/) on FFHQ in Resolution of 515x512

In your Terminal, run:

```shell
./scripts/training_demos/eg3d_ffhq512.sh <NUM_GPUS> <PATH_TO_DATA> [OPTIONS]
```

where

- `<NUM_GPUS>` refers to the number of GPUs. Setting `<NUM_GPUS>` as 1 helps launch a training job on single-GPU platforms.

- `<PATH_TO_DATA>` refers to the path of FFHQ dataset (in resolution of 256x256) with `zip` format. If running on local machines, a soft link of the data will be created under the `data` folder of the working directory to save disk space.

- `[OPTIONS]` refers to any additional option to pass. Detailed instructions on available options can be shown via `./scripts/training_demos/eg3d_ffhq512.sh <NUM_GPUS> <PATH_TO_DATA> --help`.

This demo script uses `eg3d_ffhq512` as the default value of `job_name`, which is particularly used to identify experiments. Concretely, a directory with name `job_name` will be created under the root working directory (with is set as `work_dirs/` by default). To prevent overwriting previous experiments, an exception will be raised to interrupt the training if the `job_name` directory has already existed. To change the job name, please use `--job_name=<NEW_JOB_NAME>` option.

Other 3D GAN models reproduced by our codebase can be trained similarily, please refer to scripts under `./scripts/training_demos/` for more details.

### Ablation `point embedder` using our codebase.

To investigate the effect of various point embedders, one can utilize the following command to train the models.

#### MLP-based

```shell
./scripts/training_demos/ablation3d.sh <NUM_GPUS> <PATH_TO_DATA> --job_name <YOUR_JOB_NAME> --root_work_dir <YOUR_ROOT_DIR> --ref_mode 'coordinate' --use_positional_encoding false --mlp_type 'stylenerf' --mlp_depth 16 --mlp_hidden_dim 128 --mlp_output_dim 64 --r1_gamma 1.5
```

#### Volume-based

```shell
./scripts/training_demos/ablation3d.sh <NUM_GPUS> <PATH_TO_DATA> --job_name <YOUR_JOB_NAME> --root_work_dir <YOUR_ROOT_DIR> --ref_mode 'volume' --fv_feat_res 64 --use_positional_encoding false --mlp_type 'stylenerf' --mlp_depth 16 --mlp_hidden_dim 128 --mlp_output_dim 64 --r1_gamma 1.5
```

#### Tri-plane-based

```shell
./scripts/training_demos/ablation3d.sh <NUM_GPUS> <PATH_TO_DATA> --job_name <YOUR_JOB_NAME> --root_work_dir <YOUR_ROOT_DIR> --ref_mode 'triplane' --fv_feat_res 64 --use_positional_encoding false --mlp_type 'eg3d' --mlp_depth 2 --mlp_hidden_dim 64 --mlp_output_dim 32 --r1_gamma 1.5
```

## Inspect Training Results

Besides using TensorBoard to track the training process, the raw results (e.g., training losses and running time) are saved in [JSON Lines](https://jsonlines.org/) format. They can be easily inspected with the following script

```python
import json

file_name = '<PATH_TO_WORK_DIR>/log.json'

data_entries = []
with open(file_name, 'r') as f:
    for line in f:
        data_entry = json.loads(line)
        data_entries.append(data_entry)

# An example of data entry
# {"Loss/D Fake": 0.4833524551040682, "Loss/D Real": 0.4966000154727226, "Loss/G": 1.1439273656869773, "Learning Rate/Discriminator": 0.002352941082790494, "Learning Rate/Generator": 0.0020000000949949026, "data time": 0.0036810599267482758, "iter time": 0.24490128830075264, "run time": 66108.140625}
```

## Inference for visualization
After training a model, one can employ the following scripts to run inference and visualize the results, including images, videos, and geometries.
```shell
CUDA_VISIBLE_DEVICES=0 python test_3d_inference.py --model <PATH_TO_MODEL> --work_dir <PATH_TO_WORK_DIR> --save_image true --save_video false --save_shape true --shape_res 512 --num 10 --truncation_psi 0.7
```

## Evaluate Metrics
After training a model, one can use the following scripts to evaluate various metrics, including FID, face identity consistency (ID), depth error (DE), pose error (PE) and reprojection error (RE).

```shell
python -m torch.distributed.launch --nproc_per_node=1 test_3d_metrics.py --dataset <PATH_TO_DATA> --model <PATH_TO_MODEL>  --test_fid true --align_face true --test_identity true --test_reprojection_error true --test_pose true --test_depth true --fake_num 1000
```

## TODO
- [] Upload pretrained checkpoints
- [] User Guide

## Acknowledgement

This repository is built upon [Hammer](https://github.com/bytedance/Hammer). On top of [Hammer](https://github.com/bytedance/Hammer), we reimplement [GRAF](https://github.com/autonomousvision/graf), [GIRAFFE](https://github.com/autonomousvision/giraffe), [π-GAN](https://github.com/marcoamonteiro/pi-GAN), [StyleSDF](https://github.com/royorel/StyleSDF), [StyleNeRF](https://github.com/facebookresearch/StyleNeRF), [VolumeGAN](https://github.com/genforce/volumegan), [GRAM](https://github.com/microsoft/GRAM), [EpiGRAF](https://github.com/universome/epigraf) and [EG3D](https://github.com/NVlabs/eg3d).


## BibTeX

```bibtex
@article{wang2023benchmarking,
  title   = {Benchmarking and Analyzing 3D-aware Image Synthesis with a Modularized Codebase},
  author  = {Wang, Qiuyu and Shi, Zifan and Zheng, Kecheng and Xu, Yinghao and Peng, Sida and Shen, Yujun},
  journal = {arXiv:2306.12423},
  year    = {2023}
}
```