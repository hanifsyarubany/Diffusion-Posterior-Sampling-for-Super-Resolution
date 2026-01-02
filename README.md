# [AI618 Final Project 1] Noisy Inverse Problems

## Abstract
In this task, you have to implement diffusion sampling and posterior sampling methods for upsampling 64x64 FFHQ images. The locations where you are required to complete are marked as comments.


## Prerequisites
- python 3.8 or higher

- pytorch 1.11.0 or higher

- CUDA 11.3.1 or higher

- nvidia-docker (if you use GPU in docker container)


<br />

## Getting started 

### 1) Set environment
### [Option 1] Local environment setting

Install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

<br />

### [Option 2] Build Docker image

Install docker engine, GPU driver and proper cuda before running the following commands.

Dockerfile already contains command to clone external codes. You don't have to clone them again.

--gpus=all is required to use local GPU device (Docker >= 19.03)

```
docker build -t dps-docker:latest .
docker run -it --rm --gpus=all dps-docker
```

<br />

### 4) Inference

```
python3 sample.py \
--data_dir=./downsized_4x \
--save_dir=./results;
```

### 5) Evaluation

Before you submit to kaggle, you have to **encode images to base64 and save them as csv files** using `encode.py`.

```
python3 encode.py \
--img_dir=results/ \
--sub_filename=submission.csv;
```

## Notes
- Please ensure your code is executable. Non-executable code may negatively impact your grade.
- Please write a thorough ₩requirements.txt₩ with all dependencies pinned. 