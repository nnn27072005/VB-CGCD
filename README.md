<div align="center">

<h2>Continual Generalized Category Discovery:

Learning and Forgetting from a Bayesian Perspective</h2>

[Hao Dai](https://github.com/daihao42), [Jagmohan Chauhan](https://sites.google.com/view/jagmohan-chauhan/home)

University College London

University of Southampton

[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>
<div align="center">
<img src="figures/framework.png" alt="Visualization">
</div>

## Usage

### Installation

```
conda create -n vbcgcd python=3.12.2
conda activate vbcgcd
pip install -r requirements.txt
mkdir datasets
```

## Prepare Datasets

```
python feature_extractor/dino-cifar100.py --finetuned --output_dir datasets/cifar100
```

For GC10-DET, download and extract the Kaggle dataset first:

```
kaggle datasets download -d alex000kim/gc10det -p datasets/raw/gc10det --unzip
python feature_extractor/dino-gc10det.py --raw_data_dir datasets/raw/gc10det --finetuned --output_dir datasets/gc10det
```

## Training

```
python main.py --base 50 --increment 10 --pretrained_model_name dino-vitb16-sl --data_dir datasets/cifar100 --trail_name mix_increment_mngmm_dinovb16_sl_cifar_100
```

GC10-DET training:

```
python main.py --base 5 --increment 1 --pretrained_model_name dino-vitb16-sl --dataset gc10det --data_dir datasets/gc10det --num_classes 10 --trail_name mix_increment_mngmm_dinovb16_sl_gc10det
```

## Citation

If you find our work useful, please cite our related paper:

```
# ICML 2025
@inproceedings{dai2025vbcgcd,
  title={Continual Generalized Category Discovery: Learning and Forgetting from a Bayesian Perspective},
  author={Dai, Hao and Chauhan, Jagmohan},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}

```
