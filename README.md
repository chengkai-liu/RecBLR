# Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation


> **Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation (CIKM 2024)**\
> Chengkai Liu, Jianghao Lin, Hanzhou Liu, Jianling Wang, James Caverlee\
> Paper: https://arxiv.org/abs/2406.12580

## Usage


### Requirements

* Python >= 3.7
* PyTorch >= 1.12
* CUDA >= 11.6
* Triton >= 2.2
* Install RecBole:
  * `pip install recbole`
* [optional] Install causal Conv1d with CUDA optimization for faster computation of Conv1D:
  * `pip install causal-conv1d>=1.2.0`

### Run

```
python run.py
```


Please update `config.yaml` to adjust the hyperparameters and experimental settings.

## Citation

```bibtex
@article{liu2024behavior,
  title={Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation},
  author={Liu, Chengkai and Lin, Jianghao and Liu, Hanzhou and Wang, Jianling and Caverlee, James},
  journal={arXiv preprint arXiv:2406.12580},
  year={2024}
}
```

## Acknowledgment

This project references [RecBole](https://github.com/RUCAIBox/RecBole), [Accelerated Scan](https://github.com/proger/accelerated-scan) and [Causal-Conv1d](https://github.com/Dao-AILab/causal-conv1d). We appreciate their outstanding work and commitment to open source.
