# FedFR: Joint Optimization Federated Framework for Generic and Personalized Face Recognition

[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20057/19816) [[arXiv]](https://arxiv.org/abs/2112.12496)

[Chih-Ting Liu](https://jackie840129.github.io/), Chien-Yi Wang, Shao-Yi Chien, Shang-Hong Lai, <br/>Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2022

## Generate FedFR Dataset

You can follow the steps in [split_dataset](split_dataset) to generate our pretrained, and FL dataset.

## Prerequisite

1. Put the pretrained model (["backbone.pth"](https://drive.google.com/file/d/19d-Qm-RkBh9E2P1o_ZbdrHAyoZocFZbK/view?usp=sharing)) under the `pretrain/` folder.

## Training our FedFR
1. In `config.py`, you should first change the path of `config.rec` and `config.local_rec`
2. Run with command `sh run.sh`.

## Generic Evaluation


## Personalized Evaluation
