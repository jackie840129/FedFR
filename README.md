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
3. After training, you will have models saved in the checkpoint directory, eg., `ckpt/FedFR/`.

## Generic Evaluation

You will evaluate the generic performance on IJBC dataset.

You can use `ijbc_conti.py` to continuously evaluate all checkpoints saved in the directory. (eg. epoch 5 to epoch 11)
```
python3 ijbc_conti.py --root_path PATH/TO/IJBC/ --ckpt_dir ckpt/FedFR --epoch 5 6 7 8 9 10 11 \
        --gpu 0 1 2 3 --job 'both'
```
There are two types of job, '1:1' and '1:n', as described in our paper.
If you want to evaluate both, you can use `--job 'both'`.

## Personalized Evaluation
