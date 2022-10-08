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

Each model trained by client before FedAvg model aggregation will be used to evaluate the personalized performance.

Furthermore, as described in paper, the backbone and the tranformation layer will be concatenated to generate personalized features.

The evaluation scripts are as follows:

We only provide single checkpoint for some epoch and single type of evaluation ('1:1' or '1:n')

### 1:1 Evaluation
```
python3 local_all.py --backbone 'multi' --task '1:1' --ckpt_path ckpt/FedFR \
                     --data_dir $VERI_DIR --gallery $GALLERY_DIR --epoch 11 --num_client 40 --gpu 0 1 2 3
```
- `$VERI_DIR` is the path to 'local_veri_4000' when you split your dataset, eg. '/home/jackieliu/face_recognition/ms1m_split/local_veri_4000'
- `$GALLERY_DIR` is the path to 'local_gallery_4000', eg. '/home/jackieliu/face_recognition/ms1m_split/local_gallery_4000'

### 1:n Evaluation
```
python3 local_all.py --backbone 'multi' --task '1:n' --ckpt_path ckpt/FedFR \
                     --data_dir $VERI_DIR --gallery $GALLERY_DIR --epoch 11 --num_client 40 --gpu 0 1 2 3
```
- `$VERI_DIR` is the path to 'local_veri_4000' when you split your dataset, eg. '/home/jackieliu/face_recognition/ms1m_split/local_veri_4000'
- `$GALLERY_DIR` is the path to 'local_gallery_4000', eg. '/home/jackieliu/face_recognition/ms1m_split/local_gallery_4000'

