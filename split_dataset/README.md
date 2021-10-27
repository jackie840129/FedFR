# Split MS1M dataset to our Federated Learning (FL) dataset

The original MS1M-Arcface [1,2] contains 8.5k IDs.
We split it into two parts. The first half (4.25k IDs) are used to generate the FL dataset. The other half (4.25k IDs) are used to generate pretrained dataset. 

## Prerequisite
You should first download the ms1m dataset from [[here]](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view).

The argument `--data_dir` in the following codes (split_FL.py & split_pretrain.py) is the path to `<PATH/TO/face_emore>`.

## Generate FL dataset
Run the `split_FL.py` to generate dataset.
The meaning of the argument is as follow:
- **`data_dir`** : The path to folder "face_emore". eg. "/home/jackieliu/faces_ms1m-refine-v2_112x112/face_emore"
- **`output_dir`** : Your desired path. eg. "/home/jackieliu/ms1m_split/"
- **`num_client`** : The number of clients in the federated framework. eg. 40 in our work.
- **`num_ID`** : The number of total ID in the federated framework. eg. 4000 in our work.
- **`relabel

## Generate Pretrained dataset



### Reference
[1] Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, Jianfeng Gao. Ms-celeb-1m: A dataset and benchmark for large-scale face recognition. ECCV, 2016.

[2] Jiankang Deng, Jia Guo, Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition, arXiv:1801.07698, 2018.
