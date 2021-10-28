from easydict import EasyDict as edict

config = edict()
config.dataset = "ms1m_split"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.lr = 0.05  # batch size is 512  #arg
config.step = [6,14] #arg

if config.dataset == "ms1m_split":
    config.rec = "/home/jackieliu/MS-intern/face_recognition/ms1m_split/"
    # config.val_rec =  "/lssd1/face_recognition/val"
    config.val_rec =  "/home/jackieliu/MS-intern/face_recognition/val"
    config.local_rec = "/home/jackieliu/MS-intern/face_recognition/ms1m_split/local_veri_4000"
    config.num_epoch = 16  #arg
    # config.val_targets = ["agedb_30",'cfp_fp']
    config.val_targets = ['agedb_30']

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in config.step if m - 1 <= epoch])
    config.lr_func = lr_step_func
    config.com_batch_size = 256
    config.public_batch_size = 512
    config.HN_threshold = 0.4
    config.train_decay = 8
    config.mu = 5
    config.converter_layer = 1

