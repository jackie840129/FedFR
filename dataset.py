import numbers
import os
import os.path as osp
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import config as cfg
import logging

logger = logging.getLogger('FL_face.dataset')
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class All_Client_Dataset(object):
    def __init__(self,root_dir,local_rank,args):
        self.args = args
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.num_client = args.num_client
        self.batch_size = args.batch_size
        self.dataset_dir  = osp.join(self.root_dir,'split_train_i4000c%04d'%(self.num_client))
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        ###
        self.test_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        # create train dataset for each client
        self.creating_each_client()
        # create infer dataset for each client
        self.creating_infer_each_client()
        if args.add_pretrained_data:
            self.creating_public_dataset(train=True)
            self.creating_public_dataset(train=False)
    def creating_each_client(self):
        self.train_loaders = []
        self.train_dataset_sizes = []
        self.train_class_sizes = []
        for c in range(self.num_client):
            client_dir = osp.join(self.dataset_dir,'client_%04d'%(c))
            dataset = MXFaceDataset_Split(root_dir=client_dir,local_rank=self.local_rank,transform=self.transform)
            train_loader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True,num_workers=2,pin_memory=True,drop_last=False)
            self.train_loaders.append(train_loader)
            self.train_dataset_sizes.append(len(dataset))
            self.train_class_sizes.append(dataset.num_classes)

        logger.info('--------------------')
        logger.info('Num_clients : %d'%self.num_client)
        logger.info('Train dataset class sizes /client: %d'%self.train_class_sizes[0])
        logger.info('Total ID : %d'%sum(self.train_class_sizes))
        logger.info('--------------------')

    def creating_infer_each_client(self):
        self.test_loaders = []
        for c in range(self.num_client):
            client_dir = osp.join(self.dataset_dir,'client_%04d'%(c))
            dataset = MXFaceDataset_Split(root_dir=client_dir,local_rank=self.local_rank,transform=self.test_transform)
            test_loader = DataLoader(dataset=dataset, batch_size=256,num_workers=2,pin_memory=False,drop_last=False)
            self.test_loaders.append(test_loader)
        logger.info('--------------------')
        logger.info('Creating Client Infer dataset')
        logger.info('--------------------')

    def creating_public_dataset(self,train=True):
        self.public_dataset_dir  = osp.join(self.root_dir,'split_pretrain_6000','client_0000')
        if train == True:
            train_dataset = MXFaceDataset_Split(root_dir=self.public_dataset_dir, local_rank=self.local_rank, transform=self.transform)
            self.public_train_loader = DataLoader(train_dataset,batch_size=cfg.public_batch_size,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
            logger.info('--------------------')
            logger.info('Public train dataset class sizes : %d'%train_dataset.num_classes)
            logger.info('--------------------')
        else:
            test_dataset = MXFaceDataset_Split(root_dir=self.public_dataset_dir, local_rank=self.local_rank, transform=self.test_transform)
            self.public_test_loader = DataLoader(dataset=test_dataset,batch_size=cfg.public_batch_size,num_workers=8,pin_memory=True,drop_last=False)
            logger.info('--------------------')
            logger.info('Public test dataset class sizes : %d'%test_dataset.num_classes)
            logger.info('--------------------')

class MXFaceDataset_Subset(Dataset):
    def __init__(self, imgrec, imgidx,num_classes, relabel_dict, transform):
        super(MXFaceDataset_Subset, self).__init__()
        self.imgrec = imgrec
        self.imgidx = imgidx
        self.num_classes = num_classes
        self.relabel_dict = relabel_dict
        # transform
        self.transform = transform
    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = self.relabel_dict[label]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class MXFaceDataset_Combine(Dataset):
    def __init__(self,first_dataset,second_dataset):
        super(MXFaceDataset_Combine,self).__init__()
        self.first_dataset = first_dataset
        self.second_dataset = second_dataset
        self.first_nclass = first_dataset.num_classes
        self.first_len = len(first_dataset)
        self.second_len = len(second_dataset)
        self.num_class = first_dataset.num_classes + second_dataset.num_classes
    def __getitem__(self,idx):
        if idx < self.first_len:
            img,label = self.first_dataset[idx]
            return img,label
        else:
            img,label  = self.second_dataset[idx-self.first_len]
            return img,label+self.first_nclass
    def __len__(self):
        return self.first_len+self.second_len


class MXFaceDataset_Split(Dataset):
    def __init__(self, root_dir, local_rank,transform):
        super(MXFaceDataset_Split, self).__init__()
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # process header
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = [int(i) for i in header.label]
            # (int(header.label[0]), int(header.label[1]), int(header.label[2]))
            self.imgidx = np.array(range(1, self.header0[0]))
            self.num_classes = self.header0[1]
            if len(self.header0) == 3:
                self.ID_base = self.header0[2]
            else:
                self.ID_base = None
        else:
            raise NotImplementedError()
        # transform
        self.transform = transform
    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank,transform):
        super(MXFaceDataset, self).__init__()
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # process header
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        # transform
        self.transform = transform
        self.num_classes = None
        # for KD
        self.rand = list(np.random.permutation(len(self.imgidx)))
    def __getitem__(self, index):
        idx = self.imgidx[self.rand[index]]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,label
    def __len__(self):
        return len(self.imgidx)

if __name__ == '__main__':

    dataset = MXFaceDataset_Split(root_dir='/lssd1/face_recognition/ms1m_split/split_0.20_c0001/client_0000',local_rank=0,transform=None)
    print(dataset.num_classes)
    exit(-1)
    for i in range(1000):
        dataset = MXFaceDataset(root_dir='/lssd1/face_recognition/ms1m_split/split_1000.00_c1000/client_%04d'%i,local_rank=0,transform=None)
        # dataset = MXFaceDataset(root_dir='/lssd1/face_recognition/faces_glintasia/',local_rank=0)
        # dataset.transform = None
        print(dataset.num_classes,len(dataset))
        import cv2
        # seen = set()
        output_dir = './visualization/client_%04d/'%(i)
        if not os.path.exists(output_dir):
            os.system('mkdir -p %s'%output_dir)
            # os.mkdir(output_dir)
        for j in range(len(dataset)):
            img,label = dataset[j]
            # continue
            cv2.imwrite(os.path.join(output_dir,'%04d.jpg'%(j)),img[:,:,::-1])
            continue
            if int(label.item()) not in seen:
                cv2.imwrite(os.path.join(output_dir,'%04d.png'%(j)),img[:,:,::-1])
                # seen.add(int(label.item()))
                # if len(seen) > 30 :
                    # break
    

