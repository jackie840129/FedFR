import os
import mxnet as mx
import numpy as np 
from torch.utils.data import DataLoader,Dataset
import time
from tqdm import tqdm,trange
from collections import defaultdict,OrderedDict
import pickle
import argparse
import random
import cv2
import pandas as pd
random.seed(100)
np.random.seed(100)

class MXFaceDataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        path_imgidx = os.path.join(root_dir,'train.idx')
        path_imgrec = os.path.join(root_dir,'train.rec')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx,path_imgrec,'r')
        s = self.imgrec.read_idx(0)
        header, header_img = mx.recordio.unpack(s)
        self.header_img = header_img
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
    def __getitem__(self,index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header , _ = mx.recordio.unpack(s)
        label = header.label
        return label
    def __len__(self):
        return len(self.imgidx) 

def collate_fn(batch):
    return batch

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default='/lssd1/face_recognition/faces_ms1m-refine-v2_112x112/faces_emore/')
    parser.add_argument('--output_dir',default='/lssd1/face_recognition/ms1m_split/')
    parser.add_argument('--num_client',type=int,default=1)
    parser.add_argument('--num_ID',type=int,default=10)
    args = parser.parse_args()
    print(args)

    print('- Loading original dataset %s...'%(args.data_dir.split('/')[-1]))
    dataset = MXFaceDataset(root_dir=args.data_dir)
    output_dir = args.output_dir
    os.system('mkdir -p %s'%(output_dir))
    
    if not os.path.exists(os.path.join(output_dir,'ID2idx.pickle')):
        print('- ID2idx pickle not found, creating...')
        save_id_loader = DataLoader(dataset,batch_size=128,collate_fn=collate_fn,shuffle=False,num_workers=6)
        ID_dict = defaultdict(list)

        pbar = tqdm(total = len(save_id_loader),ncols=140)
        idx = 1
        for data in save_id_loader:
            for d in data:
                ID_dict[d].append(idx)
                idx+=1
            pbar.update(1)
        pbar.close()

        with open(os.path.join(output_dir,'ID2idx.pickle'),'wb') as f:
            pickle.dump(ID_dict,f,protocol=5)
    else:
        print('- ID2idx pickle found ! loading...')
        with open(os.path.join(output_dir,'ID2idx.pickle'),'rb') as f:
            ID_dict = pickle.load(f)
    # select subset to create DS
    IDs = list(ID_dict.keys())
    random.shuffle(IDs)
    
    samp_IDs = []
    # for FL sample first half
    candidate_IDs = IDs[:int(len(IDs)*0.5)]
    for ID in candidate_IDs:
        if 110 > len(ID_dict[ID]) >100 :
            samp_IDs.append(ID)
    samp_IDs = samp_IDs[:args.num_ID]
    
    ## split train / test 
    test_ID_dict = OrderedDict()
    train_ID_dict = OrderedDict()
    relabel = 0
    for ID in samp_IDs:
        idx = ID_dict[ID]
        test_ID_dict[relabel] = ID_dict[ID][:40]
        train_ID_dict[relabel] = ID_dict[ID][40:]
        relabel += 1

    # for local test    
    test_dir = os.path.join(output_dir,'local_veri_%d'%(args.num_ID))
    os.system('mkdir -p %s'%(test_dir))
    test_id_txt = os.path.join(test_dir,'idx_id_pair.txt')
    total = sum([len(test_ID_dict[ID]) for ID in test_ID_dict])
    print('Total test images : %d, Total test ID : %d'%(total,len(test_ID_dict)))
    if not os.path.exists(test_id_txt):
        w_file = open(test_id_txt,'w')
        w_file.write('idx id\n')
        # create record
        record = mx.recordio.MXIndexedRecordIO(os.path.join(test_dir,'test.idx'),os.path.join(test_dir,'test.rec'),'w')
        pbar = tqdm(total=total,ncols=120,leave=True)
        img_idx = 1
        for ID in test_ID_dict:
            img_list = []
            name_list = []
            for idx in test_ID_dict[ID]:
                s = dataset.imgrec.read_idx(idx)
                header,img = mx.recordio.unpack(s)
                header = mx.recordio.IRHeader(0,ID,img_idx,0)
                new_pack_s = mx.recordio.pack(header,img)
                record.write_idx(img_idx,new_pack_s)
                w_file.write('%d %d\n'%(img_idx,ID))
                img_idx += 1
                pbar.update(1)
        idx0 = mx.recordio.pack(mx.recordio.IRHeader(2,[img_idx,len(test_ID_dict)],0,0),bytes(0))
        record.write_idx(0,idx0)
        record.close()
        pbar.close()
        w_file.close()

    # for local 1:n gallery 
    test_gallery_dir = os.path.join(output_dir,'local_gallery_%d'%(args.num_ID))
    total = sum([len(train_ID_dict[ID]) for ID in train_ID_dict])
    print('Total gallery images : %d, Total test ID : %d'%(total,len(train_ID_dict)))
    if not os.path.exists(test_gallery_dir):
        os.system('mkdir -p %s'%(test_gallery_dir))
        # create record
        record = mx.recordio.MXIndexedRecordIO(os.path.join(test_gallery_dir,'test.idx'),os.path.join(test_gallery_dir,'test.rec'),'w')
        pbar = tqdm(total=total,ncols=120,leave=True)
        img_idx = 1
        for ID in train_ID_dict:
            img_list = []
            name_list = []
            for idx in train_ID_dict[ID]:
                s = dataset.imgrec.read_idx(idx)
                header,img = mx.recordio.unpack(s)
                header = mx.recordio.IRHeader(0,ID,img_idx,0)
                new_pack_s = mx.recordio.pack(header,img)
                record.write_idx(img_idx,new_pack_s)
                img_idx += 1
                pbar.update(1)
        idx0 = mx.recordio.pack(mx.recordio.IRHeader(2,[img_idx,len(train_ID_dict)],0,0),bytes(0))
        record.write_idx(0,idx0)
        record.close()
        pbar.close()

    # for train
    train_dir = os.path.join(output_dir,'split_train_i%04dc%04d'%(args.num_ID,args.num_client))
    start = 0
    if not os.path.exists(train_dir):
        os.system('mkdir -p %s'%(train_dir))
        for i in range(args.num_client):
            num_ID = args.num_ID // args.num_client
            client_ID = list(range(start,start+num_ID))
        
            total = sum([len(train_ID_dict[ID]) for ID in client_ID])
            pbar = tqdm(total=total,ncols=140,leave=True)
            pbar.set_description('Client [%02d/%02d], #ID %05d, #imgs %06d'%(i+1,args.num_client,len(client_ID),total))


            client_dir = os.path.join(train_dir,'client_%04d'%i)
            os.system('mkdir -p %s'%(client_dir))
            record = mx.recordio.MXIndexedRecordIO(os.path.join(client_dir,'train.idx'),os.path.join(client_dir,'train.rec'),'w')
            img_idx = 1
            for new_ID, ID in enumerate(client_ID):
                for idx in train_ID_dict[ID]:
                    s = dataset.imgrec.read_idx(idx)
                    header,img = mx.recordio.unpack(s)
                    header = mx.recordio.IRHeader(0,new_ID,img_idx,0)
                    new_pack_s = mx.recordio.pack(header,img)
                    record.write_idx(img_idx, new_pack_s)
                    img_idx += 1
                    pbar.update(1)
            idx0 = mx.recordio.pack(mx.recordio.IRHeader(2,[img_idx,num_ID,start],0,0),bytes(0))
            record.write_idx(0, idx0)
            record.close()
            pbar.close()
            start += num_ID
