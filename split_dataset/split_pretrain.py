import os
import mxnet as mx
import numpy as np 
from torch.utils.data import DataLoader,Dataset
import time
from tqdm import tqdm,trange
from collections import defaultdict
import pickle
import argparse
import random
random.seed(100)

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
    parser.add_argument('--num_client',type=int,default=10)
    parser.add_argument('--num_ID',type=int,default=6000)
    parser.add_argument('--relabel',type=bool,default=True)
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
    
    IDs = list(ID_dict.keys())
    # shuffle the IDs with seed 100
    random.shuffle(IDs)
    # sample last half to create pretrain
    samp_IDs = []
    candidate_IDs = IDs[len(IDs)-int(len(IDs)*0.5):]
    for ID in candidate_IDs:
        if 80 > len(ID_dict[ID]) > 60 :
            samp_IDs.append(ID)
    assert len(samp_IDs) >= args.num_ID
    samp_IDs = samp_IDs[:args.num_ID]
    root_dir = os.path.join(output_dir,'split_pretrain_%d'%(len(samp_IDs)))
    os.system('mkdir -p %s'%root_dir)

    print('- %d ID are selected for creating FL dataset'%(len(samp_IDs)))
    print('- Split selected FL dataset to %d clients based on ID'%(args.num_client))
    # per client output  
    start = 0
    for i in range(args.num_client):
        num_ID = len(samp_IDs)//args.num_client + int((len(samp_IDs)%args.num_client) > i)
        client_ID = samp_IDs[start:start+num_ID]
        start += num_ID

        # create record io
        client_dir = os.path.join(root_dir,'client_%04d'%(i)) 
        os.system('mkdir -p %s'%(client_dir))
        record = mx.recordio.MXIndexedRecordIO(os.path.join(client_dir,'train.idx'),os.path.join(client_dir,'train.rec'),'w')
        # progress bar
        total = sum([len(ID_dict[ID]) for ID in client_ID])
        pbar = tqdm(total=total,ncols=140,leave=True)
        pbar.set_description('Client [%02d/%02d], #ID %05d, #imgs %06d'%(i+1,args.num_client,len(client_ID),total))


        per_client_idx = 1
        for new_ID,ID in enumerate(client_ID):
            for orig_idx in ID_dict[ID]:
                s = dataset.imgrec.read_idx(orig_idx)
                header,img = mx.recordio.unpack(s)
                if args.relabel :
                    header = mx.recordio.IRHeader(0,new_ID,per_client_idx,0)
                else:
                    header = mx.recordio.IRHeader(0,ID,per_client_idx,0)
                new_pack_s = mx.recordio.pack(header,img)
                record.write_idx(per_client_idx,new_pack_s)
                per_client_idx += 1
                pbar.update(1)
        idx0 = mx.recordio.pack(mx.recordio.IRHeader(2,[per_client_idx,len(client_ID)],0,0),img)
        record.write_idx(0,idx0)
        record.close()
        pbar.close()
        

