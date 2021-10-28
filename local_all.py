import os
import backbones
import argparse
import pickle
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve, auc
import sys
from tqdm import tqdm
import torch.nn as nn
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import defaultdict
import random
import cv2
import time
from math import ceil
import copy
from client import BCE_module
import numbers
import math
import heapq

def set_random_seed(seed_value, use_cuda=True,deterministic=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    # torch.use_deterministic_algorithms(True)
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        if deterministic == True:
            torch.backends.cudnn.deterministic = True  #needed
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super(MXFaceDataset, self).__init__()
        path_imgrec = os.path.join(root_dir, 'test.rec')
        path_imgidx = os.path.join(root_dir, 'test.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # process header
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
            self.num_classes = self.header0[1]
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            self.num_classes = None
        # transform
        self.transform = transform
        self.only_label = self.transform is None

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(int(label), dtype=torch.long)
        if self.only_label:
            return label
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,label

    def __len__(self):
        return len(self.imgidx)

def generate_gallery_labels(data_dir):
    batch_size = 1024
    test_dataset = MXFaceDataset(data_dir)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=1,drop_last=False)
    gallery_labels = []
    for batch_label in test_dataloader:
        gallery_labels.append(batch_label.numpy())
    gallery_labels = np.concatenate(gallery_labels,axis=0)
    return gallery_labels

def generate_gallery_features(backbone,data_dir,verbose=True,imgidx=None):
    batch_size = 512
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    test_dataset = MXFaceDataset(data_dir, test_transform)
    if imgidx is not None:
        test_dataset.imgidx = imgidx
    
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=6,drop_last=False)
    
    gallery_labels = []
    tic = time.time() 
    backbone = backbone.to(0)
    backbone = torch.nn.DataParallel(backbone)
    backbone.eval()
    img_feats = np.empty((len(test_dataset), 512), dtype=np.float32)
    with torch.no_grad():
        for i, (batch_img,batch_label) in enumerate(test_dataloader):
            if (i+1)%100 == 0 and verbose:
                print('%d/%d'%(i+1,len(test_dataloader)))
            output_feats = backbone(batch_img.to(0)).cpu().numpy()
            img_feats[i*batch_size: (i+1)*batch_size] = output_feats
            gallery_labels.append(batch_label.numpy())
        gallery_labels = np.concatenate(gallery_labels,axis=0)
    img_feats = sklearn.preprocessing.normalize(img_feats)
    toc = time.time()
    backbone = backbone.module.cpu()
    torch.cuda.empty_cache()
    print('Takes %.2f sec to generate imgage features'%(toc-tic),img_feats.shape)
    return img_feats,gallery_labels

def combine_features_subset(gallery_feats,gallery_labels,start_ID,end_ID):
    mean_feats = []
    for ID in range(start_ID,end_ID):
        idx = np.where(gallery_labels==ID)[0]
        mean_feats.append(np.mean(gallery_feats[idx],axis=0,keepdims=True))
    mean_feats = np.concatenate(mean_feats,axis=0)
    return mean_feats,np.arange(start_ID,end_ID)

def combine_features(gallery_feats,gallery_labels,start_ID,end_ID):
    target_idx = []
    for ID in range(start_ID,end_ID):
        target_idx.append(np.where(gallery_labels==ID)[0])
    
    mean_feats = []
    for i in range(end_ID-start_ID):
        mean_feats.append(np.mean(gallery_feats[target_idx[i]],axis=0,keepdims=True))
    mean_feats = np.concatenate(mean_feats,axis=0)
    return mean_feats, np.arange(start_ID,end_ID)

def evaluation(query_feats, gallery_feats, mask):
    Fars = [1e-6,1e-5,1e-4,1e-3]

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    neg_pair_num = query_num * gallery_num - 40*gallery_num
    print('neg_pair_num',neg_pair_num)
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        if gt != -1:
            pos_sims.append(top_sims[i, gt])
            top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    print(pos_sims.shape)

    neg_sims = top_sims[np.where(top_sims > -2.0)]
    print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    result = []
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / (40*gallery_num)
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))
        result.append(recall)
    return result,Fars


def generate_features(backbone,data_dir,flip_test=False,verbose=True): 
    ## prepare dataset
    batch_size = 512
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    test_dataset = MXFaceDataset(data_dir, test_transform)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=6,drop_last=False)

    tic = time.time() 
    backbone = backbone.to(0)
    backbone = torch.nn.DataParallel(backbone)
    backbone.eval()
    if verbose:
        print('Start generating image features')
    labels = []
    if flip_test == False:
        img_feats = np.empty((len(test_dataset), 512), dtype=np.float32)
    else:
        img_feats = np.empty((len(test_dataset), 1024), dtype=np.float32)
    with torch.no_grad():
        for i,(batch_img,batch_label) in enumerate(test_dataloader):
            if (i+1)%100 == 0 and verbose:
                print('%d/%d'%(i+1,len(test_dataloader)))
            if flip_test == False:
                output_feats = backbone(batch_img.to(0)).cpu().numpy()
                img_feats[i*batch_size: (i+1)*batch_size] = output_feats
            else:
                output_feats = backbone(batch_img.to(0)).cpu().numpy()
                output_feats_flip = backbone(torch.fliplr(batch_img).to(0)).cpu().numpy()
                img_feats[i*batch_size:(i+1)*batch_size] = np.concatenate([output_feats,output_feats_flip],axis=1)
            labels.append(batch_label.numpy())
    labels = np.concatenate(labels,axis=0)
    if flip_test:
        img_feats = img_feats[:,:img_feats.shape[1]//2] + img_feats[:,img_feats.shape[1]//2:]
    img_feats = sklearn.preprocessing.normalize(img_feats)
    toc = time.time()
    backbone = backbone.module.cpu()
    torch.cuda.empty_cache()
    if verbose:
        print('Takes %.2f sec to generate imgage features'%(toc-tic),img_feats.shape)
    return img_feats,labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--backbone',type=str,default='multi')
    parser.add_argument('--task',default='1:1')
    parser.add_argument('--ckpt_path')
    parser.add_argument('--data_dir',type=str,default='/home/jackieliu/MS-intern/face_recognition/ms1m_split/local_veri_4000')
    parser.add_argument('--gallery_data_dir',type=str,default='/home/jackieliu/MS-intern/face_recognition/ms1m_split/local_gallery_4000')
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--num_client',type=int)
    parser.add_argument('--gpu',type=str,nargs='+')
    parser.add_argument('--fp16',action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(args.gpu)

    set_random_seed(100)
    backbone = eval("backbones.{}".format('sphnet'))(False, dropout=0, fp16=args.fp16)
    
    if args.backbone == 'single':
        model_path = os.path.join(args.ckpt_path,'backbone_%d.pth'%args.epoch)
        backbone.load_state_dict(torch.load(model_path,map_location='cpu'))
        backbone.eval()

        feat_path = os.path.join(args.ckpt_path,'img_feats.npy')
        label_path = os.path.join(args.ckpt_path,'labels.npy')
        
        if not os.path.exists(feat_path):
            img_feats,labels = generate_features(backbone, args.data_dir)
            backbone.cpu()
            np.save(feat_path,img_feats)
            np.save(label_path,labels)
        else:
            print('Load veri features')
            img_feats = np.load(feat_path)
            labels = np.load(label_path)

        if args.task == '1:n':
            results = []
            # for gallery
            feat_path = os.path.join(args.ckpt_path,'img_gallery_feats.npy')
            label_path = os.path.join(args.ckpt_path,'img_gallery_labels.npy')
            if not os.path.exists(feat_path):
                img_gallery_feats, img_gallery_labels = generate_gallery_features(backbone,args.gallery_data_dir)
                backbone.cpu()
                np.save(feat_path,img_gallery_feats)
                np.save(label_path,img_gallery_labels)
            else:
                print('Load gallery features')
                img_gallery_feats = np.load(feat_path)
                img_gallery_labels = np.load(label_path)
            
            for c in range(args.num_client):
                # creating gallery
                start_ID = c*(4000//args.num_client)
                end_ID = (c+1)*(4000//args.num_client)

                gallery_feats,gallery_labels = combine_features(img_gallery_feats, img_gallery_labels, start_ID, end_ID)

                query_feats = img_feats
                query_labels = copy.deepcopy(labels)

                # gen mask
                true_idx = np.zeros(len(query_labels),dtype=np.bool)
                true_idx[start_ID*40:end_ID*40] = True
                query_labels[true_idx] -= start_ID
                query_labels[~true_idx] = -1

                result,fars = evaluation(query_feats, gallery_feats, query_labels)
                results.append(result)
            mean = np.mean(np.array(results),axis=0)
            print('-'*40)
            print('1:n average results:')
            print('Far: %r'%['%.1e'%far for far in fars])
            print('Pr : %r'%['%.5f'%pr for pr in mean])
            
            file_txt = os.path.join(args.ckpt_path,'local_log.txt')
            with open(file_txt,'a') as f:
                f.write('1:n at Epoch : %d\n'%args.epoch)
                f.write('Far: %r\n'%['%.1e'%far for far in fars])
                f.write('Pr : %r\n'%['%.5f'%pr for pr in mean])
        elif args.task == '1:1':
            gpu_counts = min(2,torch.cuda.device_count())
            file_txt = os.path.join(args.ckpt_path,'local_log.txt')
            with open(file_txt,'a') as f:
                f.write('1:1 at Epoch : %d\n'%args.epoch)

            for c in range(args.num_client):
                start = c * (4000//args.num_client) 
                end = (c+1) * (4000//args.num_client)

                os.system('python3 roc_cuda.py --feat_path %s --label_path %s --output_dir %s --ID_s_e %d %d --epoch %d --workers %d'\
                    %(feat_path,label_path,args.ckpt_path,start,end,args.epoch,gpu_counts))
            
            file_txt = os.path.join(args.ckpt_path,'local_log.txt')
            scores = []
            with open(file_txt,'r') as f:
                for line in f:
                    if 'Epoch %d, TPR'%args.epoch in line:
                        s = line.find('[')
                        end = line.find(']')
                        score = [float(i) for i in line[s+1:end].split(',')]
                        scores.append(score)
            scores = np.array(scores)
            mean = np.mean(scores,axis=0)
            with open(file_txt,'a') as f:
                f.write('Mean (-6 to -1):\n')
                f.write('[')
                for i in range(len(mean)):
                    f.write('%.2f '%(mean[len(mean)-1-i]))
                f.write(']\n')
            print('-'*40)
            print('1:1 average results (-6 to -1):')
            print('%r'%['%.2f'%mean[len(mean)-1-i] for i in range(len(mean))])
            # os.remove(feat_path)
            # os.remove(label_path)

    elif args.backbone == 'multi' :
        # path (/to/clients/)
        results = []
        for c in range(args.num_client):
            print('Process client %d...'%c)
            if args.epoch == -1:
                model_path = os.path.join(args.ckpt_path,'clients','client_%d'%(c),'backbone.pth')
                bce_path = os.path.join(args.ckpt_path,'clients','client_%d'%(c),'bce_module.pth')
            else:
                model_path = os.path.join(args.ckpt_path,'clients','client_%d'%(c),'backbone_%d.pth'%args.epoch)
                bce_path = os.path.join(args.ckpt_path,'clients','client_%d'%(c),'bce_module_%d.pth'%args.epoch)

            backbone = eval("backbones.{}".format('sphnet'))(False, dropout=0, fp16=args.fp16)
            backbone.load_state_dict(torch.load(model_path,map_location='cpu'))
            backbone.eval()
            if os.path.exists(bce_path):
                print('load BCE')
                state = torch.load(bce_path,map_location='cpu')
                bce_module = BCE_module(512, 100)
                bce_module.load_state_dict(state)
                backbone = nn.Sequential(backbone,bce_module.converter)
            
            feat_path = os.path.join(args.ckpt_path,'clients','client_%d'%(c),'img_feats.npy')
            label_path = os.path.join(args.ckpt_path,'clients','client_%d'%(c),'labels.npy')
        
            if not os.path.exists(feat_path):
                img_feats,labels = generate_features(backbone, args.data_dir)
                backbone = backbone.cpu()
                np.save(feat_path,img_feats)
                np.save(label_path,labels)
            else:
                print('Load veri features')
                img_feats = np.load(feat_path)
                labels = np.load(label_path)
            #-------------------------------------------------------------------
            if args.task == '1:n':
                # creating gallery
                gallery_feat_path = os.path.join(args.ckpt_path,'clients','client_%d'%c,'img_gallery_feats.npy')
                gallery_label_path = os.path.join(args.ckpt_path,'clients','client_%d'%c,'img_gallery_labels.npy')

                img_gallery_labels = generate_gallery_labels(args.gallery_data_dir)
                start_ID = c*(4000//args.num_client)
                end_ID = (c+1)*(4000//args.num_client)
                ## create a subset
                target_idx = []
                for ID in range(start_ID,end_ID):
                    target_idx.append(np.where(img_gallery_labels==ID)[0])
                imgidx = np.concatenate(target_idx).flatten() + 1
                
                if not os.path.exists(gallery_feat_path):
                    img_gallery_feats, img_gallery_labels = generate_gallery_features(backbone,args.gallery_data_dir,imgidx=imgidx)
                    gallery_feats,gallery_labels = combine_features_subset(img_gallery_feats, img_gallery_labels, start_ID, end_ID)
                    backbone.cpu()
                    np.save(gallery_feat_path,gallery_feats)
                    np.save(gallery_label_path,gallery_labels)
                else:
                    print('Load gallery features')
                    gallery_feats = np.load(gallery_feat_path)
                    gallery_labels = np.load(gallery_label_path)
                
                query_feats = img_feats
                query_labels = copy.deepcopy(labels)

                # gen mask
                true_idx = np.zeros(len(query_labels),dtype=np.bool)
                true_idx[start_ID*40:end_ID*40] = True
                query_labels[true_idx] -= start_ID
                query_labels[~true_idx] = -1
                result,fars = evaluation(query_feats, gallery_feats, query_labels)
                results.append(result)

            elif args.task == '1:1':
                gpu_counts = min(2,torch.cuda.device_count())
                start = c * (4000//args.num_client) 
                end = (c+1) * (4000//args.num_client)
                os.system('python3 roc_cuda.py --feat_path %s --label_path %s --output_dir %s --ID_s_e %d %d --epoch %d --workers %d'\
                    %(feat_path,label_path,args.ckpt_path,start,end,args.epoch,gpu_counts))
            os.remove(feat_path)
            os.remove(label_path)
        
        if args.task == '1:n':
            mean = np.mean(np.array(results),axis=0)
            print('-'*40)
            print('1:n average results:')
            print('Far: %r'%['%.1e'%far for far in fars])
            print('Pr : %r'%['%.5f'%pr for pr in mean])

            file_txt = os.path.join(args.ckpt_path,'local_log.txt')
            with open(file_txt,'a') as f:
                f.write('1:n at Epoch : %d\n'%args.epoch)
                f.write('Far: %r\n'%['%.1e'%far for far in fars])
                f.write('Pr : %r\n'%['%.5f'%pr for pr in mean])


        elif args.task == '1:1':
            file_txt = os.path.join(args.ckpt_path,'local_log.txt')
            scores = []
            with open(file_txt,'r') as f:
                for line in f:
                    if 'Epoch %d, TPR'%args.epoch in line:
                        s = line.find('[')
                        end = line.find(']')
                        score = [float(i) for i in line[s+1:end].split(',')]
                        scores.append(score)
            scores = np.array(scores)
            mean = np.mean(scores,axis=0)
            with open(file_txt,'a') as f:
                f.write('Mean (-6 to -1):\n')
                f.write('[')
                for i in range(len(mean)):
                    f.write('%.2f '%(mean[len(mean)-1-i]))
                f.write(']\n')
            print('-'*40)
            print('1:1 average results (-6 to -1):')
            print('%r'%['%.2f'%mean[len(mean)-1-i] for i in range(len(mean))])

