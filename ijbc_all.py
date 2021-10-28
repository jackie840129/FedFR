# coding: utf-8

import os
import pickle
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from prettytable import PrettyTable
from pathlib import Path
import sys
import warnings
from tqdm import tqdm
import torch.nn as nn
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
from skimage import transform as trans
import backbones
from torch.utils.data import TensorDataset,DataLoader
import random
from datetime import datetime as dt
import math
import heapq

def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
set_random_seed(100)

class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1, epoch=0, use_flip_test=False):
        image_size = (112, 112)
        self.image_size = image_size
        weight = torch.load(prefix)
        resnet = eval("backbones.{}".format(args.network))(False).cuda()
        resnet.load_state_dict(weight)
        print('Model create & load !')
        model = torch.nn.DataParallel(resnet)

        self.model = model
        self.model.eval()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.use_flip_test = use_flip_test

    def get(self, rimg, landmark):

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        input_blob = np.zeros((1, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        if self.use_flip_test:
            flip_imgs = torch.fliplr(imgs)
            b,c,h,w = imgs.shape
            imgs = torch.cat([imgs.unsqueeze(1),flip_imgs.unsqueeze(1)],dim=1).reshape(b*2,c,h,w)
        feat = self.model(imgs)
        if self.use_flip_test:
            feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()


# 将一个list尽量均分成n份，限制len(list)==n，份数大于原list内元素个数则分配空list[]
def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList

def read_template_subject_id_list(path):
    ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
    templates = ijb_meta[:, 0].astype(np.int)
    subject_ids = ijb_meta[:, 1].astype(np.int)
    return templates, subject_ids

def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label

def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

def get_image_feature(img_path, files_list, model_path, epoch, batch_size, use_flip_test):
    data_shape = (3, 112, 112)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []

    embedding = Embedding(model_path, data_shape, batch_size, epoch,use_flip_test)
    # load saved images
    npy_path = os.path.join('/'.join(img_path.split('/')[:-1]),'IJBC_imgs.npy')
    
    if os.path.exists(npy_path):
        print('img path exist : %s'%(npy_path))
        img_npy = np.load(npy_path)
        
        img_feats = []
        batch = 0
        for img_index, each_line in enumerate(files):
            name_lmk_score = each_line.strip().split(' ')
            
            if (img_index + 1) % batch_size == 0 or (img_index+1) == len(files):
                
                batch_data = img_npy[batch*batch_size : (batch+1)*batch_size]
                img_feats.append(embedding.forward_db(batch_data))
                batch += 1
                if batch % 100 == 0:
                    print('%d/%d'%(batch,len(files)//batch_size))
            faceness_scores.append(name_lmk_score[-1])

        img_feats = np.concatenate(img_feats,axis=0)
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        return img_feats, faceness_scores

    else:
        if use_flip_test == False:
            img_feats = np.empty((len(files), 512), dtype=np.float32)
        else:
            img_feats = np.empty((len(files), 1024), dtype=np.float32)
        batch_data = np.empty((batch_size, 3, 112, 112),dtype=np.uint8)
        batch_len = (len(files)-rare_size) // batch_size + 1
        batch = 0
        for img_index, each_line in enumerate(files[:len(files) - rare_size]):
            # 
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(img_path, name_lmk_score[0])
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            input_blob = embedding.get(img, lmk) # 1,3,112,112

            batch_data[ img_index - batch * batch_size][:] = input_blob[0]
        
            if (img_index + 1) % batch_size == 0:
                img_feats[batch * batch_size:batch * batch_size +
                                             batch_size] = embedding.forward_db(batch_data)
                batch += 1
                if batch % 100 == 0:
                    print('%d/%d'%(batch,batch_len))
            faceness_scores.append(name_lmk_score[-1])

        batch_data = np.empty((rare_size, 3, 112, 112),dtype=np.uint8)
        for img_index, each_line in enumerate(files[len(files) - rare_size:]):
            name_lmk_score = each_line.strip().split(' ')
            img_name = os.path.join(img_path, name_lmk_score[0])
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                           dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            input_blob = embedding.get(img, lmk)
            
            batch_data[img_index][:] = input_blob[0]
            if (img_index + 1) % rare_size == 0:
                img_feats[len(files) -
                          rare_size:] = embedding.forward_db(batch_data)
                batch += 1
                print('%d/%d'%(batch,batch_len))
            faceness_scores.append(name_lmk_score[-1])
    
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        img_feats = img_feats
        return img_feats, faceness_scores

def image2template_feature_11(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates

def image2template_feature_1n(img_feats=None,
                           templates=None,
                           medias=None,
                           choose_templates=None,
                           choose_ids=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates, indices = np.unique(choose_templates, return_index=True)
    unique_subjectids = choose_ids[indices]
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t, ) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], 0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    template_norm_feats = template_feats / np.sqrt(
        np.sum(template_feats**2, -1, keepdims=True))
    return template_norm_feats, unique_templates, unique_subjectids

def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats

def gen_mask(query_ids, reg_ids):
    mask = []
    for query_id in query_ids:
        pos = [i for i, x in enumerate(reg_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError(
                "RegIdsError with id = {}， duplicate = {} ".format(
                    query_id, len(pos)))
        mask.append(pos[0])
    return mask

def evaluation(query_feats, gallery_feats, mask):
    Fars = [0.01, 0.1]
    rank = dict()
    pr = dict()

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)

    # calculate top1
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num / query_num))
    rank['top1'] = correct_num/query_num
    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num / query_num))
    rank['top5'] = correct_num/query_num
    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:10]
        if mask[i] in j:
            correct_num += 1
    print("top10 = {}".format(correct_num / query_num))
    rank['top10'] = correct_num/query_num

    neg_pair_num = query_num * gallery_num - query_num
    print('neg_pair_num : ',neg_pair_num)
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    print('pos_sims : ',pos_sims.shape)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(
            far, recall, th))
        pr[far] = recall
    return rank,pr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test')
    # general
    parser.add_argument('--model-prefix', default='', help='path to load model.')
    parser.add_argument('--root-path', default='', type=str, help='')
    parser.add_argument('--result-dir', default='.', type=str, help='')
    parser.add_argument('--epoch',type=int,default=0)
    parser.add_argument('--batch-size', default=128, type=int, help='')
    parser.add_argument('--network', default='sphnet', type=str, help='')
    parser.add_argument('--job', default='1:1', type=str, help='job name, 1:1 or 1:n or both')
    parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC')
    parser.add_argument('--load_feature',default='',help='the path to load pre-forwrd features')
    args = parser.parse_args()

    target = args.target
    assert target.lower() == 'ijbc'
    model_path = args.model_prefix
    root_path = args.root_path
    result_dir = args.result_dir
    epoch = args.epoch
    job = args.job
    batch_size = args.batch_size
    
    use_norm_score = True  # if Ture, TestMode(N1)
    use_detector_score = True  # if Ture, TestMode(D1)
    use_flip_test = False  # if Ture, TestMode(F1)
    print('use_norm_score : %s, use_detector_score : %s, use_flip_test : %s.'%(use_norm_score,use_detector_score,use_flip_test))
    print('Checkpoint epoch : %d'%epoch)

    # Step1: Load Meta Data
    start = timeit.default_timer()
    templates, medias = read_template_media_list(
        os.path.join(root_path,'meta/%s_meta'%target,
        '%s_face_tid_mid.txt'%target.lower()))
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    ## 1:1 or 1:n
    start = timeit.default_timer()
    print('Job : %s'%(job))
    if job in ['1:1','both']:
        p1, p2, label = read_template_pair_list(
            os.path.join(root_path,'meta/%s_meta' % target,
                     '%s_template_pair_label.txt' % target.lower()))
    if job in ['1:n','both']:
        # gallery
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (target.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (target.lower())
        gallery_s1_templates, gallery_s1_subject_ids = read_template_subject_id_list(
            os.path.join(root_path,'meta/%s_meta'%target, gallery_s1_record))
        gallery_s2_templates, gallery_s2_subject_ids = read_template_subject_id_list(
            os.path.join(root_path,'meta/%s_meta'%target, gallery_s2_record))
        gallery_templates = np.concatenate(
            [gallery_s1_templates, gallery_s2_templates])
        gallery_subject_ids = np.concatenate(
            [gallery_s1_subject_ids, gallery_s2_subject_ids])
        # probe 
        probe_mixed_record = "%s_1N_probe_mixed.csv" % target.lower()
        probe_mixed_templates, probe_mixed_subject_ids = read_template_subject_id_list(
            os.path.join(root_path,'meta/%s_meta'%target, probe_mixed_record))

    if job not in ['1:1','1:n','both']:
        raise NotImplementedError()
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    
    # Step 2: Get Image Features
    start = timeit.default_timer()
    img_path = '%s/loose_crop' % root_path
    img_list_path = '%s/meta/%s_meta/%s_name_5pts_score.txt' % (root_path,target, target.lower())
    img_list = open(img_list_path)
    files = img_list.readlines()
    # files_list = divideIntoNstrand(files, rank_size)
    files_list = files

    if args.load_feature != '':
        print('Load features from %s'%args.load_feature)
        img_input_feats = np.load(args.load_feature)
    else: 
        img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                                       model_path, epoch,batch_size, use_flip_test)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))
        print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                                  img_feats.shape[1]))

        if use_flip_test:
            # add --- F2
            img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                         2] + img_feats[:, img_feats.shape[1] // 2:]
        else:
            img_input_feats = img_feats
            # img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

        if use_norm_score:
            img_input_feats = img_input_feats
        else:
            # normalise features to remove norm information
            img_input_feats = img_input_feats / np.sqrt(
                np.sum(img_input_feats ** 2, -1, keepdims=True))

        if use_detector_score:
            img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
        else:
            img_input_feats = img_input_feats

        # np.save(os.path.join(result_dir,'ijbc_feats.npy'),img_input_feats)
    
    if job in ['1:1','both']:
        # Step3: Get Template Features
        start = timeit.default_timer()
        template_norm_feats, unique_templates = image2template_feature_11(
                                img_input_feats, templates, medias)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        # Step 4: Get Template Similarity Scores
        # =============================================================
        # compute verification scores between template pairs.
        # =============================================================
        start = timeit.default_timer()
        score = verification(template_norm_feats, unique_templates, p1, p2)
        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        save_path = os.path.join(result_dir, '%s-1:1'%(target.lower()))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        score_save_file = os.path.join(save_path, "%s.npy" % (target.lower()))
        np.save(score_save_file, score)

        # Step 5: Get ROC Curves and TPR@FPR Table

        files = [score_save_file]
        methods = []
        scores = []
        for file in files:
            methods.append(Path(file).stem)
            scores.append(np.load(file))

        methods = np.array(methods)
        scores = dict(zip(methods, scores))
        x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
        tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])

        for method in methods:
            fpr, tpr, _ = roc_curve(label, scores[method])
            fpr = np.flipud(fpr)
            tpr = np.flipud(tpr)  # select largest tpr at same fpr
            tpr_fpr_row = []
            tpr_fpr_row.append("%s-%s" % (method, target))
            for fpr_iter in np.arange(len(x_labels)):
                _, min_index = min(
                    list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
                tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
            tpr_fpr_table.add_row(tpr_fpr_row)
        print(tpr_fpr_table)
        table_string = tpr_fpr_table.get_string()
        with open(os.path.join(save_path,'log.txt'),'a') as f:
            f.write('Epoch %d : \n'%(args.epoch))
            f.write(table_string+'\n')

    if job in ['1:n','both']:
        # Step3: Get Template Features
        start = timeit.default_timer()
        gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image2template_feature_1n(
            img_input_feats, templates, medias, gallery_templates, gallery_subject_ids)
        
        probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature_1n(
            img_input_feats, templates, medias, probe_mixed_templates, probe_mixed_subject_ids)
        print("gallery_templates_feature", gallery_templates_feature.shape)
        print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)
        print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
        print("probe_mixed_unique_subject_ids",probe_mixed_unique_subject_ids.shape)
        
        gallery_ids = gallery_unique_subject_ids
        gallery_feats = gallery_templates_feature
        probe_ids = probe_mixed_unique_subject_ids
        probe_feats = probe_mixed_templates_feature

        mask = gen_mask(probe_ids, gallery_ids)

        stop = timeit.default_timer()
        print('Time: %.2f s. ' % (stop - start))

        print('{}: start evaluation'.format(dt.now()))
        rank,pr = evaluation(probe_feats, gallery_feats, mask)
        print('{}: end evaluation'.format(dt.now()))

        save_path = os.path.join(result_dir, '%s-1:n'%(target.lower()))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path,'log.txt'),'a') as f:
            f.write('Epoch %d : \n'%(args.epoch))
            for r in rank:
                f.write('%s : %.5f\n'%(r,rank[r]))
            for far in pr:
                f.write('far = %.4f  pr = %.5f\n'%(far,pr[far]))
        