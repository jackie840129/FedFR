import argparse
import os
import torch
import numpy as np
import math
import heapq

def evaluation(query_feats, gallery_feats, mask):
    Fars = [1e-6,1e-5,1e-4,1e-3]

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    print(top_inds.shape)
    neg_pair_num = query_num * gallery_num - 40*gallery_num
    print(neg_pair_num)
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
    return result
