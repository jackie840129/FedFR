from numba import cuda
import numpy as np
import os
from sklearn.preprocessing import normalize
import time
import multiprocessing as mp
from math import ceil
import sys
import atexit
from scipy.interpolate import interp1d
from ctypes import cdll
import argparse

@cuda.jit
def calc_ROC(feature, label, subfeature, sublabel, out):
    i, j  = cuda.grid(2)
    if i < j and j < feature.shape[0] and i < sublabel.shape[0]:# and sublabel[i] * label[j] < 0:
        tmp = 0.
        for k in range(feature.shape[1]):
            tmp += subfeature[i, k] * feature[j, k]
        index_dis = int((tmp + 1) * 1000)
        # if index_dis < 0 or index_dis > 2000:
        #     print("out range: " + str(index_dis))
        if sublabel[i] == label[j]:
        #if sublabel[i] + label[j] == 0:
            cuda.atomic.add(out, 2*index_dis, 1)
        else:
            cuda.atomic.add(out, 2*index_dis+1, 1)

def gpu_job_producer(size, batch_size, target_size, _in_queue):
    index = np.arange(size)
    for i in range(ceil(target_size/batch_size)):
        min_up = min((i+1)*batch_size, target_size)
        _in_queue.put((index[i*batch_size : min_up], i*batch_size))
    while True:
        time.sleep(0.1)

def gpu_job_consumer(feature, label, _in_queue, _out_queue, device):
        cuda.select_device(device)
        blockdim = (32, 32)
        while True:
            index, start = _in_queue.get()
            feature_cuda = cuda.to_device(feature[start:, :].astype(np.float32))
            label_cuda = cuda.to_device(label[start:].astype(np.int32))
            gridnum1 = (feature_cuda.shape[0] + blockdim[0] - 1) // blockdim[0]
            gridnum2 = (len(index) + blockdim[0] - 1) // blockdim[0]
            griddim = (gridnum2, gridnum1)
            subfeature = cuda.to_device(feature[index, :].astype(np.float32))
            sublabel = cuda.to_device(label[index].astype(np.int32))
            out = cuda.to_device(np.zeros(2001*2, dtype=np.float64))
            calc_ROC[griddim, blockdim](feature_cuda, label_cuda, subfeature, sublabel, out)
            out_h = out.copy_to_host().astype(np.int64)
            _out_queue.put(out_h)


def plot_ROC(data, output_dir,epoch, target_label):
    data = np.cumsum(data, axis=0)
    TPR = [1.0] #TPR = TP / (TP + FN)
    FPR = [1.0] #FPR = FP / (FP + TN)
    for i in range(data.shape[0]):
        TPR.append((data[-1,0] - data[i, 0]) / data[-1,0])
        FPR.append((data[-1,1] - data[i, 1]) / data[-1,1])
    TPR = np.array(TPR)
    FPR = np.array(FPR)
    idx = np.argsort(FPR)
    # with open(save_ROC_path, 'w') as pf:
        # for i in range(TPR.shape[0]):
            # pf.write("%f %f\n"%(FPR[i], TPR[i]))
    ROC = interp1d(FPR[idx], TPR[idx])

    result = [float('%.2f'%(100*ROC(10**(i)))) for i in range(-1,-7,-1)]
    print('-'*80)
    print('Target label from %d to %d'%(target_label[0],target_label[-1]))
    print('Epoch %d, TPR (-1 to -6) = %r'%(epoch,result))
    print('-'*80)
    # plt.figure()
    # plt.plot(FPR, TPR, label='ROC curve')
    # plt.xlim([1e-7, 1e-2])
    # plt.ylim([0.9, 1.0])
    # plt.xscale('log')
    # plt.xticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    # plt.grid(True)
    # plt.savefig(save_ROC_path + ".png")
    save_path = os.path.join(output_dir,'local_log.txt')
    with open(save_path, "a") as pf:
        pf.write('Target label from %d to %d\n'%(target_label[0],target_label[-1]))
        pf.write('Epoch %d, TPR (-1 to -6) = %r\n'%(epoch,result))

class multiGPU(object):
    def __init__(self, feature, label, batch_size, target_size, workers):
        num = feature.shape[0]
        self._manager = mp.Manager()
        self._in_queue = self._manager.Queue(10)
        self._producer = mp.Process(target=gpu_job_producer, 
                                args=(num, batch_size, target_size, self._in_queue,)) 
        self._producer.start()
        self._out_queue = self._manager.Queue(10)
        #gpu_job_consumer(feature, label, self._in_queue, self._out_queue, 0)
        self._consumer = [mp.Process(target=gpu_job_consumer, 
                                args=(feature, label, self._in_queue, self._out_queue, device)) 
                                for device in range(workers)]
        for cons in self._consumer:
            cons.start()
        atexit.register(self._terminate)
    def _terminate(self):
        self._producer.terminate()
        for cons in self._consumer:
            cons.terminate()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    parser.add_argument('--feat_path',type=str)
    parser.add_argument('--label_path',type=str)
    parser.add_argument('--output_dir',type=str)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--batch_size',type=int,default=800)
    parser.add_argument('--ID_s_e',type=int,nargs='+')
    parser.add_argument('--epoch',type=int,default=0)
    args = parser.parse_args()

    feature = np.load(args.feat_path).astype('float32')
    label = np.load(args.label_path).astype('int32').reshape(-1)
    workers = args.workers
    batch_size = args.batch_size
    
    start_ID,end_ID = args.ID_s_e
    target_label = list(range(start_ID,end_ID))
    t_idx = label==target_label[0]
    for i in range(1,len(target_label)):
        t_idx = t_idx | (label==target_label[i])
    target_size = sum(t_idx)
    feature = np.concatenate([feature[t_idx],feature[~t_idx]],axis=0)
    label = np.concatenate([label[t_idx],label[~t_idx]])

    proccer = multiGPU(feature, label, batch_size, target_size, workers)
    out_sum = np.zeros(2001*2, dtype=np.int64)
    start = time.time()
    for _ in range(ceil(target_size / batch_size)):
        out = proccer._out_queue.get()
        out_sum += out
    print('Total pair :',out_sum.sum())
    # print('total use {:.2f}s'.format(time.time() - start))
    cuda.profile_stop()
    cuda.close()
    out_sum = out_sum.reshape([-1, 2])
    start = time.time()
    plot_ROC(out_sum, args.output_dir,args.epoch,target_label)
    # print('total use {:.2}s'.format(time.time() - start))