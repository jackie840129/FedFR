import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir')
parser.add_argument('--root_path',default='/home/jackieliu/MS-intern/face_recognition/IJBC')
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--job',default='1:1')
parser.add_argument('--epoch',type=int,nargs='+')
parser.add_argument('--gpu',type=str,nargs='+')
args = parser.parse_args()

gpu = ','.join(args.gpu)
for e in args.epoch:
    command = 'python3 ijbc_all.py \
        --model-prefix %s/backbone_%d.pth\
        --root-path %s \
        --result-dir %s \
        --epoch %d \
        --batch-size %d \
        --network sphnet \
        --job %s'%(args.ckpt_dir,e,args.root_path,args.ckpt_dir,e,args.batch_size,args.job)
    os.system('CUDA_VISIBLE_DEVICES=%s %s'%(gpu,command))