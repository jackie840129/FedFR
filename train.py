import argparse
import logging
import os
import time
import warnings
warnings.filterwarnings("ignore")
import random 
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
import pickle
import numpy as np
import backbones
import losses
from config import config as cfg
from dataset import MXFaceDataset, DataLoaderX, All_Client_Dataset
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

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
set_random_seed(100)

from client import Client 
from server import Server

def main(args):
    rank = 0
    local_rank = 0

    # logging
    if not os.path.exists(args.output_dir) and rank == 0:
        os.makedirs(args.output_dir)
    else:
        time.sleep(0.05)
    # copy code
    os.system('mkdir -p %s'%os.path.join(args.output_dir,'code'))
    os.system('cp -r *.py *.sh backbones/ eval/ utils/ %s'%(os.path.join(args.output_dir,'code')))

    log_root = logging.getLogger('FL_face')
    log_root.propagate = False
    init_logging(log_root, rank, args.output_dir)
    log_root.info(args)

    # create dataset
    all_data = All_Client_Dataset(root_dir=cfg.rec,local_rank=local_rank,args=args)

    # create clinets
    log_root.info('Start create %d clients...'%args.num_client)
    clients = []
    for i in range(args.num_client):
        if i % 20 == 0:
            print('Create client : %d/%d'%(i+1,args.num_client))
        clients.append(Client(cid=i,args=args,data=all_data))
    # create server
    log_root.info('Start create server...')
    server = Server(clients=clients,data=all_data,args=args)

    log_root.info('===Start Federated learning===')
    for i in range(args.total_round):
        ## based on csr, sample a subset of client (default: 100%)
        current_client_list = sorted(random.sample(list(range(server.num_client)),int(server.csr*server.num_client)))
        server.current_client_list = current_client_list
        log_root.info('\n ====== Round %d======'%(i))
        ## train
        server.train()
        if args.spreadout:
            server.SpreadOut(sp_iter=20,mode='mean')
        
        if i % 1 == 0 and i >= 0:
            # test on small dataset
            server.test()

        server.global_epoch += server.local_epoch
        server.global_round += 1
        torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Federated Face Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default="sphnet", help="backbone network")
    parser.add_argument('--loss', type=str, default="CosFace", help="loss function")
    ###
    parser.add_argument('--output_dir',default='./ckpt/FL_FedFR',help='output directory')
    parser.add_argument('--batch_size',default=64,type=int,help='batch_size per device')
    ##
    parser.add_argument('--local_epoch',default=1,type=int)
    parser.add_argument('--total_round',default=16,type=int)
    parser.add_argument('--data_ratio',default=0.5,type=float)
    parser.add_argument('--num_client',default=10,type=int)
    parser.add_argument('--client_sampled_ratio',default=1,type=float)
    parser.add_argument('--pretrained_root',default='')
    parser.add_argument('--lr',type=float,default=0.1)
    parser.add_argument('--lr_step',default='1000',type=str)
    parser.add_argument('--aggr_alg',default='FedAvg')
    parser.add_argument('--spreadout',action='store_true')
    parser.add_argument('--init_fc',action='store_true')
    parser.add_argument('--save_fc_iter',type=int,default=40)
    parser.add_argument('--fedface',action='store_true')
    parser.add_argument('--add_pretrained_data',action='store_true')
    ####
    parser.add_argument('--contrastive_bb',action='store_true')
    ####
    parser.add_argument('--return_all',action='store_true')
    parser.add_argument('--combine_dataset',action='store_true')
    parser.add_argument('--BCE_local',action='store_true')
    parser.add_argument('--BCE_detach',action='store_true')
    parser.add_argument('--BCE_tune',action='store_true')
    parser.add_argument('--adaptive_local_epoch',action='store_true')
    parser.add_argument('--reweight_cosface',action='store_true')
    args_ = parser.parse_args()
    #### for config
    cfg.lr = args_.lr
    cfg.total_round = args_.total_round
    ### Not actually used
    cfg.step = [int(i) for i in args_.lr_step.split(' ')]
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in cfg.step if m - 1 <= epoch])
    cfg.lr_func = lr_step_func
    ###
    args_.pretrained_bb_path = os.path.join(args_.pretrained_root,'backbone.pth') 
    args_.pretrained_fc_path = os.path.join(args_.pretrained_root,'fc.pth')
    #####
    main(args_)
