import torch
from config import config as cfg
import backbones
import logging
import random
import losses
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.distributed
from utils.utils_logging import AverageMeter, init_logging
import copy
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from eval_local import CallBack_LocalVerifi
from tqdm import tqdm
import gc
import copy
import os
import time
import pickle
import numpy as np
from functools import reduce

def FedPavg(models,weights):
    aggr = copy.deepcopy(models[0])
    weights = [ w/sum(weights) for w  in weights]
    with torch.no_grad():
        for name in aggr:
            tmp  = 0
            for i in range(len(models)):
                tmp += weights[i] * models[i][name]
            aggr[name] = tmp
    return aggr

def FedAvg_on_FC(pretrain_fc,models,weights,p):
    weights = [ w/sum(weights) for w  in weights]
    aggr = copy.deepcopy(models[0]) * weights[0]
    with torch.no_grad():
        for i in range(1,len(models)):
            aggr += models[i] * weights[i]
    if p == 1:
        pretrain_fc = aggr
    else:
        pretrain_fc = (1-p)*pretrain_fc + p * aggr
    return pretrain_fc

class SpreadOut_Module(nn.Module):
    def __init__(self,all_FC,margin=0.7,local=False,mode='sum'):
        super(SpreadOut_Module,self).__init__()
        self.FC = nn.Parameter(all_FC)
        self.margin = margin
        self.mode = mode

    def forward(self):
        FC = F.normalize(self.FC)
        similarity = torch.matmul(FC,FC.t())
        loss = F.relu(similarity.masked_select(~torch.eye(len(FC), dtype=bool).cuda())-self.margin)
        if self.mode == 'sum':
            loss = torch.sum(loss**2)
        elif self.mode == 'mean':
            loss = torch.mean(loss**2)
        return loss




class Server(object):
    def __init__(self,clients,data,args):
        self.data = data
        self.output_dir = args.output_dir
        self.args = args
        self.local_epoch = args.local_epoch
        self.clients = clients
        self.num_client = args.num_client
        self.csr = args.client_sampled_ratio
        self.logger = logging.getLogger('FL_face.server')
        if args.add_pretrained_data:
            self.public_train_loader = data.public_train_loader
            self.public_test_loader = data.public_test_loader
        ### Create backbone
        self.dropout = 0.4 if cfg.dataset is "webface" else 0
        self.federated_model = eval("backbones.{}".format(args.network))(False, dropout=self.dropout, fp16=cfg.fp16) # on cpu
        self.margin_softmax = eval("losses.{}".format(args.loss))(s=30.0,m=0.4)
        
        ## Load pretrained backbone
        if self.args.pretrained_root is not '':
            model = torch.load(self.args.pretrained_bb_path,map_location='cpu')
            self.federated_model.load_state_dict(model)
            self.logger.info('Succesfully load model from %s'%(self.args.pretrained_bb_path))
        else:
            self.logger.info('Train from scratch!')
        self.federated_model.eval()
        ###
        self.train_loss = []
        self.global_epoch = 0
        self.global_round = 0
        
        self.rank = 0 #dist.get_rank()
        self.local_rank = 0 #args.local_rank
        
        ## 1:1 verify on small dataset (cfp, lfw...)
        self.callback_verification = CallBackVerification(1, self.rank, cfg.val_targets, cfg.val_rec, self.num_client)
        
        ## 1:1 verify on our local dataset, sample 10 
        self.callback_local_veri = CallBack_LocalVerifi(frequent=1, rank=self.rank,data_dir=cfg.local_rec,output_dir=self.output_dir)
        self.local_candidates = sorted(list(np.random.permutation(self.num_client)[:10]))
        print('Local Veri Candidates',self.local_candidates)

        self.callback_checkpoint = CallBackModelCheckpoint(self.rank, self.output_dir)
        self.current_client_list = None
        
        ## local FC initialization
        self.norm_before_avg = False
        if args.init_fc:
            self.Initialize_local_FC(False)
        
        ### pretrained FC initialization
        if args.add_pretrained_data:
            if self.args.init_fc:
                self.pretrained_fc, self.pretrained_label = self.Initialize_pretrain_FC(only_labels=False)
            else:
                _,self.pretrained_label = self.Initialize_pretrain_FC(only_labels=True)
                self.pretrained_fc = torch.load(self.args.pretrained_fc_path,map_location='cpu')
                self.logger.info('Use pretrained perfect FC')
        
        ### BCE branch        
        if self.args.BCE_local:
            # self.logger.info('No init BCE')
            self.logger.info('Init BCE')
            ## No matter init_fc or not, BCE is initialized with the same weight
            for i in range(len(self.clients)):
                self.clients[i].bce_module.initialize(self.clients[i].fc_module.fc.data)

    def test(self):
        with torch.no_grad():
            self.federated_model.to(self.local_rank)
            self.federated_model = nn.DataParallel(self.federated_model)
            self.federated_model.eval()
            self.callback_verification(self.global_round, self.federated_model, None, th=0)
            # only rank 0 has highest_acc_list
            if self.callback_verification.highest_acc_list[-1][0] == self.global_round:
                self.callback_checkpoint(self.global_round,self.federated_model,None)
                self.logger.info('Save server model, epoch %d model...'%(self.global_round))
            self.federated_model = self.federated_model.module.cpu()

            if self.global_round >= 0 and self.global_round % 1 == 0:
                torch.save(self.federated_model.state_dict(), os.path.join(self.output_dir, "backbone_%d.pth"%self.global_round))
            
            # if self.local_rank == 0 and (self.global_epoch+1)%5 == 0:
                ## IJBC
                # torch.save(self.federated_model.state_dict(), os.path.join(self.output_dir, "ijbc_tmp.pth"))
                # os.system('CUDA_VISIBLE_DEVICES=%s sh run_IJBC.sh %s %s %d %s'\
                        # %(self.args.gpus,os.path.join(self.output_dir,'ijbc_tmp.pth'),\
                        #   self.output_dir,self.global_epoch,os.path.join(cfg.val_rec,'IJBC')))

    def Initialize_local_FC(self,save_to_disk=False):
        self.args.pre_init_path = os.path.join(self.args.pretrained_root,'preCos_init_AN.pth')
        ### Load pre-forward features from pretrained model
        if os.path.exists(self.args.pre_init_path):
            self.logger.info('Preload Clients FC init.')
            init_matrix = torch.load(self.args.pre_init_path,map_location='cpu')
            self.logger.info('Clients init shape %r'%list(init_matrix.shape))
            start = 0
            for i in range(len(self.clients)):
                num_classes = self.clients[i].num_classes
                self.clients[i].fc_module.update_from_tensor(init_matrix[start:start+num_classes,:])
                start += num_classes
        ### Infer the pretrained model to generate features
        else:
            collected_fc = []
            for i in range(len(self.clients)):
                self.logger.info('Client %d start initialize!'%(i))
                self.clients[i].data_update_fc(self.federated_model.state_dict(),self.norm_before_avg, \
                                                fc_name='cen_feats_init',save_to_disk=save_to_disk)
                collected_fc.append(self.clients[i].fc_module.fc.data)
            collected_fc = torch.cat(collected_fc,dim=0)
            # torch.save(collected_fc,self.args.pre_init_path)
            gc.collect()
            torch.cuda.empty_cache()
        
    def Initialize_pretrain_FC(self,load_pth=True,save_pth=False,only_labels=False):
        fc_name = os.path.join(self.args.pretrained_root,'preCos_pretrain_init_AN.pth')
        labels_name = os.path.join(self.args.pretrained_root,'preCos_pretrain_labels.pth')
        
        if os.path.exists(fc_name) and load_pth == True :
            raw_labels = torch.load(labels_name,map_location='cpu')
            self.logger.info('Preload pretrain labels, shape: %r'%list(raw_labels.shape))
            if only_labels:
                return None,raw_labels
            
            init_matrix = torch.load(fc_name, map_location='cpu')
            self.logger.info('Preload pretrain fc, shape: %r'%list(init_matrix.shape))
            return init_matrix, raw_labels

        ## forward backbone to create features 
        ### Creating model
        backbone = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
        backbone.load_state_dict(self.federated_model.state_dict())
        backbone = nn.DataParallel(backbone)
        backbone.to(self.local_rank)
        backbone.eval()
        # forward public data
        raw_labels = []
        ID_feature = dict()
        self.logger.info('Initialize pretrained fc on server')
        with torch.no_grad():
            pbar = tqdm(total=len(self.public_test_loader),ncols=120)
            for step, (img, label) in enumerate(self.public_test_loader):
                raw_labels.append(label.cpu())
                if not only_labels:
                    features = backbone(img.to(self.local_rank))
                    if self.norm_before_avg:
                        features = F.normalize(features)
                    for i,ID in enumerate(label):
                        ID = ID.item()
                        if ID not in ID_feature:
                            ID_feature[ID] = [torch.zeros_like(features[0]),0]
                        ID_feature[ID][0] += features[i]
                        ID_feature[ID][1] += 1
                pbar.update(1)
            pbar.close()

        raw_labels = torch.cat(raw_labels)
        self.logger.info('Generate pretrain labels, %r'%list(raw_labels.shape))
        if only_labels:
            del backbone
            return None, raw_labels

        init_matrix = torch.zeros(len(ID_feature),512).float()
        for i in range(len(init_matrix)):
            ## average
            init_matrix[i] = (ID_feature[i][0] / ID_feature[i][1]).cpu()
        # torch.save(init_matrix,fc_name)
        # torch.save(raw_labels, raw_name)
        self.logger.info('Generate pretrain fc, %r'%list(init_matrix.shape))
        del ID_feature, backbone
        gc.collect()
        torch.cuda.empty_cache()
        return init_matrix, raw_labels

    def Generate_pretrain_feats(self):
        self.logger.info('Generating pretrained features for HN sampling')
        backbone = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
        backbone.load_state_dict(self.federated_model.state_dict())
        backbone = nn.DataParallel(backbone)
        backbone.to(self.local_rank)
        backbone.eval()
        # forward public data
        raw_feats = []
        with torch.no_grad():
            pbar = tqdm(total=len(self.public_test_loader),ncols=120)
            for step, (img, label) in enumerate(self.public_test_loader):
                features = backbone(img.to(self.local_rank))
                features = F.normalize(features)
                raw_feats.append(features.cpu())
                pbar.update(1)
            pbar.close()
        raw_feats = torch.cat(raw_feats,dim=0)
        # torch.save(raw_feats,'tmp.pth')
        backbone = backbone.cpu()
        del backbone
        return raw_feats

    def train(self):
        models = []
        models_fc = []
        losses = []
        data_sizes =[]

        ## for feature-based HN
        if self.args.add_pretrained_data:
            # self.logger.info('load tmp.pth')
            # self.pretrained_feats = torch.load('tmp.pth')
            self.pretrained_feats = self.Generate_pretrain_feats()
        
        ### Adjust Local epoch and decay
        if self.args.adaptive_local_epoch and self.global_round != 0:
            self.local_epoch = max(4,self.local_epoch-2)
            cfg.train_decay = max(1,int(3/4*self.local_epoch))

        #### start train
        for idx,i in enumerate(self.current_client_list):
            self.logger.info('Round %d : [%d/%d] Client %d start training!'%(self.global_round,idx+1,len(self.current_client_list),i))
            self.logger.info('Server send backbone to clients')
            self.clients[i].backbone_state_dict = self.federated_model.state_dict()
            
            ## Adjust local epoch
            self.clients[i].local_epoch = self.local_epoch
            
            ## Train with local verify test
            if self.clients[i].cid in self.local_candidates:
                if self.args.add_pretrained_data:
                    self.clients[i].train_with_public_data(self.global_epoch,callback_verification=self.callback_local_veri,\
                        public_train_loader=self.public_train_loader,pretrained_fc=self.pretrained_fc,choose_hard_negative=True,\
                        pretrained_label=self.pretrained_label,pretrained_feats=self.pretrained_feats)
                else:
                    self.clients[i].train(self.global_epoch,callback_verification=self.callback_local_veri)
            ## train w/o local test
            else:
                if self.args.add_pretrained_data:
                    self.clients[i].train_with_public_data(self.global_epoch,\
                        public_train_loader=self.public_train_loader,pretrained_fc=self.pretrained_fc,choose_hard_negative=True,\
                        pretrained_label=self.pretrained_label,pretrained_feats=self.pretrained_feats)
                else:
                    self.clients[i].train(self.global_epoch)
            ####################################################### 
            losses.append(self.clients[i].get_train_loss())
            
            if self.args.return_all:
                models.append(self.clients[i].get_model())
                models_fc.append(self.clients[i].get_global_fc())
                self.clients[i].fc_module.remove_pretrain()
            else:
                models.append(self.clients[i].get_model())
            data_sizes.append(self.clients[i].get_data_size())

        avg_loss = sum(losses)/len(losses)
        self.logger.info("================")
        self.logger.info("Train Round {}. Avg Train loss among all clients {:.6f}".format(self.global_round,avg_loss))

        if self.args.return_all:
            p = 1.0
            self.logger.info('==========Do Fed FC==========')
            self.pretrain_fc = FedAvg_on_FC(self.pretrained_fc, models_fc, data_sizes, p=p)
            self.logger.info('==========Do FedPavg==========')
            if self.args.aggr_alg in ['FedAvg','FedProx']:
                aggr_state_dict = FedPavg(models,data_sizes)
                if p != 1.0:
                    global_state_dict = self.federated_model.state_dict()
                    for name in global_state_dict:
                        aggr_state_dict[name] = (1-p)*global_state_dict[name] + p * aggr_state_dict[name]
            self.federated_model.load_state_dict(aggr_state_dict)
        else:
            self.logger.info('==========Do FedPavg==========')
            if self.args.aggr_alg in ['FedAvg','FedProx']:
                aggr_state_dict = FedPavg(models,data_sizes)
            self.federated_model.load_state_dict(aggr_state_dict)

    def SpreadOut(self,sp_iter = 5,mode='sum'):
        assert(self.current_client_list is not None)
        FC_over_clients = []
        for idx,i in enumerate(self.current_client_list):
            FC_over_clients.append(self.clients[i].fc_module.fc.data)
        FC_over_clients = torch.cat(FC_over_clients,dim=0)
        self.logger.info('=====Collect FC and cat to a big matrix=====')
        ##### Initialize the SpreadOut module 
        SP = SpreadOut_Module(FC_over_clients.detach().clone(),margin=0.4,mode=mode).to(self.local_rank)
        SP_opt = torch.optim.SGD(SP.parameters(), lr=cfg.lr*10,momentum=0.9,weight_decay=cfg.weight_decay)
        self.logger.info('=====SpreadOut Module Create=====')
        for i in range(sp_iter):
            SP_opt.zero_grad()
            loss = SP()
            self.logger.info('- SP iter %d Loss :  %.5e , Start backward'%(i, loss.item()))
            loss.backward()
            SP_opt.step()
        #####
        with torch.no_grad():
            FC = SP.FC.data.cpu()
        for idx,i in enumerate(self.current_client_list):
            self.clients[i].fc_module.fc.data = FC[idx*self.clients[i].num_classes:(idx+1)*self.clients[i].num_classes]
        #####
        # if (self.global_epoch) % self.args.save_fc_iter == 0:
            # torch.save(FC_over_clients,os.path.join(self.output_dir,'Ep%d_allFC_beforeSP.pth'%self.global_epoch))  
            # torch.save(FC,os.path.join(self.output_dir,'Ep%d_allFC_afterSP.pth'%self.global_epoch))  
        self.logger.info('=====Update FC in partial FC module=====')
        ## delete garbage
        del SP,FC,FC_over_clients
        torch.cuda.empty_cache()
        gc.collect()
        return

