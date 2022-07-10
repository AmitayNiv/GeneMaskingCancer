import random
from time import time
import datetime
import numpy as np
import pandas as pd
import torch
import wandb
from run_tasks import run_train,eval_ensemble,run_create_and_save_masks#,run_masks_and_vis,run_gsea,run_heatmap_procces,run_per_sample_gsea_compare,run_per_sample_gsea
import os
import copy

CUDA_VISIBLE_DEVICES=4

class arguments(object):
   def __init__(self):
      self.seed = 3407
      self.cls_epochs = 10
      self.g_epochs = 10
      self.cls_lr = 0.0001#0.0001
      self.g_lr = 0.0002#0.0002
      self.weight_decay=5e-4
      self.dropout=0
      self.batch_size = 50
      self.batch_factor = 1
      self.train_ratio = 0.4
      self.data_type =  "CPCG"
      self.wandb_exp = False
      self.load_pretraind_weights = True
      self.save_weights = False
      self.iterations = 1
      self.working_models = {"F":True,"g":True,"F2":True,"H":False,"XGB":False,"RF":False}
      self.task = "Train"
      self.for_cpcg = True




def run(args):
    ## Init random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## Conecting to device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'Using device {device}')
    # run_gsea(args)
    if args.task =="Train":
        print("Starting Train")
        run_train(args,device)
    elif args.task =="eval_ensemble":
         print("Starting Mask Creation")
         eval_ensemble(args,device)
    elif args.task =="Masks":
        print("Start Masks Creation")
        run_create_and_save_masks(args,device)
    # elif args.task =="GSEA":
    #     print("Starting GSEA Analisys")
    #     run_gsea(args,device)
    # elif args.task =="Heatmaps":
    #     print("Starting Important Heatmaps Calculation")
    #     run_heatmap_procces(args,device)
    # elif args.task =="GSEA per Sample Compariosn":
    #     print("Starting GSEA per Sample Compariosn")
    #     run_per_sample_gsea_compare(args,device)
    # elif args.task =="GSEA per Sample":
    #     print("Starting GSEA per Sample")
    #     run_per_sample_gsea(args,device)

        


if __name__ == '__main__':
    args = arguments()
    run(args)