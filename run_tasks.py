import random
from time import time
import datetime
import numpy as np
import pandas as pd
import torch
import wandb
from data_loading import Data
from test import test,test_xgb
from train import train_G, train_classifier,train_xgb,train_H,train_f2,train_random_forest
from utils import get_mask, get_mask_and_mult,init_models,features_f_corelation,load_datasets_list,save_weights,load_weights,concat_average_dfs
from visualization import visulaize_tsne, visulaize_umap
import os
import copy
import joblib
import xgboost as xgb
import gseapy as gp
from data_loading import Data
from time import time
import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import eli5
import scipy
import shap

def run_train(args,device):
    """
    Run training for all chosen datasets, for all working models

    Arguments:
    args [obj] - Arguments
    device

    """
    ##
    time_for_file = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    datasets_list = load_datasets_list(args)
    first_data_set = True
    global_time = time()
    for i,f in enumerate(datasets_list):
        first_iteration = True
        dataset_time = time()
        ## Init WandB experiment
        if args.wandb_exp:
            wandb_exp = wandb.init(anonymous="must")
            wandb_exp.name = f"Train_{f.name}"
            wandb_exp.config.update(args.__dict__)
        else: 
            wandb_exp = None
        ##

        for j in range(args.iterations):
            res_dict = {}
            res_prints = ""
            iter_time = time()
            data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False)
            print(f"Training iteration:{j} dataset:{data.data_name}")
            
            if args.working_models["F"] or args.working_models["g"]:
                cls,g_model = init_models(args=args,data=data,device=device)
                cls,cls_res_dict = train_classifier(args,device=device,data_obj=data,model=cls,wandb_exp=wandb_exp)
                args.batch_factor=4
                args.weight_decay=0
                g_model ,g_res_dict= train_G(args,device,data_obj=data,classifier=cls,model=g_model,wandb_exp=wandb_exp)

                res_dict.update(cls_res_dict)
                res_dict.update(g_res_dict)
                res_prints+="\nF Resutls\n"
                res_prints+=str(cls_res_dict)
                res_prints+="\nG Resutls\n"
                res_prints+=str(g_res_dict)
                if args.save_weights:
                    save_weights(cls=cls,g=g_model,data=data)

            args.batch_factor=1
            args.weight_decay=5e-4
            if args.working_models["H"]:
                g_model_copy_h = copy.deepcopy(g_model)
                h,g_model_copy_h,h_res_dict = train_H(args,device,data_obj=data,g_model=g_model_copy_h,wandb_exp=None,model=None,train_H=True)
                res_dict.update(h_res_dict)
                res_prints+="\nF2_c Resutls\n"
                res_prints+=str(h_res_dict)
                if args.save_weights:
                    save_weights(cls=h,g=g_model_copy_h,data=data,base="H")
            if args.working_models["F2"]:
                g_model_copy_f2 = copy.deepcopy(g_model)
                f2,g_model_copy_f2,f2_res_dict = train_H(args,device,data_obj=data,g_model=g_model_copy_f2,wandb_exp=None,model=None,train_H=False)
                res_dict.update(f2_res_dict)
                res_prints+="\nF2 Resutls\n"
                res_prints+=str(f2_res_dict)
                if args.save_weights:
                    save_weights(cls=f2,g=g_model_copy_f2,data=data,base="F2")

            if args.working_models["XGB"]:
                xgb_cls,xgb_res_dict = train_xgb(data,device)
                res_dict.update(xgb_res_dict)
                res_prints+="\nXGB Resutls\n"
                res_prints+=str(xgb_res_dict)
                if args.save_weights:
                    save_weights(cls=xgb_cls,g=None,data=data,base="XGB")
            
            if args.working_models["RF"]:
                rf_cls,rf_res_dict = train_random_forest(data,device)
                res_dict.update(rf_res_dict)
                res_prints+="\nRF Resutls\n"
                res_prints+=str(rf_res_dict)
                if args.save_weights:
                    save_weights(cls=rf_cls,g=None,data=data,base="RF")


            print(f"############### Results on {data.data_name} ############################")
            print(res_prints)
            print(f"#####################################################################")
            

            if first_iteration:
                single_data_res_df = pd.DataFrame(res_dict, index=[data.data_name])
                first_iteration = False
            else:
                single_res_df = pd.DataFrame(res_dict, index=[data.data_name])
                single_data_res_df = pd.concat([single_data_res_df, single_res_df])
            time_diff = datetime.timedelta(seconds=time()-iter_time)
            print("{}: iteration #{} took {}".format(data.data_name,j+1,time_diff))
            print(f"#################################")
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("{}: {} iterations took {}".format(data.data_name,args.iterations,time_diff))  
        print(f"#################################")     
        single_data_res_mean = pd.DataFrame(single_data_res_df.mean()).T
        single_data_res_std = pd.DataFrame(single_data_res_df.std()).T
        single_data_res_mean.index = [data.data_name]
        single_data_res_std.index = [data.data_name]
        if first_data_set:
            full_resutls_df = single_data_res_df
            mean_resutls_df = single_data_res_mean
            std_results_df = single_data_res_std
            first_data_set = False
            
        else:
            full_resutls_df = pd.concat([full_resutls_df, single_data_res_df])
            mean_resutls_df = pd.concat([mean_resutls_df, single_data_res_mean])
            std_results_df = pd.concat([std_results_df, single_data_res_std])
            

        ### Saving the results every Iteration
        full_resutls_df.to_csv(f"./results/{time_for_file}_full_res_df.csv")
        if args.iterations>1:
            mean_resutls_df.to_csv(f"./results/{time_for_file}_mean_res_df.csv")
            std_results_df.to_csv(f"./results/{time_for_file}_std_res_df.csv")

    time_diff = datetime.timedelta(seconds=time()-global_time)
    print("All training took: {}".format(time_diff))   
    print(f"#################################")  


def run_create_and_save_masks(args,device,models =["G","F2","H"]):
    """
    Cerating masks and save masks to files 

    Arguments:
    args [obj] - Arguments
    device
    models [list] - models to create masks for

    """
    datasets_list = load_datasets_list(args)
    for i,f in enumerate(datasets_list):
        dataset_time = time()
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True)
        print(f"Masking dataset:{data.data_name}")
        if not os.path.exists(f"./masks/{data.data_name}/"):
            os.mkdir(f"./masks/{data.data_name}/")
        ###########################################################
        for mod in models:
            base_print = "" if mod =="G" else mod
            _,g = load_weights(data,device,base_print,only_g=True)
                
            mask = get_mask(g,data,args,device)

            mask_df = pd.DataFrame(np.array(mask.detach().cpu(),dtype=float),columns = list(data.colnames))
            mask_df.to_csv(f"./masks/{data.data_name}/{mod}_mask.csv")
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("{}:took {}".format(data.data_name,time_diff))  


def run_masks_and_vis(args,device):
    """
    Cerating mask and UMAP projection (input and mask) of H for all datasets

    Arguments:
    args [obj] - Arguments
    device

    """
    datasets_list = load_datasets_list(args)
    for i,f in enumerate(datasets_list):
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,test_set=True)
        print(f"Masking dataset:{data.data_name}")
        if not os.path.exists(f"./results/{data.data_name}/"):
            os.mkdir(f"./results/{data.data_name}/")
        _,g = load_weights(data,device,"H")
        mask = get_mask(g,data,args,device)

        mask_df = pd.DataFrame(np.array(mask.detach().cpu(),dtype=float),columns = list(data.colnames))
        mask_df["label"] = data.named_labels.values 
        visulaize_umap(copy.deepcopy(mask_df),"mask_df",data)
        input_df = pd.DataFrame(np.array(data.all_dataset.X_data.detach().cpu(),dtype=float),columns = list(data.colnames))
        input_df["label"] = data.named_labels.values 
        visulaize_umap(copy.deepcopy(input_df),"input",data)


def run_gsea(args,device):
    """
    Run GSEA analisys for all datasets, for all models (G,H,F2,XGB,RF) and saving results to files

    Arguments:
    args [obj] - Arguments
    device

    """
    datasets_list = load_datasets_list(args)
    # with open("./data/immunai_data_set.gmt")as gmt:
    cols = ["Data","Model","nes","pval","fdr"]
    results_df = pd.DataFrame(columns=cols)
    global_time = time()
    for i,f in enumerate(datasets_list):
        dataset_time = time()
        print(f"\n### Starting work on {f.name[:-5]} ###")
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False,test_set=True)
        for mod in ["G","H","F2"]:
            base_print = "" if mod =="G" else mod
            _,g = load_weights(data,device,base_print,only_g=True)
            mask_df = get_mask(g,data,args,device)
            #### Important criterion
            mean = mask_df.mean(dim=0)
            ten_p = torch.quantile(mean,0.85)
            val = torch.where(mean>ten_p,mean,torch.std(mask_df, dim=0))
            ###########

            rnk = pd.DataFrame(columns=["0","1"])
            rnk["0"] = data.colnames
            rnk["1"] = val.cpu()
            rnk = rnk.sort_values(by="1",ascending=False)
            pre_res = gp.prerank(rnk=rnk, gene_sets=f'./data/gmt_files/all.gmt',
                    processes=4,
                    permutation_num=100, # reduce number to speed up testing
                    no_plot =True,
                    outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all', format='png', seed=6,min_size=1, max_size=600)
            res_list = [data.data_name,mod,pre_res.res2d["nes"].values[0],pre_res.res2d["pval"].values[0],pre_res.res2d["fdr"].values[0]]
            single_res_df =pd.DataFrame([res_list],columns=cols)
            results_df = pd.concat([results_df, single_res_df])

        ###############################################
        xgb_cls,_ = load_weights(data,device,"XGB",only_g=True)
        xgb_rank = pd.DataFrame(columns=["0","1"])
        xgb_rank["0"] = data.colnames
        xgb_rank["1"] = xgb_cls.feature_importances_
        xgb_rank = xgb_rank.sort_values(by="1",ascending=False)
        pre_res_xgb = gp.prerank(rnk=xgb_rank, gene_sets=f'./data/gmt_files/all.gmt',
            processes=4,
            permutation_num=100, # reduce number to speed up testing
            no_plot =True,
            outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all_xgb', format='png', seed=6,min_size=1, max_size=600)
        res_list = [data.data_name,"XGB",pre_res_xgb.res2d["nes"].values[0],pre_res_xgb.res2d["pval"].values[0],pre_res_xgb.res2d["fdr"].values[0]]
        single_res_df =pd.DataFrame([res_list],columns=cols)
        results_df = pd.concat([results_df, single_res_df])

        ###############################################
        rf_model,_ = load_weights(data,device,"RF",only_g=True)
        rf_rank = pd.DataFrame(columns=["0","1"])
        rf_rank["0"] = data.colnames
        rf_rank["1"] = rf_model.feature_importances_
        rf_rank = rf_rank.sort_values(by="1",ascending=False)
        pre_res_rf = gp.prerank(rnk=rf_rank, gene_sets=f'./data/gmt_files/all.gmt',
            processes=4,
            permutation_num=100, # reduce number to speed up testing
            no_plot =True,
            outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all_rf', format='png', seed=6,min_size=1, max_size=600)
        res_list = [data.data_name,"RF",pre_res_rf.res2d["nes"].values[0],pre_res_rf.res2d["pval"].values[0],pre_res_rf.res2d["fdr"].values[0]]
        single_res_df =pd.DataFrame([res_list],columns=cols)
        results_df = pd.concat([results_df, single_res_df])
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("Working on {}:took {}".format(data.data_name,time_diff))
        print(f"#################################")  

    results_df.to_csv("./results/prerank/prerank_res_df_std.csv")
    time_diff = datetime.timedelta(seconds=time()-global_time)
    print("All training took: {}".format(time_diff))   
    print(f"#################################")  

    


def run_heatmap_procces(args,device):
    """
    Create heatmaps for specific genes per patients - Working only for datasets with patient data

    Arguments:
    args [obj] - Arguments
    device

    """
    datasets_list = load_datasets_list(args)
    for i,f in enumerate(datasets_list):
        dataset_time = time()
        print(f"\n### Starting work on {f.name[:-5]} ###")
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False,test_set=True)
        _,g_model = load_weights(data,device,"F2_c",only_g=True)
        dataset_loader = DataLoader(dataset=data.all_dataset,batch_size=len(data.all_dataset)//8,shuffle=False)
        cols = list(data.colnames)
        cols.append("y")
        mask_df = pd.DataFrame(columns=cols)
        print(f"Creating mask for {data.data_name}")
        with torch.no_grad():
            g_model.eval()
            for X_batch, y_batch in dataset_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mask = g_model(X_batch)
                mask.requires_grad = False

                y = np.expand_dims(np.argmax(np.array(y_batch.detach().cpu()),axis=1), axis=1)
                mask = np.concatenate((np.array(mask.detach().cpu()),y),axis=1)
            
                mask_df = pd.concat([mask_df,pd.DataFrame(mask,columns=cols)])
            mask_df= mask_df.reset_index()



            mask_df["label"] = data.named_labels.values
            if hasattr(data,"patient"):
                mask_df["patient"] = data.patient.values




        ten_p = np.quantile(mask_df[data.colnames].values.mean(axis=0),0.9)

        max_patient = data.full_data.obs.patient.value_counts()[:40].index
        mask_df = mask_df[mask_df['patient'].isin(max_patient)]
        mask_df['patient'] = np.array(mask_df['patient'])

        data_types = ["naive CD8","memory CD8","naive CD4","memory CD4"]
        for current_data_name in data_types:
            current_data = mask_df[mask_df["label"]==current_data_name]
            current_data_mean = current_data.groupby("patient")[data.colnames].agg(np.mean)
            current_data_mean.columns = data.colnames
            current_data_std = current_data.groupby("patient")[data.colnames].agg(np.std)
            current_data_std.columns = data.colnames


            current_data_mean_mean = current_data_mean.mean(axis=0).sort_values(ascending=False)
            best_genes = current_data_mean_mean[current_data_mean_mean>ten_p].index[:15]
            df = current_data_mean[best_genes]
            df = df/df.max().max()
            plt.figure(figsize=(10,15))
            plt.imshow(df.values,cmap="hot")
            plt.xticks(np.arange(0.5, len(best_genes), 1), best_genes,rotation = 90)
            plt.yticks(np.arange(0.5, df.shape[0], 1), current_data_mean.index)
            # plt.title(current_data_name,fontsize=16)
            plt.colorbar()
            plt.savefig(f"./results/heatmap_{current_data_name}.png",bbox_inches='tight',
            pad_inches=0.1,)




def run_per_sample_gsea_compare(args,device):
    """
    Run per sample GSEA analisys based on our H, ELI5 for RF or SHAP for XGB
    saving data per dataset and overall comparison

    Arguments:
    args [obj] - Arguments
    device

    """
    datasets_list = load_datasets_list(args)
    stat_df = pd.DataFrame(columns=["Data","Our mean","RF mean","XGB mean","Our std","RF std","XGB std","Our var","RF var","XGB var","RF T-test","RF P-value","XGB T-test","XGB P-value"])
    cols = ["Sample","y","Our nes","RF nes","XGB nes","Our pval","RF pval","XGB pval","Our fdr","RF fdr","XGB fdr"]
    global_time = time()
    for i,f in enumerate(datasets_list):
        dataset_time = time()
            
        results_df = pd.DataFrame(columns=cols)
        print(f"\n### Starting work on {f.name[:-5]} ###")
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False,test_set=True)
        _,g = load_weights(data,device,"H",only_g=True)
        rf_model,_ = load_weights(data,device,"RF",only_g=True)
        xgb_cls,_ = load_weights(data,device,"XGB",only_g=True)


        number_of_random_samples = 100 if data.all_dataset.X_data.shape[0]>100 else data.all_dataset.X_data.shape[0]
        random_samples =np.random.random_integers(0,data.all_dataset.X_data.shape[0]-1,size=number_of_random_samples)  
        for j,sample in enumerate(random_samples):
            print(f"\rRunning GSEA on sample {j+1}/{number_of_random_samples}")
            with torch.no_grad():
                g.eval()
                X_batch = data.all_dataset.X_data[sample]
                y = np.array(data.all_dataset.y_data[sample])
                y = np.argmax(y) 
                mask = g(torch.unsqueeze(X_batch,0).to(device))

            rnk = pd.DataFrame(columns=["0","1"])
            rnk["0"] = data.colnames
            rnk["1"] = torch.squeeze(mask).cpu()
            rnk = rnk.sort_values(by="1",ascending=False)
            pre_res = gp.prerank(rnk=rnk, gene_sets=f'./data/gmt_files/all.gmt',
                    processes=4,
                    permutation_num=100, # reduce number to speed up testing
                    no_plot =True,
                    outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all', format='png', seed=6,min_size=1, max_size=600)
           
            ####################################################################
            aux1 = eli5.sklearn.explain_prediction.explain_prediction_tree_classifier(rf_model, np.array(X_batch),feature_names=np.array(data.colnames),targets=[y])
            aux1 = eli5.format_as_dataframe(aux1).drop(0)

            rf_rnk = pd.DataFrame(columns=["0","1"])
            rf_rnk["0"] = aux1.feature
            rf_rnk["1"] = aux1.weight
            rf_rnk = rf_rnk.sort_values(by="1",ascending=False)
            rf_pre_res = gp.prerank(rnk=rf_rnk, gene_sets=f'./data/gmt_files/all.gmt',
                    processes=4,
                    permutation_num=100, # reduce number to speed up testing
                    no_plot =True,
                    outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all', format='png', seed=6,min_size=1, max_size=600)
            ####################################################################
            explainer = shap.Explainer(xgb_cls)
            shap_values = explainer(torch.unsqueeze(X_batch,0))
            xgb_rnk = pd.DataFrame(columns=["0","1"])
            xgb_rnk["0"] = data.colnames
            xgb_rnk["1"] = shap_values[0].values[:,y]
            xgb_rnk = xgb_rnk.sort_values(by="1",ascending=False)
            xgb_rnk_pre_res = gp.prerank(rnk=xgb_rnk, gene_sets=f'./data/gmt_files/all.gmt',
                    processes=4,
                    permutation_num=100, # reduce number to speed up testing
                    no_plot =True,
                    outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all', format='png', seed=6,min_size=1, max_size=600)
           

            res_list = [sample,data.named_labels[y],pre_res.res2d["nes"].values[0],rf_pre_res.res2d["nes"].values[0],xgb_rnk_pre_res.res2d["nes"].values[0]\
                ,pre_res.res2d["pval"].values[0],rf_pre_res.res2d["pval"].values[0],xgb_rnk_pre_res.res2d["pval"].values[0],pre_res.res2d["fdr"].values[0],
                rf_pre_res.res2d["fdr"].values[0],xgb_rnk_pre_res.res2d["fdr"].values[0]
                ]
            single_res_df =pd.DataFrame([res_list],columns=cols)
            results_df = pd.concat([results_df, single_res_df])
        
        if hasattr(data,"patient"):
            results_df["patient"] = data.patient.values[random_samples]

        results_df = results_df.replace(np.inf,np.NAN)
        results_df.to_csv(f'./results/prerank/{f.name[:-5]}/{f.name[:-5]}_per_sample_res.csv',index=False)
        print(f"############### Results on {number_of_random_samples} samples from {data.data_name} ############################")
        rf_stats_res = scipy.stats.ttest_rel(results_df["Our nes"].values.astype(float),results_df["RF nes"].values.astype(float),nan_policy="omit")
        print(f"RF -- T-test res:{rf_stats_res.statistic}| P-value:{rf_stats_res.pvalue}")
        xgb_stats_res = scipy.stats.ttest_rel(results_df["Our nes"].values.astype(float),results_df["XGB nes"].values.astype(float),nan_policy="omit")
        print(f"XGB -- T-test res:{xgb_stats_res.statistic}| P-value:{xgb_stats_res.pvalue}")
        our_mean = np.nanmean(results_df["Our nes"].values.astype(float))
        our_std = np.nanstd(results_df["Our nes"].values.astype(float))
        our_var = np.nanvar(results_df["Our nes"].values.astype(float))
        rf_mean = np.nanmean(results_df["RF nes"].values.astype(float))
        rf_std = np.nanstd(results_df["RF nes"].values.astype(float))
        rf_var = np.nanvar(results_df["RF nes"].values.astype(float))
        xgb_mean = np.nanmean(results_df["XGB nes"].values.astype(float))
        xgb_std = np.nanstd(results_df["XGB nes"].values.astype(float))
        xgb_var = np.nanvar(results_df["XGB nes"].values.astype(float))
        print(f"Our's: mean:{our_mean}| std:{our_std}| var:{our_var}")
        print(f"RF's: mean:{rf_mean}| std:{rf_std}| var:{rf_var}")
        print(f"XGB's: mean:{xgb_mean}| std:{xgb_std}| var:{xgb_var}")
        
        res_list = [f.name[:-5],our_mean,rf_mean,xgb_mean,our_std,rf_std,xgb_std,our_var,rf_var,xgb_var,rf_stats_res.statistic,rf_stats_res.pvalue,xgb_stats_res.statistic,xgb_stats_res.pvalue]
                
        single_stat_df =pd.DataFrame([res_list],columns=["Data","Our mean","RF mean","XGB mean","Our std","RF std","XGB std","Our var","RF var","XGB var","RF T-test","RF P-value","XGB T-test","XGB P-value"])
        stat_df = pd.concat([stat_df, single_stat_df])
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("Working on {}:took {}".format(data.data_name,time_diff))
        print(f"#################################")  
        stat_df.to_csv(f'./results/prerank/stst_res.csv',index=False)


def run_per_sample_gsea(args,device):
    """
    Run per sample GSEA analisys based on our methods for ablation
    saving data per dataset and overall results

    Arguments:
    args [obj] - Arguments
    device

    """
    datasets_list = load_datasets_list(args)
    cols = ["Sample","y","nes","pval","fdr"]
    stat_df =pd.DataFrame(columns=["data","model","Our mean","Our std","Our var"])
    global_time = time()
    for i,f in enumerate(datasets_list):
        dataset_time = time()
            
        results_df = pd.DataFrame(columns=cols)
        print(f"\n### Starting work on {f.name[:-5]} ###")
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False,test_set=True)
        number_of_random_samples = 1000 if data.all_dataset.X_data.shape[0]>1000 else data.all_dataset.X_data.shape[0]
        random_samples =np.random.random_integers(0,data.all_dataset.X_data.shape[0]-1,size=number_of_random_samples) 
        for mod in ["G","F2","F2_c"]:
            base_print = "" if mod =="G" else mod
            _,g = load_weights(data,device,base_print,only_g=True)

            for j,sample in enumerate(random_samples):
                print(f"\rRunning GSEA on sample {j+1}/{number_of_random_samples}")
                with torch.no_grad():
                    g.eval()
                    X_batch = data.all_dataset.X_data[sample]
                    y = np.array(data.all_dataset.y_data[sample])
                    y = np.argmax(y) 
                    mask = g(torch.unsqueeze(X_batch,0).to(device))

                rnk = pd.DataFrame(columns=["0","1"])
                rnk["0"] = data.colnames
                rnk["1"] = torch.squeeze(mask).cpu()
                rnk = rnk.sort_values(by="1",ascending=False)
                pre_res = gp.prerank(rnk=rnk, gene_sets=f'./data/gmt_files/all.gmt',
                        processes=4,
                        permutation_num=100, # reduce number to speed up testing
                        no_plot =True,
                        outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all', format='png', seed=6,min_size=1, max_size=600)

                res_list = [sample,data.named_labels[y],pre_res.res2d["nes"].values[0],
                    pre_res.res2d["pval"].values[0],pre_res.res2d["fdr"].values[0]]
                single_res_df =pd.DataFrame([res_list],columns=cols)
                results_df = pd.concat([results_df, single_res_df])
            
            results_df = results_df.replace(np.inf,np.NAN)
            print(f"############### {mod} Results on {number_of_random_samples} samples from {data.data_name} ############################")

            our_mean = np.nanmean(results_df["nes"].values.astype(float))
            our_std = np.nanstd(results_df["nes"].values.astype(float))
            our_var = np.nanvar(results_df["nes"].values.astype(float).astype(float))
            print(f"{mod}: mean:{our_mean}| std:{our_std}| var:{our_var}")

        
            res_list = [data.data_name,mod,our_mean,our_std,our_var]
            single_stat_df =pd.DataFrame([res_list],columns=["data","model","Our mean","Our std","Our var"])
            stat_df = pd.concat([stat_df, single_stat_df])
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("Working on {}:took {}".format(data.data_name,time_diff))
    time_diff = datetime.timedelta(seconds=time()-dataset_time)
    stat_df.to_csv(f'./results/prerank/stst_res_ablation.csv',index=False)
    
    print(f"#################################")  
    