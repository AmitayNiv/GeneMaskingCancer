from pkg_resources import safe_name
from pyparsing import col
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models import Classifier, G_Model
import matplotlib.pyplot as plt
import copy
import os
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import gseapy as gp
from data_loading import PC_Data
from time import time
import datetime
import eli5


def get_mask(g_model,data_obj,args,device,bin_mask=False):
    """
    Run inference of G model on entire dataset and return the provided mask by G

    Arguments:
    g_model [obj] - G model
    data_obj [obj] - Data object
    args [obj] - arguments
    device 
    bin_mask [binary] - If to apply TH on mask results

    Return:
    mask_arr [torch.tensor] - G output mask
    """
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset)//16,shuffle=False)
    print(f"Creating mask for {data_obj.data_name}")
    first_batch = True
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = g_model(X_batch)
            if bin_mask:
                mask = torch.where(mask>0.5,1,0)
            mask.requires_grad = False
            if first_batch:
                mask_arr = mask
                first_batch = False
            else:
                mask_arr = torch.cat((mask_arr,mask), 0)
    return mask_arr



def get_mask_and_mult(g_model,data_obj,args,device,bin_mask=False):
    """
    Run inference of G model on entire dataset and return the provided mask by G, adding lable vector and patient vector if exists

    Arguments:
    g_model [obj] - G model
    data_obj [obj] - Data object
    args [obj] - arguments
    device 
    bin_mask [binary] - If to apply TH on mask results

    Return:
    mask_arr [pandas DF] - G output mask
    mask_x_df [pandas DF] - G output mask multiplied by the input 
    input_df [pandas DF] - input data
    """
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset)//8,shuffle=False)
    cols = list(data_obj.colnames)
    double_cols = copy.deepcopy(cols)
    double_cols.extend(cols)
    double_cols.append("y")
    cols.append("y")
    mask_df = input_df = pd.DataFrame(columns=cols)
    mask_x_df = pd.DataFrame(columns=double_cols)
    print(f"creating mask for {data_obj.data_name}")
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = g_model(X_batch)
            if bin_mask:
                mask = torch.where(mask>0.5,1,0)
            cropped_features = X_batch*mask
            X_test_batch_bin = torch.where(X_batch==0, 1, 0)

            cropped_features_neg = X_test_batch_bin *mask
            cropped_features = torch.concat((cropped_features,cropped_features_neg),dim=1)

            
            y = np.expand_dims(np.argmax(np.array(y_batch.detach().cpu()),axis=1), axis=1)
            mask = np.concatenate((np.array(mask.detach().cpu()),y),axis=1)
            cropped_features = np.concatenate((np.array(cropped_features.detach().cpu()),y),axis=1)
            input_x = np.concatenate((np.array(X_batch.detach().cpu()),y),axis=1)

            mask_df = pd.concat([mask_df,pd.DataFrame(mask,columns=cols)])
            mask_x_df = pd.concat([mask_x_df,pd.DataFrame(cropped_features,columns=double_cols)])
            input_df = pd.concat([input_df,pd.DataFrame(input_x,columns=cols)])
    
    mask_df,mask_x_df,input_df = mask_df.reset_index(),mask_x_df.reset_index(),input_df.reset_index()



    mask_df["label"]= mask_x_df["label"] = input_df["label"] = data_obj.named_labels.values
    if hasattr(data_obj,"patient"):
        mask_df["patient"]= mask_x_df["patient"] = input_df["patient"] = data_obj.patient.values

    return mask_df,mask_x_df,input_df


def init_models(args,data,device,base = ""):
    """
    Init G and F models

    Arguments:
    data [obj] - Data object
    args [obj] - arguments
    device 
    base [str] - name of the model to load ("" for F and G)

    Return:
    cls [obj] - Classifier
    g_model [obj] - related G model
    """
    if args.load_pretraind_weights:
        cls,g_model = load_weights(data,device,base)
    else:
        print("Initializing classifier")
        cls = Classifier(data.n_features ,dropout=args.dropout,number_of_classes=data.number_of_classes,first_division=2)
        cls = cls.to(device)
        print("Initializing G model")
        g_model = G_Model(data.n_features,first_division=2)
        g_model = g_model.to(device)
    return cls,g_model
    



def save_weights(cls,g,data,base = ""):
    """
    Save weights of models

    Arguments:
    cls [obj] - Classifier
    g [obj] - G model
    data [obj] - Data object
    base [str] - name of the model to load ("" for F and G)

    """
    base_print = base+"_" if base != "" else base
    if not os.path.exists(f"./weights/{data.data_name}/"):
        os.mkdir(f"./weights/{data.data_name}/")
    if base=="XGB":
        cls.save_model(f"./weights/{data.data_name}/{base}.json")
    elif base=="RF":
        joblib.dump(cls, f"./weights/{data.data_name}/{base}.joblib")
    else:
        torch.save(cls,f"./weights/{data.data_name}/{base_print}cls.pt")
        torch.save(g,f"./weights/{data.data_name}/{base_print}g.pt")
    print(f"{base} Models was saved to ./weights/{data.data_name}")


def load_weights(data,device,base = "",only_g=False):
    """
    Loading weights of models

    Arguments:
    data [obj] - Data object
    base [str] - name of the model to load ("" for F and G)
    device 
    only_g [binary] -  Indicate if load only G model 

    
    Return:
    cls [obj] - Classifier
    g_model [obj] - related G model

    """
    base_print = base+"_" if base != "" else base
    if base =="XGB":
        print(f"Loading pre-trained weights for {base} classifier {data.data_name}")
        cls = xgb.XGBClassifier()
        cls.load_model(f"./weights/{data.data_name}/{base}.json")
        g_model = None
    elif base =="RF":
        print(f"Loading pre-trained weights for {base} classifier {data.data_name}")
        cls = joblib.load(f"./weights/{data.data_name}/{base}.joblib")
        g_model = None
    else:
        if only_g:
            cls = None
        else:
            print(f"Loading pre-trained weights for {base} classifier {data.data_name}")
            cls = torch.load(f"./weights/{data.data_name}/{base_print}cls.pt").to(device)
        print(f"Loading pre-trained weights for {base} G model {data.data_name}")
        g_model = torch.load(f"./weights/{data.data_name}/{base_print}g.pt").to(device)
    return cls,g_model


def concat_average_dfs(aux2,aux3):
    try:
        aux2.set_index(['feature', 'target'],inplace = True)
    except:
        pass
    try:
        aux3.set_index(['feature', 'target'],inplace = True)
    except:
        pass
    # Concatenating and creating the meand
    aux = pd.DataFrame(pd.concat([aux2['weight'],aux3['weight']]).groupby(level = [0,1]).mean())
    return aux


def get_tree_explaination(data):
    """
    Create gene important list base on eli5 library

    Arguments:
    data [obj] - Data object


    
    Return:
    rf_important [pd Dataframe] - Random forest feature important per sample

    """
    cols = data.colnames
    cols.append("y")
    rf_important = pd.DataFrame(columns=cols)
    ###############################################
    rf_model,_ = load_weights(data,None,"RF",only_g=True)
    X = np.array(data.all_dataset.X_data)

    y = np.array(data.all_dataset.y_data)
    y = np.argmax(y,axis=1)
    for sample in range(X.shape[0]):
        aux1 = eli5.sklearn.explain_prediction.explain_prediction_tree_classifier(rf_model,X[sample],feature_names=np.array(data.colnames),targets=[y[sample]])
        aux1 = eli5.format_as_dataframe(aux1).drop(0)
        sample_important = pd.DataFrame(np.zeros((1,len(cols))),columns=cols)
        sample_important[aux1.feature.values]=aux1.weight.values
        sample_important["y"] = y[sample]

        rf_important = pd.concat([rf_important,sample_important],axis=1)
    return rf_important

             
