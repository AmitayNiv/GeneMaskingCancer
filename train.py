import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import xgboost as xgb
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from models import Classifier,G_Model
from metrics import evaluate


def train_classifier(args,device,data_obj,model,wandb_exp):
    """
    Train F classifier

    Arguments:
    args [obj] - Arguments
    device
    data_obj [obj] - Data object
    model [obj] - pretraind/initialized model to train
    wandn_exp - None/ weight and baises object

    Return:
    best_model [obj] - trained F model
    res_dict [dict] -  model results on test set
    """
    print("\nStart training classifier")
    train_losses = []
    val_losses =[]
    best_model_auc = 0


    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=args.batch_size)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))



    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.cls_lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=float(args.cls_lr), steps_per_epoch=len(train_loader)//args.batch_factor+1, epochs=args.cls_epochs)

    if args.wandb_exp:
        wandb_exp.config.update({"num_of_features":data_obj.n_features,"num_of_classes":data_obj.number_of_classes,"n_train":data_obj.train_samples.shape[0],
        "n_val":data_obj.val_samples.shape[0],"n_test":data_obj.test_samples.shape[0]})


    global_step = 0
    for e in range(args.cls_epochs):
        model.train()
        train_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            

            y_train_pred = model(X_train_batch)
            loss = criterion(y_train_pred, y_train_batch)
            loss.backward()

            
            if (global_step + 1) % args.batch_factor == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            
            train_loss += loss.item()

            global_step += 1
        else:
            val_loss = 0
        
            with torch.no_grad():
                model.eval()
                val_epoch_loss = 0
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    y_val_pred = model(X_val_batch)     
                    val_loss = criterion(y_val_pred, y_val_batch)
                    val_epoch_loss += val_loss.item()
                    val_score = evaluate(y_val_batch, y_val_pred)

    

            print("Epoch: {}/{}.. ".format(e+1, args.cls_epochs),
                "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                "Val AUC: {:.5f}.. ".format(val_score["auc"]),
                "Val AUPR: {:.5f}.. ".format(val_score["aupr"]),
                "Val F1: {:.5f}.. ".format(val_score["f1"]),
                "Val Precision: {:.5f}.. ".format(val_score["precision"]),
                "Val Recall: {:.5f}.. ".format(val_score["recall"]),
                "Val ACC: {:.5f}.. ".format(val_score["accuracy"]))
            
            if args.wandb_exp:
                wandb_exp.log({"epoch":e+1,"train loss":train_loss/len(train_loader),"Val mAUC":val_score["mauc"],"Val medAUC":val_score["med_auc"],
                "Val mAUPR":val_score["maupr"],"Val wegAUPR":val_score["weight_aupr"],"Val medAUPR":val_score["med_aupr"],
                "Val Accuracy":val_score["accuracy"],"lr":optimizer.state_dict()["param_groups"][0]["lr"]
                })

            if val_score["auc"] >best_model_auc:
                best_model_auc = val_score["auc"]
                best_model_res = val_score.copy()
                best_model_index = e+1
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                best_model = copy.deepcopy(model)
    if args.wandb_exp:
        wandb_exp.log({"Val mAUC":best_model_res["mauc"],"Val medAUC":best_model_res["med_auc"],
        "Val mAUPR":best_model_res["maupr"],"Val wegAUPR":best_model_res["weight_aupr"],"Val medAUPR":best_model_res["med_aupr"],"Val Accuracy":best_model_res["accuracy"]
        })
    best_model = best_model.to(device)
    with torch.no_grad():
        best_model.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            y_pred_score = best_model(X_test_batch)
            test_score = evaluate(y_test_batch, y_pred_score)
            print("Classifier test results:")
            print(test_score)
        res_dict = {
            "F AUC":test_score["auc"],"F AUPR":test_score["aupr"],"F F1":test_score["f1"],
            "F Precision":test_score["precision"],"F Recall":test_score["recall"],
            "F Accuracy":test_score["accuracy"]
            }
        if args.wandb_exp:
            wandb_exp.config.update(res_dict)
    return best_model,res_dict


def train_G(args,device,data_obj,classifier,model=None,wandb_exp=None):
    """
    Train G model

    Arguments:
    args [obj] - Arguments
    device
    data_obj [obj] - Data object
    classifier [obj] - pretraind F model 
    wandn_exp - None/ weight and baises object

    Return:
    best_G_model [obj] - trained G model
    res_dict [dict] -  model results on test set
    """
    print("\nStart training G model")
    classifier.eval()

    train_losses = []
    val_losses_list =[]
    best_model_auc = 0



    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=args.batch_size)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(model.parameters(), lr=args.g_lr,weight_decay=args.weight_decay)
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G,max_lr=float(args.g_lr),steps_per_epoch=len(train_loader)//args.batch_factor+1, epochs=args.g_epochs)


    if args.wandb_exp:
        wandb_exp.config.update({"num_of_features":data_obj.n_features,"num_of_classes":data_obj.number_of_classes,"n_train":data_obj.train_samples.shape[0],
        "n_val":data_obj.val_samples.shape[0],"n_test":data_obj.test_samples.shape[0]})

    global_step = 0
    for e in range(args.g_epochs):
        classifier.eval()
        train_loss = 0
        loss_list = []
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            mask = model(X_train_batch)
            cropped_features = X_train_batch * mask
            output = classifier(cropped_features)
            loss = criterion(output, y_train_batch) + mask.mean()
            loss.backward()
            
            if (global_step + 1) % args.batch_factor == 0:
                optimizer_G.step()
                optimizer_G.zero_grad(set_to_none=True)
                scheduler_G.step()
            train_loss += loss.item()
            global_step +=1
        else:
            val_losses = 0
        
            with torch.no_grad():
                model.eval()
                classifier.eval()
                val_epoch_loss = 0
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                    mask_val = model(X_val_batch)   
                    print("mask mean = {:.5f}, max= {:.5f}, min = {:.5f}".format(mask_val.mean(),mask_val.max(),mask_val.min()))
                    cropped_features = X_val_batch * mask_val
                    output = classifier(cropped_features)
                    val_loss = criterion(output, y_val_batch) + mask_val.mean()
                    val_losses +=  val_loss.item()

                    val_score = evaluate(y_val_batch, output)
                    print("Epoch: {}/{}.. ".format(e+1, args.g_epochs),
                    "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                    "Val AUC: {:.5f}.. ".format(val_score["auc"]),
                    "Val AUPR: {:.5f}.. ".format(val_score["aupr"]),
                    "Val F1: {:.5f}.. ".format(val_score["f1"]),
                    "Val Precision: {:.5f}.. ".format(val_score["precision"]),
                    "Val Recall: {:.5f}.. ".format(val_score["recall"]),
                    "Val ACC: {:.5f}.. ".format(val_score["accuracy"]))

                if args.wandb_exp:
                    wandb_exp.log({"epoch":e+1,"G train loss":train_loss/len(train_loader),"G Val mAUC":val_score["mauc"],"G Val medAUC":val_score["med_auc"],
                    "G Val mAUPR":val_score["maupr"],"G Val wegAUPR":val_score["weight_aupr"],"G Val medAUPR":val_score["med_aupr"],
                    "G Val Accuracy":val_score["accuracy"],"lr":optimizer_G.state_dict()["param_groups"][0]["lr"]
                    })
                if val_score["auc"] >best_model_auc:
                    best_model_auc = val_score["auc"]
                    best_model_res = val_score.copy()
                    best_model_index = e+1
                    print("Model Saved,Epoch = {}".format(best_model_index))
                    best_G_model = copy.deepcopy(model)
                    # torch.save(model,best_model_path)	
    if args.wandb_exp:
        wandb_exp.log({"G Val mAUC":best_model_res["mauc"],"G Val medAUC":best_model_res["med_auc"],
        "G Val mAUPR":best_model_res["maupr"],"G Val wegAUPR":best_model_res["weight_aupr"],
        "G Val medAUPR":best_model_res["med_aupr"],"G Val Accuracy":best_model_res["accuracy"]
        })   
    with torch.no_grad():
        best_G_model.eval()
        classifier.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            # print(f"############### Results on {data_obj.data_name} ############################")
            # print("Results without G")
            # y_pred_score = classifier(X_test_batch)
            # cls_test_score = evaluate(y_test_batch,y_pred_score)
            # print(cls_test_score)
            print("G Test Results")
            mask_test = best_G_model(X_test_batch)
            cropped_features = X_test_batch * mask_test
            y_pred_score = classifier(cropped_features)
            test_score = evaluate(y_test_batch, y_pred_score)
            print(test_score)

        res_dict = {
            "G AUC":test_score["auc"],"G AUPR":test_score["aupr"],"G F1":test_score["f1"],
            "G Precision":test_score["precision"],"G Recall":test_score["recall"],
            "G Accuracy":test_score["accuracy"]
            }
        if args.wandb_exp:
            wandb_exp.config.update(res_dict)
    return best_G_model, res_dict



def train_H(args,device,data_obj,g_model,model=None,wandb_exp=None,train_H=True):
    """
    Train H model or F2 model (for ablation study)

    Arguments:
    args [obj] - Arguments
    device
    data_obj [obj] - Data object
    g_model [obj] - pretraind G model 
    model [obj] - pretraind/Initialized F model 
    wandn_exp - None/ weight and baises object
    train_H [binary] - Indicate if training H or F2

    Return:
    best_model [obj] - trained H model
    best_G_model [obj] - trained G model
    res_dict [dict] -  model results on test set
    """
    if train_H: 
        model_name = "H"
    else:
        model_name = "F2"

    print(f"\nStart training {model_name} classifier")
    train_losses = []
    val_losses =[]
    best_model_auc = 0
    g_model.to(device)


    train_loader = DataLoader(dataset=data_obj.train_dataset,batch_size=args.batch_size)
    val_loader = DataLoader(dataset=data_obj.val_dataset, batch_size=len(data_obj.val_dataset))
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))


    if model == None:
        n_features = data_obj.n_features*2 if train_H else data_obj.n_features
        model = Classifier(n_features ,dropout=args.dropout,number_of_classes=data_obj.number_of_classes)
        model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.cls_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=float(args.cls_lr), steps_per_epoch=len(train_loader)//args.batch_factor+1, epochs=args.cls_epochs)
    optimizer_G = optim.Adam(g_model.parameters(), lr=args.g_lr/10,weight_decay=args.weight_decay)
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G,max_lr=float(args.g_lr)/10,steps_per_epoch=len(train_loader)//(args.batch_factor*4)+1, epochs=args.g_epochs)


    global_step = 0

    
    for e in range(args.cls_epochs):
        g_model.train()
        model.train()
        train_loss = 0
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            
            mask = g_model(X_train_batch)
            cropped_features = X_train_batch*mask
            if train_H:
                X_train_batch_bin = torch.where(X_train_batch==0, 1, 0)
                cropped_features_neg = X_train_batch_bin *mask
                cropped_features = torch.concat((cropped_features,cropped_features_neg),dim=1)
            y_train_pred = model(cropped_features)
            loss = criterion(y_train_pred, y_train_batch)+ mask.mean()
            loss.backward()

            
            if (global_step + 1) % args.batch_factor == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (global_step + 1) % (args.batch_factor*8) == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()
                scheduler_G.step()
                

            
            train_loss += loss.item()

            global_step += 1
        else:
            val_loss = 0
        
            with torch.no_grad():
                model.eval()
                g_model.eval()
                val_epoch_loss = 0
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                    mask = g_model(X_val_batch)
                    print("mask mean = {:.5f}, max= {:.5f}, min = {:.5f}".format(mask.mean(),mask.max(),mask.min()))
                    cropped_features = X_val_batch*mask
                    if train_H:
                        X_val_batch_bin = torch.where(X_val_batch==0, 1, 0)
                        cropped_features_neg = X_val_batch_bin *mask
                        cropped_features = torch.concat((cropped_features,cropped_features_neg),dim=1)
                    y_val_pred = model(cropped_features)

                    val_loss = criterion(y_val_pred, y_val_batch)+ mask.mean()
                    val_epoch_loss += val_loss.item()
                    val_score = evaluate(y_val_batch, y_val_pred)

                    print("Epoch: {}/{}.. ".format(e+1, args.cls_epochs),
                    "Training Loss: {:.5f}.. ".format(train_loss/len(train_loader)),
                    "Val AUC: {:.5f}.. ".format(val_score["auc"]),
                    "Val AUPR: {:.5f}.. ".format(val_score["aupr"]),
                    "Val F1: {:.5f}.. ".format(val_score["f1"]),
                    "Val Precision: {:.5f}.. ".format(val_score["precision"]),
                    "Val Recall: {:.5f}.. ".format(val_score["recall"]),
                    "Val ACC: {:.5f}.. ".format(val_score["accuracy"]))
            if val_score["auc"] >best_model_auc:
                best_model_auc = val_score["auc"]
                best_model_index = e+1
                print("Model Saved, Auc ={:.4f} , Epoch ={}".format(best_model_auc,best_model_index))
                best_model = copy.deepcopy(model)
                best_g = copy.deepcopy(g_model)


    best_model = best_model.to(device)
    with torch.no_grad():
        best_model.eval()
        best_g.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)

            mask = best_g(X_test_batch)
            cropped_features = X_test_batch*mask
            if train_H:
                X_test_batch_bin = torch.where(X_test_batch==0, 1, 0)
                cropped_features_neg = X_test_batch_bin *mask
                cropped_features = torch.concat((cropped_features,cropped_features_neg),dim=1)

            y_pred_score = best_model(cropped_features)
            test_score = evaluate(y_test_batch, y_pred_score)
            print(f"{model_name} test results:")
            print(test_score)
            res_dict = {
            f"{model_name} AUC":test_score["auc"],f"{model_name} AUPR":test_score["aupr"],f"{model_name} F1":test_score["f1"],
            f"{model_name} Precision":test_score["precision"],f"{model_name} Recall":test_score["recall"],
            f"{model_name} Accuracy":test_score["accuracy"]
            }
    return best_model,best_g,res_dict


def train_xgb(data_obj,device):
    """
    Train XGBoost model

    Arguments:
    data_obj [obj] - Data object
    device

    Return:
    xgb_cl [obj] - trained XGB model
    res_dict [dict] -  model results on test set
    """
    print("\nStart training XGBoost")
    xgb_cl = xgb.XGBClassifier()
    X_train = np.array(data_obj.train_dataset.X_data)

    y_train = np.array(data_obj.train_dataset.y_data)
    # y_train = np.argmax(y_train,axis=1)
    xgb_cl.fit(X_train, y_train)

    print(f"XGB Test Results on {data_obj.data_name}")
    X_test = np.array(data_obj.test_dataset.X_data)
    y_test = np.array(data_obj.test_dataset.y_data)
    y_pred_score =  xgb_cl.predict_proba(X_test)
    y_pred_score = torch.from_numpy(y_pred_score).to(device)
    # indexes = y_pred_score.max(dim=1).indices
    y_pred_score = y_pred_score[:,1]
    test_score = evaluate(y_test,y_pred_score)
    print(test_score)
    res_dict = {
            "XGB AUC":test_score["auc"],"XGB AUPR":test_score["aupr"],"XGB F1":test_score["f1"],
            "XGB Precision":test_score["precision"],"XGB Recall":test_score["recall"],
            "XGB Accuracy":test_score["accuracy"]
            }
    return xgb_cl,res_dict

def train_random_forest(data_obj,device):
    """
    Train Random Forest model

    Arguments:
    data_obj [obj] - Data object
    device

    Return:
    cls [obj] - trained RF model
    res_dict [dict] -  model results on test set
    """
    print("\nStart training Random Forest Classifier")
    clf = RandomForestClassifier(n_estimators=100)
    X_train = np.array(data_obj.train_dataset.X_data)
    y_train = np.array(data_obj.train_dataset.y_data)
    # y_train = np.argmax(y_train,axis=1)
    clf.fit(X_train, y_train)

    print(f"Random Forest Test Results on {data_obj.data_name}")
    X_test = np.array(data_obj.test_dataset.X_data)
    y_test = np.array(data_obj.test_dataset.y_data)
    y_pred_score =  clf.predict_proba(X_test)
    y_pred_score = torch.from_numpy(y_pred_score).to(device)
    y_pred_score = y_pred_score[:,1]
    test_score = evaluate(y_test,y_pred_score)
    print(test_score)
    res_dict = {
            "RF AUC":test_score["auc"],"RF AUPR":test_score["aupr"],"RF F1":test_score["f1"],
            "RF Precision":test_score["precision"],"RF Recall":test_score["recall"],
            "RF Accuracy":test_score["accuracy"]
            }
    return clf,res_dict


