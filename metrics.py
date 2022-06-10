from sklearn import metrics
import torch
import torch.nn.functional as F
import numpy as np


# def evaluate(y_test, y_pred_score):
#     with torch.no_grad():
#         # y_pred_score = torch.softmax(y_pred_score, dim = 1)
#         # _, y_pred_tags = torch.max(y_pred_score, dim = 1)
#         # y_pred = F.one_hot(y_pred_tags,num_classes = y_test.shape[1])
#         y_pred = (y_pred_score>0.5).astype(int)
#         try:
#             y_test = y_test.cpu()
#         except:
#             pass
#         y_pred = y_pred.cpu()
#         accuracy = metrics.accuracy_score(y_test, y_pred)


#         aupr = metrics.average_precision_score(y_test, y_pred_score.detach().cpu())
#         fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_score.detach().cpu(), pos_label=1)

        
#         positive_indexes = torch.where(torch.as_tensor(y_test).sum(axis=0)>0,True,False)
#         auc = metrics.roc_auc_score(y_test[:,positive_indexes], y_pred_score[:,positive_indexes].detach().cpu())
#         auc_weighted = metrics.roc_auc_score(y_test[:,positive_indexes], y_pred_score[:,positive_indexes].detach().cpu(),average="weighted")
#         auprs_weighted = metrics.average_precision_score(y_test[:,positive_indexes], y_pred_score[:,positive_indexes].detach().cpu(),average="weighted")
        
#         conf_mat = metrics.multilabel_confusion_matrix(y_test, y_pred)
#         acc_vec = []
#         for mat in conf_mat:
#             acc_vec.append(np.sum(mat.diagonal()) / np.sum(mat))
        
#         score = {}
#         score['accuracy'] = accuracy
#         score['weight_accuracy'] = (np.array(acc_vec)*np.array(y_test).sum(axis=0)/np.array(y_test).sum()).sum()
        
#         score['mauc'] = np.nanmean(auc)
#         score['weight_auc'] = auc_weighted
#         score['med_auc'] = np.nanmedian(auc)
#         score['maupr'] = np.nanmean(aupr)
#         score['weight_aupr'] = auprs_weighted
#         score['med_aupr'] = np.nanmedian(aupr)
#         score['maccuracy'] = np.mean(np.array(acc_vec))
        
#         # assert auprs_weighted == score['maupr']
#         return score

def evaluate(y_test, y_pred_score):
    with torch.no_grad():
        y_pred = torch.where(y_pred_score>0.5,1, 0)
        try:
            y_test = y_test.cpu()
        except:
            pass
        y_pred = y_pred.cpu()
        accuracy =  metrics.accuracy_score(y_test, y_pred)
        # print(y_pred_score)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_score.detach().cpu(), pos_label=1)

        auc = metrics.auc(fpr, tpr)
        auc_other = metrics.roc_auc_score(y_test, y_pred_score.detach().cpu())

        f1 = metrics.f1_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        aupr = metrics.average_precision_score(y_test, y_pred_score.detach().cpu())
        score = {}
        score['accuracy'] = accuracy
        score['auc'] = auc
        score['aupr'] = aupr
        score['precision'] = precision
        score['f1'] = f1
        score['recall'] = recall
        return score