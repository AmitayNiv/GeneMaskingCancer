import torch
from torch.utils.data import DataLoader
import numpy as np

from metrics import evaluate

def test(classifier,g_model,device,data_obj):
    """
    test models on test data

    Arguments:
    classifier [obj] - trained F/H model
    g_model [obj] - trained G model
    data_obj [obj] - Data object
    device


    """
    test_loader = DataLoader(dataset=data_obj.test_dataset, batch_size=len(data_obj.test_dataset))
    print(f"Test results on {data_obj.data_name} dataset")
    with torch.no_grad():
        g_model.eval()
        classifier.eval()
        for X_test_batch, y_test_batch in test_loader:
            X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
            print("Results without G")
            y_pred_score = classifier(X_test_batch)
            test_score = evaluate(y_test_batch,y_pred_score)
            print(test_score)
            print("Results with G")
            mask_test = g_model(X_test_batch)
            cropped_features = X_test_batch * mask_test
            y_pred_score = classifier(cropped_features)
            test_score = evaluate(y_test_batch, y_pred_score)
            print(test_score)

def test_xgb(xgb_cls,data_obj,device):
    """
    test XGB model on test data

    Arguments:
    xgb_cls [obj] - trained XGB model
    data_obj [obj] - Data object
    device

    """
    print(f"XGB Test Results on {data_obj.data_name}")
    X_test = np.array(data_obj.test_dataset.X_data)
    y_test = np.array(data_obj.test_dataset.y_data)
    y_pred_score =  xgb_cls.predict_proba(X_test)
    y_pred_score = torch.from_numpy(y_pred_score).to(device)
    test_score = evaluate(y_test,y_pred_score)
    print(test_score)

