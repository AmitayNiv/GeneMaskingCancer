import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class Data:
    def __init__(self, data_path,gene_list=None,known_sep=True):


        self.mut_raw_data = pd.read_csv(os.path.join(data_path, "mutation_raw_data.csv"),index_col=0)
        self.cna_raw_data = pd.read_csv(os.path.join(data_path, "cna_raw_data.csv"),index_col=0)
        self.response = pd.read_csv(os.path.join(data_path, "response.csv"),index_col=0)













        self.validation_set = pd.read_csv(os.path.join(main_folder_path, 'data/sep/val_idx.csv')).values.squeeze()
        self.testing_set = pd.read_csv(os.path.join(main_folder_path, 'data/sep/test_idx.csv')).values.squeeze()
        self.training_set = pd.read_csv(os.path.join(main_folder_path, 'data/sep/train_idx.csv')).values.squeeze()
        self.x_all = pd.read_csv(os.path.join(main_folder_path, "data/CPCG/x_{}.csv".format(data_type)),
                                    index_col=0)
        self.samples = self.x_all.index
        self.genes = self.x_all.columns
        self.y_all = pd.read_csv(os.path.join(main_folder_path, "Data/CPCG/y_all.csv"), index_col=0, header=None)

        self.x_train = self.x_all.iloc[self.training_set].values
        self.y_train = self.y_all.iloc[self.training_set].values  # .astype(int)
        self.x_validate_ = self.x_all.iloc[self.validation_set].values
        self.y_validate_ = self.y_all.iloc[self.validation_set].values  # .astype(int)
        self.x_test = self.x_all.iloc[self.testing_set].values
        self.y_test = self.y_all.iloc[self.testing_set].values  # .astype(int)


        print("Load {} data".format(data_type))
        print(self.x_all.columns)
        print('training shape: ')
        print(
            "X_train:{},Y_train:{}||X_val:{},Y_val:{}||X_test:{},Y_test:{}||POS: train:{}| val:{}| test:{}".format(
                self.x_train.shape,
                self.y_train.shape, self.x_validate_.shape, self.y_validate_.shape, self.x_test.shape,
                self.y_test.shape, sum(self.y_train), sum(self.y_validate_), sum(self.y_test)))


class CPCG_Data:
    def __init__(self,data_type,train_ratio=0.6):
        self.x_all = pd.read_csv(os.path.join(main_folder_path,"Data/CPCG/x_{}_CPCG.csv".format(data_type)),index_col=0)
        print(self.x_all.columns)
        self.y_all = pd.read_csv(os.path.join(main_folder_path,"Data/CPCG/y_all_CPCG.csv"),index_col=0,header=None)

        self.samples = self.x_all.index
        self.genes = np.array(self.x_all.columns)
        # if order!=None:
        #     self.x_all = self.x_all.iloc[order]
        #     self.y_all = self.y_all.iloc[order]
        index_pos = np.where(self.y_all.values == 1)[0]
        index_neg = np.where(self.y_all.values == 0)[0]
        n_pos = index_pos.shape[0]
        n_neg = index_neg.shape[0]


        # select the same number of samples as the positive class
        # index_neg = index_neg[0:n_pos]

        ####
        val_ratio = (100 - 100*train_ratio)/200 # 0.1
        print("Train/Val/Test: {}/{}/{}".format(train_ratio*100,val_ratio*100,val_ratio*100))
        index_pos_train = index_pos[:int(n_pos*train_ratio)]
        index_neg_train = index_neg[:int(n_neg*train_ratio)]
        index_pos_val = index_pos[int(n_pos*train_ratio):int(n_pos*train_ratio)+int(n_pos*val_ratio)+1]
        index_neg_val = index_neg[int(n_neg*train_ratio):int(n_neg*train_ratio)+int(n_neg*val_ratio)+1]
        index_pos_test = index_pos[int(n_pos*train_ratio)+int(n_pos*val_ratio)+1:]
        index_neg_test = index_neg[int(n_neg*train_ratio)+int(n_neg*val_ratio)+1:]


        x_train_pos = self.x_all.values[index_pos_train, :]
        x_train_neg = self.x_all.values[index_neg_train, :]
        self.x_train = np.concatenate((x_train_neg,x_train_pos))
        y_train_pos = self.y_all.values[index_pos_train, :]
        y_train_neg = self.y_all.values[index_neg_train, :]
        self.y_train = np.concatenate((y_train_neg,y_train_pos))


        x_val_pos = self.x_all.values[index_pos_val, :]
        x_val_neg = self.x_all.values[index_neg_val, :]
        self.x_validate_ = np.concatenate((x_val_pos, x_val_neg))
        y_val_pos = self.y_all.values[index_pos_val, :]
        y_val_neg = self.y_all.values[index_neg_val, :]
        self.y_validate_ = np.concatenate((y_val_pos, y_val_neg))



        x_test_pos = self.x_all.values[index_pos_test, :]
        x_test_neg = self.x_all.values[index_neg_test, :]
        self.x_test = np.concatenate((x_test_pos, x_test_neg))
        y_test_pos = self.y_all.values[index_pos_test, :]
        y_test_neg = self.y_all.values[index_neg_test, :]
        self.y_test = np.concatenate((y_test_pos, y_test_neg))

        print(self.x_all.columns)
        # print(self.samples[index_pos_test])
        # print(self.samples[index_neg_test])
        print ('training shape: ')
        print ("X_train:{},Y_train:{}||X_val:{},Y_val:{}||X_test:{},Y_test:{}||POS: train:{}| val:{}| test:{}".format(self.x_train.shape, 
              self.y_train.shape, self.x_validate_.shape, self.y_validate_.shape, self.x_test.shape,self.y_test.shape, sum(self.y_train),sum(self.y_validate_),sum(self.y_test)))

    
        print("Load {} data".format(data_type))