import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import copy
import torch


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class Single_data:
    def __init__(self,parent,data_type,cnv_levels):
        self.data_name = data_type
        self.n_features = parent.n_features
        self.number_of_classes = parent.number_of_classes
        if self.data_name == "mut_important":
            self.raw = copy.deepcopy(parent.mut_raw_data)
            self.raw[self.raw > 1.] = 1.
        elif self.data_name == "cna_del":
            self.raw = copy.deepcopy(parent.cna_raw_data)
            self.raw[self.raw >= 0.0] = 0.
            if cnv_levels == 3:
                self.raw[self.raw < 0.0] = 1.0
            else:  # cnv == 5 , use everything
                self.raw[self.raw == -1.] = 0.5
                self.raw[self.raw == -2.] = 1.0
        elif self.data_name == "cna_amp":
            self.raw = copy.deepcopy(parent.cna_raw_data)
            self.raw[self.raw <= 0.0] = 0.
            if cnv_levels == 3:
                self.raw[self.raw > 0.0] = 1.0
            else:  # cnv == 5 , use everything
                self.raw[self.raw == 1.] = 0.5
                self.raw[self.raw == 2.] = 1.0
        
        all_cols_df = pd.DataFrame(index=parent.genes)

        # self.raw = self.raw.loc[:,parent.genes]
        all = self.raw.T.join(all_cols_df, how='right').T
        all = all.fillna(0)
        all = all.join(parent.response, how='inner')
        all = all[~all['response'].isnull()]

        y_all = all['response']
        all = all.drop(['response'],axis=1)

        parent.train_samples = np.intersect1d(parent.train_samples,all.index)
        parent.val_samples = np.intersect1d(parent.val_samples,all.index)
        parent.test_samples = np.intersect1d(parent.test_samples,all.index)


        x_train = all.loc[parent.train_samples]
        y_train = y_all.loc[parent.train_samples]
        x_validation = all.loc[parent.val_samples]
        y_validation = y_all.loc[parent.val_samples]
        x_test = all.loc[parent.test_samples]
        y_test = y_all.loc[parent.test_samples]

        self.train_dataset  = ClassifierDataset(torch.from_numpy(x_train.values).float(), torch.from_numpy(y_train.values).float())
        self.val_dataset   = ClassifierDataset(torch.from_numpy(x_validation.values).float(), torch.from_numpy(y_validation.values).float())
        self.test_dataset   = ClassifierDataset(torch.from_numpy(x_test.values).float(), torch.from_numpy(y_test.values).float())

        self.all_dataset = ClassifierDataset(torch.from_numpy(all.values).float(), torch.from_numpy(y_all.values).float())

        print(f"Loading {self.data_name}:\n Total {y_all.shape[0]} samples, {all.shape[1]} features")
        print(f"X_train:{x_train.shape[0]} samples || X_val:{x_validation.shape[0]} samples || X_test:{x_test.shape[0]} samples")
        print(
            "X_train:{},Y_train:{}||X_val:{},Y_val:{}||X_test:{},Y_test:{}||POS: train:{}| val:{}| test:{}".format(
                x_train.shape,
                y_train.shape, x_validation.shape, y_validation.shape, x_test.shape,
                y_test.shape, sum(y_train), sum(y_validation), sum(y_test)))

class PC_Data:
    def __init__(self,filter_genes=False,data_types = ['mut_important','cna_del','cna_amp'],intersection=True,known_sep=True,train_ratio=0.6):
        self.filter_genes = filter_genes
        self.data_types = data_types
        self.intersection = intersection
        self.train_ratio = train_ratio
        self.datasets = {}
        self.number_of_classes = 1

        

        self.data_path = "./data/pnet"
        self.mut_raw_data = pd.read_csv(os.path.join(self.data_path, "mutation_raw_data.csv"),index_col=0)
        self.cna_raw_data = pd.read_csv(os.path.join(self.data_path, "cna_raw_data.csv"),index_col=0)

        self.get_response()
        self.get_gene_list()
        self.get_splits(known_sep = known_sep)


        for key in self.data_types:
            self.datasets[key] = Single_data(self,key,cnv_levels=3)



    def get_response(self):
        self.response = pd.read_csv(os.path.join(self.data_path, "response.csv")).set_index('id')
        # self.response = self.response[~self.response['response'].isnull()]


    def get_gene_list(self):
        print("Filtering Genes")
        if self.filter_genes:
            df = pd.read_csv("data/tcga_prostate_expressed_genes_and_cancer_genes.csv", header=0)
            self.selected_genes = set(list(df['genes']))
            ####
            f = 'data/protein-coding_gene_with_coordinate_minimal.txt'
            coding_genes_df = pd.read_csv(f, sep='\t', header=None)
            coding_genes_df.columns = ['chr', 'start', 'end', 'name']
            coding_genes = set(coding_genes_df['name'].unique())
            self.selected_genes =  self.selected_genes.intersection(coding_genes)
        
        cols_list_set = []
        for c in [self.mut_raw_data,self.cna_raw_data]:    
            curr_list = set(list(c.columns))
            if self.filter_genes:
                curr_list = set.intersection(curr_list,self.selected_genes)
            cols_list_set.append(curr_list)

        if self.intersection:
            cols = set.intersection(*cols_list_set)
        else:
            cols = set.union(*cols_list_set)

        cols = np.unique(list(cols))
        print(f"After filtering using {len(cols)} genes")

        cols.sort()
        self.genes = cols
        self.n_features = len(cols)
        

        
    
    def get_splits(self,known_sep=True):
        if known_sep:
            training_set = pd.read_csv(os.path.join(self.data_path, 'splits/training_set.csv'))['id']
            validation_set = pd.read_csv(os.path.join(self.data_path, 'splits/validation_set.csv'))['id']
            testing_set = pd.read_csv(os.path.join(self.data_path, 'splits/test_set.csv'))['id']
        else:
            training_set, test_val = train_test_split(self.response["id"], shuffle=True,test_size=1-self.train_ratio ,random_state=None)
            validation_set,testing_set = train_test_split(test_val, shuffle=True,test_size=0.5 ,random_state=None)
        
        self.train_samples = training_set
        self.val_samples = validation_set
        self.test_samples = testing_set


if __name__ == '__main__':
    d = PC_Data(filter_genes=True,intersection=False)
    # d = PC_Data()
    print()