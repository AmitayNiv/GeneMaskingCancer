import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pkg_resources import safe_name
import umap
from utils import load_datasets_list


def visulaize_tsne(table,table_name,dataset,wandb_exp=None):


    feat_cols = dataset.colnames

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    current_data_set = table
    data_subset = current_data_set[feat_cols].values

    tsne_results = tsne.fit_transform(data_subset)
    current_data_set['tsne-2d-one'] = tsne_results[:,0]
    current_data_set['tsne-2d-two'] = tsne_results[:,1]


    fig, ax = plt.subplots(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hls",15),
    data=current_data_set,
    legend="full",
    alpha=0.3).set(title=f"{table_name}|{dataset.data_name}|#samples:{current_data_set.shape[0]}")
    data_set_name_png =f"tsne_{table_name}.png"
    res_folder_path = f"./results/{dataset.data_name}/"
    plt.savefig(os.path.join(res_folder_path,data_set_name_png))
    plt.cla()
    plt.close("all")





def visulaize_umap(table,table_name,dataset):
    print(f'Craeting UMAP projection of the {table_name}| dataset:{dataset.data_name}')
    feat_cols = dataset.colnames
    reducer = umap.UMAP(random_state=42)
    
    current_data_set = table#.loc[(table['label']=="memory CD8") | (table['label']=="naive CD8")]
    data_subset = current_data_set[feat_cols].values

    reducer.fit(data_subset)
    embedding = reducer.transform(data_subset)
    
    current_data_set['embedding_0'] = embedding[:,0]
    current_data_set['embedding_1'] = embedding[:,1]


    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='embedding_0', y='embedding_1',
        hue="label",
        palette=sns.color_palette("hls",15),
        data=current_data_set,
        legend="full",
        alpha=0.3)
    plt.title(f'UMAP projection of the {table_name}| dataset:{dataset.data_name}', fontsize=16)


    data_set_name_png =f"umap_{table_name}.png"
    res_folder_path = f"./results/{dataset.data_name}/"
    plt.savefig(os.path.join(res_folder_path,data_set_name_png))

    plt.cla()
    plt.close("all")



def visulaize_gsne_per_sample_res(args):
    datasets_list = load_datasets_list(args)
    for i,f in enumerate(datasets_list):
        df = pd.read_csv(f'./results/prerank/{f.name[:-5]}/{f.name[:-5]}_per_sample_res.csv')
        rf_sr= 1-(np.isnan(df["RF nes"].values).sum()/df["RF nes"].shape[0])
        our_sr= 1-(np.isnan(df["Our nes"].values).sum()/df["RF nes"].shape[0])
        print("RF sr:{:.3f}| our sr:{:.3f} #### {}".format(rf_sr,our_sr,f.name[:-19]) )


        df.replace(np.nan,-2.5)
        df["RF nes"].values[np.where(np.isnan(df["RF nes"].values))]=-2.5
        df["Our nes"].values[np.where(np.isnan(df["Our nes"].values))]=-2.5

        labels =list(np.unique(df["y"].values))

        fig = plt.figure(figsize=(15,15))
        plt.vlines(x=-2.5, ymin=-2.5, ymax=2.5,color="black",linestyles="dashed",label="inf")
        plt.text(x=-2.45, y=2.4, s="inf",color="black")
        plt.xlabel("RF NES",fontsize= 22)
        plt.ylabel("Our NES",fontsize= 22)
        plt.plot([-2.5,2.5],[-2.5,2.5],'--', color="red")
        plt.text(x=2.45, y=2.52, s="x=y",color="red")
        scatter =plt.scatter(df["RF nes"].values,df["Our nes"].values, c=df.y.astype('category').cat.codes, cmap='viridis')


        # add legend to the plot with names
        plt.legend(handles=scatter.legend_elements(alpha=0.6)[0], 
                labels=labels,
                title="Cell Type",loc="lower right",fontsize = 22)
        plt.savefig(f"./results/per_sample_plots/scatter_{f.name[:-5]}.png")