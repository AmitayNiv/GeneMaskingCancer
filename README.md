
## <div align="center">GeneMasking</div>

GeneMasking is a new method for finding the unique set of distinguishing genes for each input single-cell gene expression.
The method trains, jointly, two neural networks: a classifier $f$ and a feature selection network $g$, such that the classification of the entire sample is identical to the classification obtained based on the genes selected by $g$.

<p align="center">
  <img src="/architecture.png">
</p>


## <div align="center">Data Preprocess</div>

We used 16 published CITE-seq datasets:

Dataset | #cells | #classes
| :---: | :---: | :---:
10X pbmc 1k | 713 | 10
Hao 2020  | 161764  | 15
Su 2020 |  559583  | 15
Fournie 2020 |  5559 |  7
Granja 2019 bmmc  | 12602 |  10
10X malt 10k  | 8412  | 7
10X pbmc 5k nextgem  | 5527 |  13
Granja 2019 pbmc  | 14804  | 10
Stoeckius 2017 cbmc |  8617 | 8
Kotliarov 2020 |  58654  | 15
10X pbmc 10k  | 8201 |  10
Butler 2019  | 33454  | 15
Arunachalam 2020 |  63469  | 15
10X pbmc 5k v3  | 5247 |  13
Witkowski 2020  | 42621  | 13
Wang 2020  | 1372  | 5


All data need to be downlaod from the following link: 
[Datasets](https://fh-pi-gottardo-r-eco-public.s3.amazonaws.com/SingleCellDatasets/SingleCellDatasets.html "Named link title").
The datasets should be saved into ./data/singleCell/ folder.

## <div align="center">Run</div>
To run the project use the run.py which is the main script. The run.py file contains several arguments that control the different tasks and runs. The desired task should be placed in Arguments.task. Every other parameter is set to default and may be modified as required.

Task | Description
| :---: | :---: |
Train | Run training and evaluation on Arguments.working_models
Mask Creation | Cerating masks and save masks to files
Masks Visualizatin | Cerating mask and UMAP projection (input and mask) of H
GSEA | Run GSEA analisys for all datasets, for all models (G,H,F2,XGB,RF) and saving results to files
Heatmaps | Create heatmaps for specific genes per patients - Working only for datasets with patient data
GSEA per Sample Compariosn | Run per sample GSEA analisys based on our H, ELI5 for RF or SHAP for XGB, saving data per dataset and overall comparison
GSEA per Sample | Run per sample GSEA analisys based on our methods for ablation,saving data per dataset and overall results





