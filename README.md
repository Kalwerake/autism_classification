# Machine Learning based Autism Spectrum Diagnosis

This project is based on the ABIDE I dataset, avaible in a public AWS S3 Bucket.
Download scripts are included here.

Abbreviations:
FC: functional connectivity
DFC: dynamic Functional connectivity
GCI: granger causlity index
lsGCI: large scale granger causlaity index


## 1. Project Structure

All raw and extracted data is named to be easily identified via a unique identifier `FILE_ID` followed by suffix and file extension FILE_ID is given in phenotype description csv found in  `/phenotype_files/pheno_clean.csv`
 
`{FILE_ID}_{suffix}.{extension}`
### MscProject
* atlas
  *  \#  all atlas label files in format {atlas}_labels.csv
* bash_scripts
  * download.sh
* custom_dataset
  * \_\_init__.py \# custom package using pytorch.utils.data.DataSet
* dfc_cc200
  * {FILE_ID}_dfc.npy  \# all extracted dFC arrays
* EDA_notebooks
  * dfc_exploration.ipynb
  * phenotype_EDA.ipynb
* fc_cc200
  * {FILE_ID}_fc.npy  \# all extracted FC arrays
* feature_extraction \# custom package of feature extraction algorthms
  * \_\_init__.py
  * dfc.py \# dfc extraction module
  * fc.npy \# fc ectraction module
  * granger.py \# granger causality module
* func_preproc
  * {FILE_ID}_func_preproc.nii.gz #all fMRI data
* gci_ls_cc200
  * {FILE_ID}_gci.npy  \# all extracted lsGCI arrays
* gci_cc200
  * {FILE_ID}_gci.npy \# all extracted GCI arrays
* missing_lists # directory, lists of all subjects with missing values
  * {atlas name}_missing.txt
* model_evaluation # directory storing all model metrics
* nn_functions # custom package for nn training
  * \_\_init__.py
  * custom_dataset.py
  * functions.py
  * models.py
* phenotype_files  # directory for phenotype description files
  * pheno_clean.csv
  * pheno_nn.csv
  * Phenotypic_V1_0b_preprocessed.csv
* rois_cc200 # timeseries data from cc200 atlas
  * {file_id}_rois_cc200.1D
* vectorised # folder for all vectorised data
  * {atlas}_filtered.csv
  * {atlas}_vectorised.csv.gz
* crrn_final.ipynb
* feature_exploration.ipynb
* final_boruta_nn.ipynb
* lsGCI.ipynb
* ml.ipynb
* model_performance.ipynb
* svc_cc400.ipynb
* twin_surrogates.ipynb
* README.md
* requirements.txt
* aug_dfc.py
* boruta_ml.py
* dfc_maker.py
* download_abide.py
* granger_calculate.py
* surrogate_model.py
* twin_surrogates.py
## 2. Getting Started

This project was built on Python 3.10, `requirments.txt` lists all required packages.
First step is to install all packages pip environment is recommended, since using conda threw out some errors.

`! pip install requirements.txt`

## 3. Downloading data

The data download scripts are in two parts, The first download includes:

1. phenotype description file which provide a unique identifier for each subject and diagnostic class. Used for model training and validation as an annotation file
2. atlas label file which provides the ROI number for each atlas and corrospind anatomical names
3. Raw fMRI data (optional)
4. ROI time-series data based on CC200 atlas (optional)

For first download run `download_abide.py` in terminal, script will create subdirectories and download data to these 

To download only phenotype descriptor and atlas labels run in terminal:

`python3 download_phenotype_files.py`

To include fMRI data, its optional but is needed if `EDA_notebooks/phenotype_EDA.ipynb` is to make sense: 

`python3 download_phenotype_files.py --func_preproc`

For CC200 timeseries data:

`python3 download_phenotype_files.py --rois_cc200`

The second batch does not need to be downloaded right away,it includes roi time series data. You have the option of downloading all the raw time-series data and saving to local directory, 
or extracting fc, dfc, gci matrices and saving only them. 
The option to vectorise these matrices and only save them is there as well.
the script is `download_process.py` and the cleaned phenotype descriptor file given in `phenotype_files/pheno_clean.csv` is required. The phenotype description file from initial download needs cleaning as described in [Cleaning phenotype description file](#cleaning-phenotype-description-file) to produce `pheno_clean.csv`


To download raw roi time series data, e.g cc400 atlas

`python3 download_process.py --atlas "rois_cc400" --df "phenotype_files/pheno_clean.csv" --store "raw"`

To store only square fc matrices

`python3 download_process.py --atlas "rois_cc400" --df "phenotype_files/pheno_clean.csv" --store "fc"`

To store only vectorised fc data

`python3 download_process.py --atlas "rois_cc400" --df "phenotype_files/pheno_clean.csv" --store "vectorised"`

NB: when saving fc matrices or vectorised fc matrices, any subjects with missing values
are identified and excluded from final output, a list of FILE_IDs with missing values are 
written to `missing_lists`

It's best to use provided bash script in `bash_scripts\download.sh` simply change name of parent directory


## 4. Cleaning phenotype description file

There are severel issues with the downloaded phenotype description file `phenotype_files\Phenotypic_V1_0b_preprocessed.csv`. 
These issues and fixes are detailed in `EDA_notebooks/phenotype_EDA.ipynb`, this notebook also displays class distributions. 
But the cleaned file `phenotype_files/pheno_clean.csv` is provided.

## 5. Exploring feature extraction methods

Exploration of what time series data looks like, and how to calculate FC and DFC 
matrices were carried out using notebook `EDA_notebooks/dfc_exploration.ipynb`.

## 6. DFC feature extraction

The first feature extraction method trialled was DFC, carried out using `dfc_maker.py`.
The DFC matrices will be calculated and stored in subdirectory `dfc_cc200/`.
This required the use of custom package `feature_extraction.dfc.DFC`, and a phenotype descriptor file.
When running for the first time I used the phenotype descriptor `phenotype_files/pheno_clean.csv`, 
this caused errors due to flatline time series in certain samples, this issue is detailed in `EDA_notebooks/dfc_exploration.ipynb`
All samples with flat lines were discovered and unique identifiers removed from original phenotype descriptor file to make new file
`phenotype_files/pheno_nn.csv` this procedure is detailed in `EDA_notebooks/phenotype_EDA.ipynb`. `pheno_nn.csv` is provided, and should be used to dfc calculation.

run on terminal:

`python3 dfc_maker.py --df "phenotype_files/pheno_nn.csv" --data rois_cc200 --extension "_rois_cc200.1D" --save "dfc_cc200" `

## 7. Fit DFC to CNN-LSTM

To fit the dfc to CNN-LSTM (CRNN) model see `crnn_final.ipynb`, this notebook was used on Google Colab.
And DFC data in `dfc_cc200` was uploaded onto my Google Drive to fit the model, and saves cross validation scores to a google drive folder. 
Since the model is not very reliable I did not bother saving the model weights and biases.

## 8. Data Augmentation

Data augmentation method was trialled via the twin surrogate method.
Exploring this method was carried out via `twin_surrogates.ipynb`. 
To generate surrogate data Macleod HPC cluster was used, run script `twin_surrogates.py`:

`python3 twin_surrogates.py --df "phenotype_files/pheno_nn.csv" --data rois_cc200 --save "rois_aug" `

To generate DFC matrices, data saved to `dfc_aug`:

`python3 aug_dfc.py --df "phenotype_files/pheno_nn.csv" --data rois_aug --save "dfc_aug" --window 70 `

To fit CRNN model use `surrogate_model.py` requires package `nn_functions`:

E.g train model with batch size of 116, learning rate of 0.0001, epochs 100, with early stop with patience of 10 and min delta of 2.9
script will make directories `network_params` and `model_evaluation` if they do not already exist

`python3 surrogate_model.py --df "phenotype_files/pheno_nn.csv" --data "aug_dfc" --batch 116 --lr 0.001 --epochs 100 --workers 1 --model_save --model_name "aug_crnn" --early_stop --patience 10 --delta 2.9`
 
Model performance metrics will be stored in `model_evaluation/aug_crnn.pickle` and model parameters will be saved in `network_params/aug_crnn.pth`

NB: this network takes 5 days to train and accuracy is at 50%.

## 8. Feature Selection and Machine Learning models (FC)

If the second batch of downloads in [Section 3 Downloading data](#3-downloading-data) was done as 
shown for all atlas-based datasets this section only takes one step:
`boruta_ml.py` can:
1. apply vectorised data to boruta feature selection algorithm 
2. fit the filtered data to machine learning models, 
3. save all evaluation metrics into `model_evaluation/{atlas}` script can create subdirectory tree. 

To apply for all atlases run bash script: `bash_scripts/run_boruta.sh`

individual example for cc200 atlas-based vectorised fc dataset:

`python3 boruta_ml.py --atlas "cc200" --df "vectorised/cc200_vectorised.csv.gz" --no-filtered`

## 9. Granger Causality
Granger causality was prototyped in `lsGCI.ipynb`, parameters were determined.

To calculate granger causality metrics for rois_cc200 data use `granger_calculate.py`
requires `feature_extraction.granger.gci` and from `feature_extraction.granger.large_scale_gci`

To calculate large scale granger causality, and store square matrices:

`python3 granger_calculate.py --df "phenotype_files/pheno_nn.csv" --data "rois_cc200" \
--save "gci_ls_cc200" --extension "_rois_cc200.1D" --large --no-vectorise`

To calculate large scale granger causality, and store vectorised data:

`python3 granger_calculate.py --df "phenotype_files/pheno_nn.csv" --data "rois_cc200" \
--save "gci_ls_cc200" --extension "_rois_cc200.1D" --large --vectorise`

Boruta algorithm and models were fitted on the gci data in `ml.ipynb` since the vectorised fc data provided
better results, no scripts were written for automating extraction,feature selection and models
gci data

## 10. CC400 fcn and ensemble model

The cc400 fully connected neural (FCN) network and the ensemble model was prototyped using
`final_boruta_nn.ipynb`. This notebook was built on google colab. But it can be run on local
simply pick `df = pd.read_csv('vectorised/fc_cc400_filtered.csv')` in second cell.
 The FCN is light weight and does not require a gpu.
The evaluation metrics will be saved in `model_evaluation/ensemble_metrics.csv`

## 11. Visualisation of model performances

Model performances were visualised using `model_performance.ipynb`. 
## 12. Hyperparameter turning and Feature Importance

Hyper-parameter tuning and feature importance analysis was tried out using `svc_cc400.ipynb`
Here the results of the feature importance analysis are saved in working directory as `cc400_highest_features.csv`
Adjacency tables were also made for control and asd subjects and saved in working directory as:
`'cc400_control_adj.csv'` and `'cc400_asd_adj.csv'`

The results of feature importance analysis, 
finding locations  and anatomical names of important rois was carried out on `feature_exploration.ipynb`




