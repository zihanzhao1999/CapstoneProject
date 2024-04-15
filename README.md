# CapstoneProject - Diseases Prediction

## Introduction:

In the field of healthcare, the integration of machine learning technologies has played an important role, particularly in the realm of disease diagnosis and management. A good machine learniong model can shorten the time spent on diagnosis thus better support the treatment.
Our group aim to find better models with higher prediction accuracy to assist healthcare providers with patients’ diagnoses.
For now, we have implemented different models to be trained on the ground truth pathology and differential diagnosis. Please refer to `predict_pathology` and `predict_differential_diagnosis` folder.

## Getting Started

### Dependencies

* Python >= 3.8
* tqdm == 4.65.0
* torchinfo == 1.8.0
* autogluon == 1.0.0
* Pandas; Numpy; Matplotlib; Scipy; Sklearn; Pytorch; Tensorflow; AutoGluon


## Requirments:

![requirements](https://img.shields.io/badge/Python->3.8.0-3480eb.svg?longCache=true&style=flat&logo=python)
```
pip install -r requirements.txt
```
Note: By default, CPU version of Pytorch is installed. If you like to use GPU version, please refer to https://pytorch.org/.

Or try following command after installing requirements.txt:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data source: 
[DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://arxiv.org/pdf/2205.09148.pdf) 
and [its data](https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374)

Please download and place the data in `data` folder

## Authors

Ting Husan Chung //
Jiaying Zheng //
Zihan Zhao

## Citation:
```
Fansi Tchango, A., Goel, R., Wen, Z., Martel, J., & Ghosn, J. (2022).
Ddxplus: A new dataset for automatic medical diagnosis.
Advances in Neural Information Processing Systems, 35, 31306-31318.
```
