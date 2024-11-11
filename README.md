
# CapstoneProject - Diseases Prediction 
![AD_icon_100](https://github.com/zihanzhao1999/CapstoneProject/assets/111836220/cc73e827-c1b6-47be-b790-8bf827ffe372)

## Introduction & Motivation:
In the field of healthcare, the integration of machine learning technologies has played an important role, particularly in the realm of disease diagnosis and management. A good machine learning model can shorten the time spent on diagnosis thus better support the treatment.
Our group aim to find better models with higher prediction accuracy to assist healthcare providers with patients’ diagnoses.

For now, we have implemented different models to be trained on the ground truth pathology and differential diagnosis. Please refer to `predict_pathology` and `predict_differential_diagnosis` folder.

Also, based on the best model(LSTM) and a simple inquiry system we invented, we implemented the Auto Diagnosis System.
Moreover, we have implemented a Streamlit app for the AD System. Please refer to `AD_system`. 


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

### Dependencies
* Python >= 3.8
* tqdm == 4.65.0
* torchinfo == 1.8.0
* autogluon == 1.0.0
* zstandard == 0.22.0
* streamlit == 1.33.0
* Pandas; Numpy; Matplotlib; Scipy; Sklearn; Pytorch; Tensorflow; AutoGluon

## Run AD System as Streamlit app
```
cd AD_system
streamlit run streamlit_app.py
```
![image](https://github.com/zihanzhao1999/CapstoneProject/assets/111836220/a3844094-c03f-4edf-82fa-c428457a4b22)

## Data source: 
[DDXPlus: A New Dataset For Automatic Medical Diagnosis](https://arxiv.org/pdf/2205.09148.pdf) 
and [its data](https://figshare.com/ndownloader/articles/20043374/versions/14)

Please download and place the data in `data` folder

## Authors

Ting Husan Chung

Jiaying Zheng 

Zihan Zhao

## Citation:
```
Fansi Tchango, A., Goel, R., Wen, Z., Martel, J., & Ghosn, J. (2022).
Ddxplus: A new dataset for automatic medical diagnosis.
Advances in Neural Information Processing Systems, 35, 31306-31318.
```
```
Alam, M. M., Raff, E., Oates, T., & Matuszek, C. (2023). Ddxt: Deep generative
transformer models for differential diagnosis. Deep Generative Models
for Health Workshop NeurIPS 2023
```
