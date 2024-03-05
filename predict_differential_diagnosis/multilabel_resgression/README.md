## Multilabel Regression Model

This model uses NN with FastAI backend by Autogluon to train each label(pathlogy) based on the values in ground true differential diagnosis (regression). Each label has its own network and takes the previous outputs of networks as input. Thus, the prediction of a specific label can only be obtained if previous predictions have made. 

Source code for the multilabel predictor can be found [here](https://auto.gluon.ai/1.0.0/tutorials/tabular/advanced/tabular-multilabel.html).
