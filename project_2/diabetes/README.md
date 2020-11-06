<h1>Predict readmission within 30 days of discharge</h1>

This directory contains our work on the fist task of the second project: predicting readmission  within 30 days of discharge from a [Diabetes 130-US hospitals for years 1999-2008 UCI data set](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

The directory consists of the the following notebooks:

* `data_exploration.ipynb` contains plots for the data exploration, including wordclouds and lda topic distribution
* `models.ipynb` contains the first set of fitted models and develops a data preprocessing pipeline
* `hyper_parameter_tuning.ipynb` builds upon `models.ipynb` and extends it to create a flexible framework for hyperparameter tuning
