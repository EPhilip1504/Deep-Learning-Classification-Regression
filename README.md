# Project 2 - Neural Networks for Classification and Regression

This repository contains two deep learning projects implemented in PyTorch as a collaborative effort:
1.  **San Francisco Crime Classification:** A multi-class classification model predicting crime categories from location and time data.
2.  **Concrete Compressive Strength Regression:** A regression model predicting concrete strength based on its ingredients and age.

This project was developed as a collaborative effort:
* **Elijah Philip**
* **Mani Tofigh**

We worked together to perform data exploritary analysis, preprocessing, training, and evaluating the results of the models created.

## Table of Contents
- [San Francisco Crime Classification](#san-francisco-crime-classification)
  - [Dataset Description](#dataset-description)
  - [NN Architectures and Parameters](#nn-architectures-and-parameters)
  - [Results Analysis](#results-analysis)
  - [Learning Success Evaluation](#learning-success-evaluation)
- [Regression Problem: Concrete Compressive Strength](#regression-problem-concrete-compressive-strength)
  - [Problem and Dataset Descriptions](#problem-and-dataset-descriptions)
  - [Neural Network Architectures, Parameters, and Results](#neural-network-architectures-parameters-and-results-regression-task)
  - [Comments on Results and Learning Success](#comments-on-results-and-learning-success-regression-task)
---


## Classification Problem Section: San Francisco Crime Classification

### Dataset Description

Dataset contains 878,049 crime incidents from SFPD (2003-2015). Goal: predict crime category from location/time data.

features:
- X, Y (Longitude/Latitude)
- Hour (from timestamp)
- DayOfYear
- PdDistrict_enc (encoded district)

Target: 39 crime categories with significant imbalance (LARCENY/THEFT: 174,900 vs. others much lower).

Sample data:
```
data head:
                 Dates        Category                      Descript  \
0  2015-05-13 23:53:00        WARRANTS                WARRANT ARREST
1  2015-05-13 23:53:00  OTHER OFFENSES      TRAFFIC VIOLATION ARREST
2  2015-05-13 23:33:00  OTHER OFFENSES      TRAFFIC VIOLATION ARREST
3  2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM LOCKED AUTO
4  2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM LOCKED AUTO
```

Correlation analysis showed weak relationships between features and target:

![Correlation Heatmap](correlation_heatmap.png)


```
correlation with target:
Category_enc      1.000000
Hour              0.023524
Day               0.000805
DayOfWeek_enc     0.000388
DayOfYear         0.000075
Month             0.000008
WeekOfYear       -0.000137
Y                -0.000414
Year             -0.021803
X                -0.024401
PdDistrict_enc   -0.040674
```

### NN Architectures and Parameters

Implemented 4 model architectures to compare performance:

![Model Comparison Table](model_comparison.png)

```
Model Comparison:
          model hidden_dims optimizer  learning_rate  dropout  final_accuracy
0   small_model       64-32      adam         0.0010      0.2       24.443938
1  medium_model      128-64      adam         0.0010      0.3       24.469563
2   large_model     256-128      adam         0.0005      0.4       24.699618
3     sgd_model      128-64       sgd         0.0100      0.3       23.914356
```

Architecture details:
- Each model used ReLU activation and dropout for regularization
- Input layer: 5 features
- Output layer: 39 classes (crime categories)
- CrossEntropyLoss function for multi-class classification
- Models trained for 30 epochs

Training progression (large_model):
```
Epoch 0, Train loss: 2.6493, Train acc: 21.08%, Val loss: 2.5934, Val acc: 22.58%
Epoch 5, Train loss: 2.5687, Train acc: 23.31%, Val loss: 2.5430, Val acc: 23.81%
...
Epoch 30, Train loss: 2.5349, Train acc: 24.04%, Val loss: 2.5067, Val acc: 24.70%
```

Best model architecture:
```
Best model: large_model
CrimeClassifier(
  (layer1): Linear(in_features=5, out_features=256, bias=True)
  (dropout1): Dropout(p=0.4, inplace=False)
  (layer2): Linear(in_features=256, out_features=128, bias=True)
  (dropout2): Dropout(p=0.4, inplace=False)
  (layer3): Linear(in_features=128, out_features=39, bias=True)
)
```

### Results Analysis

Best model: large_model (256-128 hidden dims) with Adam optimizer achieved 24.70% validation accuracy.

Class-wise performance:

![Classification Report Confusion Matrix](confusion_matrix.png)

```
Classification Report (Top 15 Classes):
                        precision    recall  f1-score   support

         LARCENY/THEFT       0.28      0.82      0.42     35099
        OTHER OFFENSES       0.20      0.34      0.25     25026
          NON-CRIMINAL       0.19      0.02      0.03     18500
               ASSAULT       0.22      0.02      0.03     15364
         DRUG/NARCOTIC       0.29      0.38      0.33     10723
         VEHICLE THEFT       0.21      0.04      0.07     10671
             VANDALISM       0.00      0.00      0.00      9124
              WARRANTS       0.00      0.00      0.00      8514
              BURGLARY       0.00      0.00      0.00      7389
        SUSPICIOUS OCC       0.00      0.00      0.00      6252
        MISSING PERSON       0.44      0.10      0.16      5129
               ROBBERY       0.00      0.00      0.00      4592
                 FRAUD       0.00      0.00      0.00      3277
FORGERY/COUNTERFEITING       0.00      0.00      0.00      2092
       SECONDARY CODES       0.00      0.00      0.00      2063
```

observations:
- Model primarily predicts common crimes (LARCENY/THEFT has 0.82 recall)
- Many classes never predicted (9 of top 15 have zero recall)
- DRUG/NARCOTIC has best balanced performance (F1=0.33)
- Weighted avg F1-score: 0.17

### Learning Success Evaluation

Learning was partially successful:
1. Model improved from initial 21% to 24.7% accuracy (random guess: 1/39 = 2.56%)
2. Training/validation curves show proper convergence without overfitting
3. Several classes show decent prediction capability

However, limitations exist:

Class imbalance severely affected model performance

Limited feature correlation with target (max 0.04)

Many crime categories never predicted

Overall accuracy (24.7%) leaves room for improvement

Data preparation process was successful though:
Split data into 80% training and 20% validation (702,439 vs 175,610 samples)

Features properly normalized using StandardScaler

Tensors created with correct shapes and types

## Regression Problem

## Problem and Dataset Descriptions

### Problem 1: Regression - Predicting Concrete Compressive Strength

* **Objective:** To predict the quantitative value of Concrete Compressive Strength (measured in MPa) based on the concrete's ingredients and age. This is a supervised learning regression task.
* **Dataset:** The [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) was used.
    * **Original Data:** 1030 instances, 8 quantitative input features (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age) and 1 quantitative output variable (Concrete Compressive Strength). No missing values.
    * **Preprocessing:**
        * **Feature Removal:** 'Blast Furnace Slag', 'Fly Ash', 'Coarse Aggregate', and 'Fine Aggregate' were removed.
        * **Features Used:** The model was trained using 'Cement', 'Water', 'Superplasticizer', and 'Age' (4 input features).
        * **Data Splitting:** The data was split chronologically *after* feature removal: 80% for training, and 20% was initially set aside. This 20% was likely further divided to create distinct 10% validation and 10% test sets (`random_state=123` used for the initial split).
        * **Scaling:** `StandardScaler` was applied *after* splitting the data (fit on training data only, then transformed train, validation, and test sets).

## Neural Network Architectures, Parameters, and Results (Regression Task)

* **Architectures Tried:** Three feedforward neural network architectures were implemented using PyTorch:
    * `ModelV1`: Input(4) -> Linear(30) -> ReLU -> Linear(10) -> ReLU -> Linear(1)
    * `ModelV2_Deeper`: Input(4) -> Linear(30) -> ReLU -> Linear(20) -> ReLU -> Linear(10) -> ReLU -> Linear(1)
    * `ModelV3_Wider`: Input(4) -> Linear(50) -> ReLU -> Linear(1)
* **Hyperparameters Tested:**
    * **Optimizers:** Adam, SGD
    * **Learning Rates:** 0.01, 0.005, 0.02
* **Best Model and Results:** The best performance on the validation set was achieved using:
    * **Architecture:** `ModelV2_Deeper`
    * **Optimizer:** Adam
    * **Learning Rate:** 0.01
    * **Test Set Performance:**
        * **R²:** 0.811
        * **RMSE:** 7.20 MPa
        * **MAE:** 5.48 MPa
        * *(Average Validation Loss during training: 0.0464)*

## Comments on Results and Learning Success (Regression Task)

* **Learning Success:** Yes, learning appears to have been **successful** for the regression task.
* **Reasoning:**
    * An **R² value of 0.811** on the unseen test data indicates that the model can explain approximately 81% of the variance in Concrete Compressive Strength using the four selected input features (Cement, Water, Superplasticizer, Age). This suggests the model captured significant underlying patterns in the data.
    * The **RMSE (7.20 MPa)** and **MAE (5.48 MPa)** quantify the typical prediction error magnitude in the original units (MPa). While an R² of 0.81 is statistically good, the practical acceptability of an average error around 5.5-7.2 MPa depends entirely on the tolerance for error in the specific civil engineering application this model would be used for.



## Classification Problem Section: San Francisco Crime Classification

### Dataset Description

Dataset contains 878,049 crime incidents from SFPD (2003-2015). Goal: predict crime category from location/time data.

features:
- X, Y (Longitude/Latitude)
- Hour (from timestamp)
- DayOfYear
- PdDistrict_enc (encoded district)

Target: 39 crime categories with significant imbalance (LARCENY/THEFT: 174,900 vs. others much lower).

Sample data:
```
data head:
                 Dates        Category                      Descript  \
0  2015-05-13 23:53:00        WARRANTS                WARRANT ARREST
1  2015-05-13 23:53:00  OTHER OFFENSES      TRAFFIC VIOLATION ARREST
2  2015-05-13 23:33:00  OTHER OFFENSES      TRAFFIC VIOLATION ARREST
3  2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM LOCKED AUTO
4  2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM LOCKED AUTO
```

Correlation analysis showed weak relationships between features and target:

![Correlation Heatmap](correlation_heatmap.png)


```
correlation with target:
Category_enc      1.000000
Hour              0.023524
Day               0.000805
DayOfWeek_enc     0.000388
DayOfYear         0.000075
Month             0.000008
WeekOfYear       -0.000137
Y                -0.000414
Year             -0.021803
X                -0.024401
PdDistrict_enc   -0.040674
```

### NN Architectures and Parameters

Implemented 4 model architectures to compare performance:

![Model Comparison Table](model_comparison.png)

```
Model Comparison:
          model hidden_dims optimizer  learning_rate  dropout  final_accuracy
0   small_model       64-32      adam         0.0010      0.2       24.443938
1  medium_model      128-64      adam         0.0010      0.3       24.469563
2   large_model     256-128      adam         0.0005      0.4       24.699618
3     sgd_model      128-64       sgd         0.0100      0.3       23.914356
```

Architecture details:
- Each model used ReLU activation and dropout for regularization
- Input layer: 5 features
- Output layer: 39 classes (crime categories)
- CrossEntropyLoss function for multi-class classification
- Models trained for 30 epochs

Training progression (large_model):
```
Epoch 0, Train loss: 2.6493, Train acc: 21.08%, Val loss: 2.5934, Val acc: 22.58%
Epoch 5, Train loss: 2.5687, Train acc: 23.31%, Val loss: 2.5430, Val acc: 23.81%
...
Epoch 30, Train loss: 2.5349, Train acc: 24.04%, Val loss: 2.5067, Val acc: 24.70%
```

Best model architecture:
```
Best model: large_model
CrimeClassifier(
  (layer1): Linear(in_features=5, out_features=256, bias=True)
  (dropout1): Dropout(p=0.4, inplace=False)
  (layer2): Linear(in_features=256, out_features=128, bias=True)
  (dropout2): Dropout(p=0.4, inplace=False)
  (layer3): Linear(in_features=128, out_features=39, bias=True)
)
```

### Results Analysis

Best model: large_model (256-128 hidden dims) with Adam optimizer achieved 24.70% validation accuracy.

Class-wise performance:

![Classification Report Confusion Matrix](confusion_matrix.png)

```
Classification Report (Top 15 Classes):
                        precision    recall  f1-score   support

         LARCENY/THEFT       0.28      0.82      0.42     35099
        OTHER OFFENSES       0.20      0.34      0.25     25026
          NON-CRIMINAL       0.19      0.02      0.03     18500
               ASSAULT       0.22      0.02      0.03     15364
         DRUG/NARCOTIC       0.29      0.38      0.33     10723
         VEHICLE THEFT       0.21      0.04      0.07     10671
             VANDALISM       0.00      0.00      0.00      9124
              WARRANTS       0.00      0.00      0.00      8514
              BURGLARY       0.00      0.00      0.00      7389
        SUSPICIOUS OCC       0.00      0.00      0.00      6252
        MISSING PERSON       0.44      0.10      0.16      5129
               ROBBERY       0.00      0.00      0.00      4592
                 FRAUD       0.00      0.00      0.00      3277
FORGERY/COUNTERFEITING       0.00      0.00      0.00      2092
       SECONDARY CODES       0.00      0.00      0.00      2063
```

observations:
- Model primarily predicts common crimes (LARCENY/THEFT has 0.82 recall)
- Many classes never predicted (9 of top 15 have zero recall)
- DRUG/NARCOTIC has best balanced performance (F1=0.33)
- Weighted avg F1-score: 0.17

### Learning Success Evaluation

Learning was partially successful:
1. Model improved from initial 21% to 24.7% accuracy (random guess: 1/39 = 2.56%)
2. Training/validation curves show proper convergence without overfitting
3. Several classes show decent prediction capability

However, limitations exist:

Class imbalance severely affected model performance

Limited feature correlation with target (max 0.04)

Many crime categories never predicted

Overall accuracy (24.7%) leaves room for improvement

Data preparation process was successful though:
Split data into 80% training and 20% validation (702,439 vs 175,610 samples)

Features properly normalized using StandardScaler

Tensors created with correct shapes and types

## Regression Problem

## Problem and Dataset Descriptions

### Problem 1: Regression - Predicting Concrete Compressive Strength

* **Objective:** To predict the quantitative value of Concrete Compressive Strength (measured in MPa) based on the concrete's ingredients and age. This is a supervised learning regression task.
* **Dataset:** The [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) was used.
    * **Original Data:** 1030 instances, 8 quantitative input features (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age) and 1 quantitative output variable (Concrete Compressive Strength). No missing values.
    * **Preprocessing:**
        * **Feature Removal:** 'Blast Furnace Slag', 'Fly Ash', 'Coarse Aggregate', and 'Fine Aggregate' were removed.
        * **Features Used:** The model was trained using 'Cement', 'Water', 'Superplasticizer', and 'Age' (4 input features).
        * **Data Splitting:** The data was split chronologically *after* feature removal: 80% for training, and 20% was initially set aside. This 20% was likely further divided to create distinct 10% validation and 10% test sets (`random_state=123` used for the initial split).
        * **Scaling:** `StandardScaler` was applied *after* splitting the data (fit on training data only, then transformed train, validation, and test sets).

## Neural Network Architectures, Parameters, and Results (Regression Task)

* **Architectures Tried:** Three feedforward neural network architectures were implemented using PyTorch:
    * `ModelV1`: Input(4) -> Linear(30) -> ReLU -> Linear(10) -> ReLU -> Linear(1)
    * `ModelV2_Deeper`: Input(4) -> Linear(30) -> ReLU -> Linear(20) -> ReLU -> Linear(10) -> ReLU -> Linear(1)
    * `ModelV3_Wider`: Input(4) -> Linear(50) -> ReLU -> Linear(1)
* **Hyperparameters Tested:**
    * **Optimizers:** Adam, SGD
    * **Learning Rates:** 0.01, 0.005, 0.02
* **Best Model and Results:** The best performance on the validation set was achieved using:
    * **Architecture:** `ModelV2_Deeper`
    * **Optimizer:** Adam
    * **Learning Rate:** 0.01
    * **Test Set Performance:**
        * **R²:** 0.811
        * **RMSE:** 7.20 MPa
        * **MAE:** 5.48 MPa
        * *(Average Validation Loss during training: 0.0464)*

## Comments on Results and Learning Success (Regression Task)

* **Learning Success:** Yes, learning has been successful for the regression task.
* **Reasoning:**
    * An # Project 2 - Neural Networks for Classification and Regression

This repository contains two deep learning projects implemented in PyTorch:
1.  **San Francisco Crime Classification:** A multi-class classification model predicting crime categories from location and time data.
2.  **Concrete Compressive Strength Regression:** A regression model predicting concrete strength based on its ingredients and age.

## Table of Contents
- [Project Overview](#project-overview)
- [San Francisco Crime Classification](#san-francisco-crime-classification)
  - [Dataset Description](#dataset-description)
  - [NN Architectures and Parameters](#nn-architectures-and-parameters)
  - [Results Analysis](#results-analysis)
  - [Learning Success Evaluation](#learning-success-evaluation)
- [Regression Problem: Concrete Compressive Strength](#regression-problem-concrete-compressive-strength)
  - [Problem and Dataset Descriptions](#problem-and-dataset-descriptions)
  - [Neural Network Architectures, Parameters, and Results](#neural-network-architectures-parameters-and-results-regression-task)
  - [Comments on Results and Learning Success](#comments-on-results-and-learning-success-regression-task)
- [How to Run the Notebooks](#how-to-run-the-notebooks)
- [Requirements](#requirements)
- [Contributors / Acknowledgements](#contributors--acknowledgements)

---


## Classification Problem Section: San Francisco Crime Classification

### Dataset Description

Dataset contains 878,049 crime incidents from SFPD (2003-2015). Goal: predict crime category from location/time data.

features:
- X, Y (Longitude/Latitude)
- Hour (from timestamp)
- DayOfYear
- PdDistrict_enc (encoded district)

Target: 39 crime categories with significant imbalance (LARCENY/THEFT: 174,900 vs. others much lower).

Sample data:
```
data head:
                 Dates        Category                      Descript  \
0  2015-05-13 23:53:00        WARRANTS                WARRANT ARREST
1  2015-05-13 23:53:00  OTHER OFFENSES      TRAFFIC VIOLATION ARREST
2  2015-05-13 23:33:00  OTHER OFFENSES      TRAFFIC VIOLATION ARREST
3  2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM LOCKED AUTO
4  2015-05-13 23:30:00   LARCENY/THEFT  GRAND THEFT FROM LOCKED AUTO
```

Correlation analysis showed weak relationships between features and target:

![Correlation Heatmap](correlation_heatmap.png)


```
correlation with target:
Category_enc      1.000000
Hour              0.023524
Day               0.000805
DayOfWeek_enc     0.000388
DayOfYear         0.000075
Month             0.000008
WeekOfYear       -0.000137
Y                -0.000414
Year             -0.021803
X                -0.024401
PdDistrict_enc   -0.040674
```

### NN Architectures and Parameters

Implemented 4 model architectures to compare performance:

![Model Comparison Table](model_comparison.png)

```
Model Comparison:
          model hidden_dims optimizer  learning_rate  dropout  final_accuracy
0   small_model       64-32      adam         0.0010      0.2       24.443938
1  medium_model      128-64      adam         0.0010      0.3       24.469563
2   large_model     256-128      adam         0.0005      0.4       24.699618
3     sgd_model      128-64       sgd         0.0100      0.3       23.914356
```

Architecture details:
- Each model used ReLU activation and dropout for regularization
- Input layer: 5 features
- Output layer: 39 classes (crime categories)
- CrossEntropyLoss function for multi-class classification
- Models trained for 30 epochs

Training progression (large_model):
```
Epoch 0, Train loss: 2.6493, Train acc: 21.08%, Val loss: 2.5934, Val acc: 22.58%
Epoch 5, Train loss: 2.5687, Train acc: 23.31%, Val loss: 2.5430, Val acc: 23.81%
...
Epoch 30, Train loss: 2.5349, Train acc: 24.04%, Val loss: 2.5067, Val acc: 24.70%
```

Best model architecture:
```
Best model: large_model
CrimeClassifier(
  (layer1): Linear(in_features=5, out_features=256, bias=True)
  (dropout1): Dropout(p=0.4, inplace=False)
  (layer2): Linear(in_features=256, out_features=128, bias=True)
  (dropout2): Dropout(p=0.4, inplace=False)
  (layer3): Linear(in_features=128, out_features=39, bias=True)
)
```

### Results Analysis

Best model: large_model (256-128 hidden dims) with Adam optimizer achieved 24.70% validation accuracy.

Class-wise performance:

![Classification Report Confusion Matrix](confusion_matrix.png)

```
Classification Report (Top 15 Classes):
                        precision    recall  f1-score   support

         LARCENY/THEFT       0.28      0.82      0.42     35099
        OTHER OFFENSES       0.20      0.34      0.25     25026
          NON-CRIMINAL       0.19      0.02      0.03     18500
               ASSAULT       0.22      0.02      0.03     15364
         DRUG/NARCOTIC       0.29      0.38      0.33     10723
         VEHICLE THEFT       0.21      0.04      0.07     10671
             VANDALISM       0.00      0.00      0.00      9124
              WARRANTS       0.00      0.00      0.00      8514
              BURGLARY       0.00      0.00      0.00      7389
        SUSPICIOUS OCC       0.00      0.00      0.00      6252
        MISSING PERSON       0.44      0.10      0.16      5129
               ROBBERY       0.00      0.00      0.00      4592
                 FRAUD       0.00      0.00      0.00      3277
FORGERY/COUNTERFEITING       0.00      0.00      0.00      2092
       SECONDARY CODES       0.00      0.00      0.00      2063
```

observations:
- Model primarily predicts common crimes (LARCENY/THEFT has 0.82 recall)
- Many classes never predicted (9 of top 15 have zero recall)
- DRUG/NARCOTIC has best balanced performance (F1=0.33)
- Weighted avg F1-score: 0.17

### Learning Success Evaluation

Learning was partially successful:
1. Model improved from initial 21% to 24.7% accuracy (random guess: 1/39 = 2.56%)
2. Training/validation curves show proper convergence without overfitting
3. Several classes show decent prediction capability

However, limitations exist:

Class imbalance severely affected model performance

Limited feature correlation with target (max 0.04)

Many crime categories never predicted

Overall accuracy (24.7%) leaves room for improvement

Data preparation process was successful though:
Split data into 80% training and 20% validation (702,439 vs 175,610 samples)

Features properly normalized using StandardScaler

Tensors created with correct shapes and types

## Regression Problem

## Problem and Dataset Descriptions

### Problem 1: Regression - Predicting Concrete Compressive Strength

* **Objective:** To predict the quantitative value of Concrete Compressive Strength (measured in MPa) based on the concrete's ingredients and age. This is a supervised learning regression task.
* **Dataset:** The [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) was used.
    * **Original Data:** 1030 instances, 8 quantitative input features (Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age) and 1 quantitative output variable (Concrete Compressive Strength). No missing values.
    * **Preprocessing:**
        * **Feature Removal:** 'Blast Furnace Slag', 'Fly Ash', 'Coarse Aggregate', and 'Fine Aggregate' were removed.
        * **Features Used:** The model was trained using 'Cement', 'Water', 'Superplasticizer', and 'Age' (4 input features).
        * **Data Splitting:** The data was split chronologically *after* feature removal: 80% for training, and 20% was initially set aside. This 20% was likely further divided to create distinct 10% validation and 10% test sets (`random_state=123` used for the initial split).
        * **Scaling:** `StandardScaler` was applied *after* splitting the data (fit on training data only, then transformed train, validation, and test sets).

## Neural Network Architectures, Parameters, and Results (Regression Task)

* **Architectures Tried:** Three feedforward neural network architectures were implemented using PyTorch:
    * `ModelV1`: Input(4) -> Linear(30) -> ReLU -> Linear(10) -> ReLU -> Linear(1)
    * `ModelV2_Deeper`: Input(4) -> Linear(30) -> ReLU -> Linear(20) -> ReLU -> Linear(10) -> ReLU -> Linear(1)
    * `ModelV3_Wider`: Input(4) -> Linear(50) -> ReLU -> Linear(1)
* **Hyperparameters Tested:**
    * **Optimizers:** Adam, SGD
    * **Learning Rates:** 0.01, 0.005, 0.02
* **Best Model and Results:** The best performance on the validation set was achieved using:
    * **Architecture:** `ModelV2_Deeper`
    * **Optimizer:** Adam
    * **Learning Rate:** 0.01
    * **Test Set Performance:**
        * **R²:** 0.811
        * **RMSE:** 7.20 MPa
        * **MAE:** 5.48 MPa
        * *(Average Validation Loss during training: 0.0464)*

## Comments on Results and Learning Success (Regression Task)

* **Learning Success:** Yes, learning appears to have been **successful** for the regression task.
* **Reasoning:**
    * An **R² value of 0.811** on the unseen test data indicates that the model can explain approximately 81% of the variance in Concrete Compressive Strength using the four selected input features (Cement, Water, Superplasticizer, Age). This suggests the model captured significant underlying patterns in the data.
    * The **RMSE (7.20 MPa)** and **MAE (5.48 MPa)** quantify the typical prediction error magnitude in the original units (MPa). While an R² of 0.81 is statistically good, the practical acceptability of an average error around 5.5-7.2 MPa depends entirely on the tolerance for error in the specific civil engineering application this model would be used for.
 on the unseen test data indicates that the model can explain approximately 81% of the variance in Concrete Compressive Strength using the four selected input features (Cement, Water, Superplasticizer, Age). This suggests the model captured significant underlying patterns in the data.
    * The **RMSE (7.20 MPa)** and **MAE (5.48 MPa)** quantify the typical prediction error magnitude in the original units (MPa). While an R² of 0.81 is statistically good, the practical acceptability of an average error around 5.5-7.2 MPa depends entirely on the tolerance for error in the specific civil engineering application this model would be used for.
