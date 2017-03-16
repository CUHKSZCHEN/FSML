# FSML

A machine learning project written in F#

## Introduction

### Version
0.1.2

### Algorithms implemented:
  - Linear Regression
    * no penalty
    * Lasso (L1 penalty) via coordinate descent
    * Ridge (L2 penalty)
  - Logistic Regression
    * no penalty
    * Lasso (L1 penalty) via coordinate descent
    * Ridge (L2 penalty) via iterated reweighted least square
  - SVM via Sequential Minimal Optimization (SMO)
    * linear kernel
    * rbf(Gaussian) kernel 
  - Gradient Boosting Machine (GBM)
    * Gaussian response, i.e., least square loss function
    * Binomial response, i.e., logloss loss function
    * cross validation fit is not working well now


### Algorithms todo list:
  - cox ph model
  - subdistribution model

## Examples


### 1. Linear regression

We use the data /data/continuous.txt, which is stored in the libsvm format. Please note that no imputation is implemented at this time so missing values in the data file would throw exceptions.


```fsharp
open DataTypes
open LinearRegression
open Utilities
[<EntryPoint>]
let main argv = 
    let dat= new readData (@"/data/continuous.txt", "continuous")
    let seed=1 // random seed
    let folds=3 // split data into 3 folds
    let datFold = new data (dat.CreateFold folds seed ,dat.Features) // prepare data
    let fold=1 // use fold 1 for test and the others for train
    let xTrain,yTrain= datFold.Train fold
    // in case to train the model using all data:
    //let xTrain,yTrain= datFold.All 
    let xTest,yTest= datFold.Test fold

    // train a standard linear regression
    let lm= new LM(xTrain,yTrain)
    do lm.Fit()
    let pTrain = lm.Predict xTrain // make prediction on training data
    let pTest = lm.Predict xTest // make prediction on testing data
    let rmseTrain=RMSE yTrain pTrain // compute RMSE on training data
    let rmseTest=RMSE yTest pTest // compute RMSE on testing data
    printfn "train rmse: %A" rmseTrain
    printfn "test  rmse: %A" rmseTest
    printfn "beta: %A" (lm.Beta.ToArray())

    // train a linear regression with L2 penalty, i.e., ridge linear regression
    let lml2= new LMRidge(xTrain,yTrain,0.2) // lambda = 0.2 is the penalty parameter
    do lml2.Fit()
    let pTrainl2 = lml2.Predict xTrain // make prediction on training data
    let pTestl2 = lml2.Predict xTest // make prediction on testing data
    let rmseTrainl2=RMSE yTrain pTrainl2 // compute RMSE on training data
    let rmseTestl2=RMSE yTest pTestl2 // compute RMSE on testing data
    printfn "train rmse: %A" rmseTrainl2
    printfn "test  rmse: %A" rmseTestl2
    printfn "beta: %A" (lml2.Beta.ToArray())

    // train a linear regression with L1 penalty, i.e., lasso linear regression
    let lml1= new LMLasso(xTrain,yTrain,0.2) // lambda = 0.2 is the penalty parameter
    do lml1.Fit()
    let pTrainl1 = lml1.Predict xTrain // make prediction on training data
    let pTestl1 = lml1.Predict xTest // make prediction on testing data
    let rmseTrainl1=RMSE yTrain pTrainl1 // compute RMSE on training data
    let rmseTestl1=RMSE yTest pTestl1 // compute RMSE on testing data
    printfn "train rmse: %A" rmseTrainl1
    printfn "test  rmse: %A" rmseTestl1
    printfn "beta: %A" (lml1.Beta.ToArray())
    0
```


### 2. Logistic regression

We use the data /data/binary.txt, which is stored in the libsvm format.


```fsharp
open DataTypes
open LogisticRegression
open Utilities
[<EntryPoint>]
let main argv = 
    let dat= new readData (@"/data/binary.txt", "binary")
    let seed=1 // random seed
    let folds=3 // split data into 3 folds
    let datFold = new data (dat.CreateFold folds seed ,dat.Features) // prepare data
    let fold=1 // use fold 1 for test and the others for train
    let xTrain,yTrain= datFold.Train fold
    // in case to train the model using all data:
    //let xTrain,yTrain= datFold.All 
    let xTest,yTest= datFold.Test fold

    // train a standard logistic regression
    let lr= new LR(xTrain,yTrain)
    do lr.Fit()
    let pTrain = lr.Predict (xTrain, "response") // make prediction of probabilities on training data, if the second parameter is ignored, default link function would be used.
    let pTest = lr.Predict (xTest, "response") // make prediction of probabilities on testing data
    let aucTrain= AUC yTrain pTrain // compute auc on training data
    let aucTest= AUC yTest pTest // compute auc on testing data
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest
    printfn "beta: %A" (lr.Beta.ToArray())

    // train a logistic regression with L2 penalty, i.e., ridge logistic regression
    let lrl2= new LRRidge(xTrain,yTrain,0.2) // lambda = 0.2 is the penalty parameter
    do lrl2.Fit()
    let pTrainl2 = lrl2.Predict xTrain // make prediction on training data
    let pTestl2 = lrl2.Predict xTest // make prediction on testing data
    let aucTrainl2 =AUC yTrain pTrainl2 // compute auc on training data
    let aucTestl2 =AUC yTest pTestl2 // compute auc on testing data
    printfn "train auc: %A" aucTrainl2
    printfn "test  auc: %A" aucTestl2
    printfn "beta: %A" (lrl2.Beta.ToArray())

    // train a logistic regression with L1 penalty, i.e., lasso logistic regression
    let lrl1= new LRLasso(xTrain,yTrain,0.2) // lambda = 0.2 is the penalty parameter
    do lrl1.Fit()
    let pTrainl1 = lrl1.Predict xTrain // make prediction on training data
    let pTestl1 = lrl1.Predict xTest // make prediction on testing data
    let aucTrainl1 =AUC yTrain pTrainl1 // compute auc on training data
    let aucTestl1 =AUC yTest pTestl1 // compute auc on testing data
    printfn "train auc: %A" aucTrainl1
    printfn "test  auc: %A" aucTestl1
    printfn "beta: %A" (lrl1.Beta.ToArray())
    0
```

## 3. SVM

We use the data /data/binary.txt, which is stored in the libsvm format.


```fsharp
open DataTypes
open SVM
open Utilities
[<EntryPoint>]
let main argv = 
    let dat= new readData (@"/data/binary.txt", "binary")
    let seed=1 // random seed
    let folds=3 // split data into 3 folds
    let datFold = new data (dat.CreateFold folds seed ,dat.Features) // prepare data
    let fold=1 // use fold 1 for test and the others for train
    let xTrain,yTrain= datFold.Train fold
    // in case to train the model using all data:
    //let xTrain,yTrain= datFold.All 
    let xTest,yTest= datFold.Test fold
    
    // train a svm model with linear kernel 
    let svmLinear=new SVM (xTrain, yTrain,1.0) // default kernel is linear, and the penalty parameter lambda is set to 1.0

    do svmLinear.Fit()
    let pTrain = svmLinear.Predict xTrain
    let pTest = svmLinear.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "%A" "linear SVM:"
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest

    // train a svm model with rbf kernel
    let svmRBF=new SVM (xTrain, yTrain,0.2,"rbf",1.0) // 0.2 is the penalty parameter and 1.0 is the parameter for rbf kernel

    do svmRBF.Fit()
    let pTrainRBF = svmRBF.Predict xTrain
    let pTestRBF = svmRBF.Predict xTest
    let aucTrainRBF=AUC yTrain pTrainRBF
    let aucTestRBF=AUC yTest pTestRBF
    printfn "%A" "rbf SVM:"
    printfn "train auc: %A" aucTrainRBF
    printfn "test  auc: %A" aucTestRBF
    
    0
```

## 4. Gradient boosting machine (GBM)

The current version of GBM can train either Gaussian or binomial response. In this example we train a binary classification model using the data /data/binary.txt, which is stored in the libsvm format.


```fsharp
open DataTypes
open GBM
open Utilities
[<EntryPoint>]
let main argv = 
    let dat= new readData (@"/data/binary.txt", "binary")
    let seed=1 // random seed
    let folds=3 // split data into 3 folds
    let datFold = new data (dat.CreateFold folds seed ,dat.Features) // prepare data
    let fold=1 // use fold 1 for test and the others for train
    let xTrain,yTrain= datFold.Train fold
    // in case to train the model using all data:
    //let xTrain,yTrain= datFold.All 
    let xTest,yTest= datFold.Test fold
    
    // train a gbm model with the following parameters:
    // number of trees: 5
    // depth of each tree: 4
    // learning rate: 0.2
    // regularization parameter lambda: 0.1
    // regularization parameter gamma: 0.0
    // row(sample wise) subsample ratio: 0.7
    // col(feature wise) subsample ratio 0.6
    
    let gbm = GBM (xTrain,yTrain,"binomial",4,0.2,1.0,0.0,0.7,0.6)
    gbm.Fit(100)
    let pred = gbm.Predict (xTest ,"response")
    printfn "AUC: %A \t logloss: %A" (AUC yTest pred) (logloss yTest pred)
    
    // cross validation
    gbm.CVFit("AUC",100)
    0
```
