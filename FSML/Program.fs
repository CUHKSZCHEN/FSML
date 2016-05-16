open DataTypes
open LogisticRegression
[<EntryPoint>]
let main argv = 
    // download sample data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes
    let dat= new readData @"/Users/nl/Downloads/diabetes.txt"
    let seed=1 // random seed
    let folds=3 // use 3-folds cross-validation
    let datFold = new data (dat.CreateFold folds seed ,dat.Features)
    let fold=1 // use fold 1 for test
    let xTrain,yTrain= datFold.Train fold
    let xTest,yTest= datFold.Test fold
    let lr= new LR (xTrain, yTrain)
    let iter=100
    do lr.Fit
    let pTrain = lr.Predict xTrain
    let pTest = lr.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest

    let lasso = new LASSO (xTrain, yTrain,0.1)
    do lasso.Fit
    let pTrain = lasso.Predict xTrain
    let pTest = lasso.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest

    let ridge = new RIDGE (xTrain, yTrain,0.1)
    do ridge.Fit
    let pTrain = ridge.Predict xTrain
    let pTest = ridge.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest
    0
