// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open DataTypes
open LogisticRegression
[<EntryPoint>]
let main argv = 
    // download sample data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes
    let dat= new readData @"c://R//diabetes.txt"
    let seed=1 // random seed
    let folds=3 // use 3-folds cross-validation
    let datFold = new data (dat.CreateFold folds seed ,dat.Features)
    let fold=1 // use fold 1 for test
    let xTrain,yTrain= datFold.Train fold
    let xTest,yTest= datFold.Test fold
    let lr= new LR (xTrain, yTrain)
    let iter=100
    do lr.Fit iter
    let pTrain = lr.Predict xTrain
    let pTest = lr.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest


