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
