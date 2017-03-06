open DataTypes
open LogisticRegression
[<EntryPoint>]
let main argv = 
    // download sample data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes
    let dat= new readData (@"/Downloads/diabetes.txt", "binary")
    let seed=1 // random seed
    let folds=3 // use 3-folds cross-validation
    let datFold = new data (dat.CreateFold folds seed ,dat.Features)
    let fold=1 // use fold 1 for test
    let xTrain,yTrain= datFold.Train fold
    let xTest,yTest= datFold.Test fold
    printfn "%A" "logistic regression:"
    let lr= new LR (xTrain, yTrain)
    let iter=100
    do lr.Fit()
    let pTrain = lr.Predict xTrain
    let pTest = lr.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest

    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest
    printfn "beta: %A" (lr.Beta.ToArray())

    let lasso = new LASSO (xTrain, yTrain,0.1)
    do lasso.Fit()
    let pTrain = lasso.Predict xTrain
    let pTest = lasso.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "%A" "lasso regression:"
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest

    printfn "beta: %A" (lasso.Beta.ToArray())


    let ridge = new RIDGE (xTrain, yTrain,0.1)
    do ridge.Fit()
    let pTrain = ridge.Predict xTrain
    let pTest = ridge.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "%A" "ridge regression:"
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest
    printfn "beta: %A" (ridge.Beta.ToArray())


    let ridge1= new RIDGE (xTrain, yTrain,0.1,"cd")
    do ridge1.Fit()
    let pTrain = ridge1.Predict xTrain
    let pTest = ridge1.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "%A" "ridge exact regression:"
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest
    printfn "beta: %A" (ridge1.Beta.ToArray())

    let svm=new SVM (xTrain, yTrain,1.0)

    do svm.Fit()
    let pTrain = svm.Predict xTrain
    let pTest = svm.Predict xTest
    let aucTrain=AUC yTrain pTrain
    let aucTest=AUC yTest pTest
    printfn "%A" "SVM classification:"
    printfn "train auc: %A" aucTrain
    printfn "test  auc: %A" aucTest

    0
