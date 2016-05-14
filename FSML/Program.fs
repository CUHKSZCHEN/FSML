// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open DataTypes
open LogisticRegression
[<EntryPoint>]
let main argv = 

    let dat= new readData @"c://R//diabetes1.txt"
    let seed=1
    let folds=3
    let datFold = new data (dat.CreateFold folds seed ,dat.Features)
    let fold=1
    let xTrain,yTrain= datFold.Train fold
    let xTest,yTest= datFold.Test fold
    let lr= new LR (xTrain, yTrain)
    let iter=100
    do (lr.Fit iter)

    //lr.p
    //let p= lrfit.p
    0 // return an integer exit code


