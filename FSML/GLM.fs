module GLM
    
    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    [<AbstractClass>]
    type model()=
        abstract Family: string with get
        abstract Penalty: string with get
        abstract member Fit: unit -> unit
        abstract member Predict: Vector<double> * ?value:string -> Vector<double>
        abstract member Predict: Matrix<double> * ?value:string -> Vector<double>

    //type glm (xTrain:Matrix<double>,yTrain:Vector<double>,family:string,?penlaty:string,?lambda:double)=
    //    let Lambda = defaultArg lambda 0.0
    //    let Penalty = defaultArg penlaty "None"
    //    member this.model =
    //                       match (family,Penalty) with
    //                       | (family,penalty) when (family = "Gaussian" || family = "gaussian") && (penalty = "L2")  -> LMRidge(xTrain,yTrain,Lambda)
    //                       | (family,penalty) when (family = "Gaussian" || family = "gaussian") && (penalty = "L1")  -> LMLasso(xTrain,yTrain,Lambda)
    //                       | (family,penalty) when (family = "Gaussian" || family = "gaussian") && (penalty = "None")  -> LM(xTrain,yTrain)
    //                       | (family,penalty) when (family = "binomial") && (penalty = "None")  -> LR(xTrain,yTrain)
    //                       | (family,penalty) when (family = "binomial") && (penalty = "L2")  -> LRRidge(xTrain,yTrain,Lambda)
    //                       | (family,penalty) when (family = "binomial") && (penalty = "L1")  -> LRLasso(xTrain,yTrain,Lambda)
    //                       | (family,penalty) -> raiseExcetion "not implemented"
    
