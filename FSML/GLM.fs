module GLM
    
    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    let predictMatchGLM (link:Vector<double>) (family:string) (value:string)= 
        match value,family with
        | "link",_ -> link
        | "response", "gaussian" -> link
        | "response", "Gaussian" -> link
        | "response", "binomial" -> logistic(link)
        | _ -> raiseExcetion "predict either link or response"


    [<AbstractClass>]
    type model()=
        abstract Family: string with get
        abstract Penalty: string with get
        abstract member Fit: unit -> unit
        abstract member Predict: Vector<double> * ?value:string -> Vector<double>
        abstract member Predict: Matrix<double> * ?value:string -> Vector<double>
