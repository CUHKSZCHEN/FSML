module GLM
    
    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    let predictMatchGLM (link:Vector<double>) (family:string) (value:string)= 
        match value,family with
        | "link",_ -> link
        | "response", InvariantEqual "Gaussian" -> link
        | "response", InvariantEqual "binomial" -> logistic(link)
        | _ -> raiseException "predict either link or response"


    [<AbstractClass>]
    type model()=
        abstract Family: string with get
        abstract Penalty: string with get
        abstract member Fit: ?decomposition:string -> unit
        abstract member Predict: Vector<double> * ?value:string -> Vector<double>
        abstract member Predict: Matrix<double> * ?value:string -> Vector<double>
