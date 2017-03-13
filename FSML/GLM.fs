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
