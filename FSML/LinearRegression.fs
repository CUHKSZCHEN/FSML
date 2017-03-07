module LinearRegression

    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra


    type LinearR (x:Matrix<double>,y:Vector<double>)=

        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, (x.ToRowMatrix().InsertColumn(0, DenseVector.create x.Count 1.0)))
        
        member this.Predict (x:Matrix<double>) =
                    predictWith1 (this.Beta, (x.InsertColumn(0, DenseVector.create x.RowCount 1.0)))

        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        // chol decomposition
        //member this.Fit () = this.Beta <-  ( xWith1.TransposeThisAndMultiply(xWith1).Cholesky().Solve(xWith1.TransposeThisAndMultiply(y)) )
        // QR decomposition
        member this.Fit () = 
            this.Beta <- QRUpdate xWith1 y
