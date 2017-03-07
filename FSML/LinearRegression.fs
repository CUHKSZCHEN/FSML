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


    type LinearRidge (x:Matrix<double>,y:Vector<double>,lambda)=

        let n= y.Count
        let Lambda= lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        //let normalizedXWith1=normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        let beta0 = y.Sum()/(double n)
        member this.Predict (x:Vector<double>) = 
            beta0 + predictWith1 (this.Beta, (normalize ((V x), mu, sigma)))
        
        member this.Predict (x:Matrix<double>) =
            beta0 + predictWith1 (this.Beta, (normalize ((M x), mu , sigma)))

        member val Beta = (DenseVector.zero (x.ColumnCount)) with get,set

        member this.Fit () =
            let xTilde = DenseMatrix.stack [normalizedX; DenseVector.create normalizedX.ColumnCount (sqrt(Lambda)) |> DenseMatrix.ofDiag ]
            let yTilde =  Array.concat[(y-beta0).AsArray(); Array.zeroCreate normalizedX.ColumnCount] |> DenseVector.ofArray
            this.Beta <- QRUpdate xTilde yTilde
