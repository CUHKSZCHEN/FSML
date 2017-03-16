module LinearRegression

    open DataTypes
    open Utilities
    open GLM
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type LM (xTrain:Matrix<double>,yTrain:Vector<double>)=
        inherit model()
        let x,y=xTrain,yTrain
        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        override this.Family =  "Gaussian"
        override this.Penalty = "None"

        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        override this.Fit (?decomposition:string) = 
            let decomp = defaultArg decomposition "QR"
            let updateF = match decomp with 
                | InvariantEqual "QR" -> QRUpdate
                | InvariantEqual "SVD" -> SVDUpdate 
                | _ -> raiseException "select either QR or SVD"
            this.Beta <- updateF xWith1 y
        
        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinear(this.Beta.[1..],V x) + this.Beta.[0]
            predictMatchGLM link (this.Family) value

        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinear(this.Beta.[1..],M x) + this.Beta.[0]
            predictMatchGLM link (this.Family) value

 
    type LMRidge (xTrain:Matrix<double>,yTrain:Vector<double>,lambda)=
        inherit model()
        let x,y=xTrain,yTrain

        let n= y.Count
        let Lambda=  lambda

        let checkLambda = if Lambda < 0.0 then raiseException "lambda should be positive"

        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)

        override this.Family = "Gaussian"
        override this.Penalty = "L2"
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],V x,mu,sigma ) + this.Beta.[0]
            predictMatchGLM link (this.Family) value


        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..], M x, mu, sigma) + this.Beta.[0]
            predictMatchGLM link (this.Family) value


        override this.Fit (?decomposition:string) =
            let decomp = defaultArg decomposition "QR"
            let beta0 = y.Sum()/(double n)
            let xTilde = DenseMatrix.stack [normalizedX; DenseVector.create normalizedX.ColumnCount (sqrt(Lambda)) |> DenseMatrix.ofDiag ]
            let yTilde =  Array.concat[(y-beta0).AsArray(); Array.zeroCreate normalizedX.ColumnCount] |> DenseVector.ofArray
            let updateF = match decomp with 
                | InvariantEqual "QR" -> QRUpdate
                | InvariantEqual "SVD" -> SVDUpdate 
                | _ -> raiseException "select either QR or SVD"
            this.Beta <- Array.concat[[|beta0|]; (updateF xTilde yTilde).AsArray()] |> DenseVector.ofArray

    type LMLasso (xTrain:Matrix<double>,yTrain:Vector<double>,lambda)=
        inherit model()
        let x,y=xTrain,yTrain
        let n= y.Count
        let Lambda=lambda 
        let checkLambda = if Lambda < 0.0 then raiseException "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXwith1= normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        let coordinateDescent (beta:Vector<double>) (co:int)=
            let yiTilde = predictWith1 (beta, normalizedXwith1) - beta.[co]*normalizedXwith1.Column(co)
            let z=normalizedXwith1.Column(co)* (y-yiTilde)/(double n)
            if co = 0 then z
            else (STO z lambda)

        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=
            for i in [0..beta.Count-1] do 
                beta.Item(i) <- (coordinateDescent beta i)
            let yiTilde = predictWith1 (beta, normalizedXwith1)
            let J = (y-yiTilde)*(y-yiTilde)/(double n)/2.0  
            let loss = J + (beta.[1..].Map (fun e -> abs(e))).Sum()*Lambda
            do printfn "Real loss: %10.15f \t penalized Loss: %10.15f" J  loss
            beta, loss
        
        override this.Family = "Gaussian"
        override this.Penalty = "L1"

        member val eps = 1e-16 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        override this.Fit(?decomposition:string) = this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],V x,mu,sigma ) + this.Beta.[0]
            predictMatchGLM link (this.Family) value


        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],M x,mu,sigma ) + this.Beta.[0]
            predictMatchGLM link (this.Family) value
