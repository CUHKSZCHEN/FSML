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
        override this.Family = "LM"
        override this.Penalty = "None"

        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        override this.Fit () = 
            this.Beta <- QRUpdate xWith1 y
        
        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinear(this.Beta.[1..],V x) + this.Beta.[0]
            match value with 
            | "link" -> link
            | "response" -> link
            | _ -> raiseExcetion "predict either link or response"

        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinear(this.Beta.[1..],M x) + this.Beta.[0]
            match value with 
            | "link" -> link
            | "response" -> link
            | _ -> raiseExcetion "predict either link or response"
 
    type LMRidge (xTrain:Matrix<double>,yTrain:Vector<double>,lambda)=
        inherit model()
        let x,y=xTrain,yTrain

        let n= y.Count
        let Lambda=  lambda

        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"

        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let mutable beta0 = y.Sum()/(double n)

        override this.Family = "LM"
        override this.Penalty = "L2"

        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta,V x,mu,sigma ) + beta0
            match value with 
            | "link" -> link
            | "response" -> link
            | _ -> raiseExcetion "predict either link or response"

        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta, M x, mu, sigma) + beta0
            match value with 
            | "link" -> link
            | "response" -> link
            | _ -> raiseExcetion "predict either link or response"

        member val Beta = (DenseVector.zero (x.ColumnCount)) with get,set

        override this.Fit () =
            let xTilde = DenseMatrix.stack [normalizedX; DenseVector.create normalizedX.ColumnCount (sqrt(Lambda)) |> DenseMatrix.ofDiag ]
            let yTilde =  Array.concat[(y-beta0).AsArray(); Array.zeroCreate normalizedX.ColumnCount] |> DenseVector.ofArray
            this.Beta <- QRUpdate xTilde yTilde

    type LMLasso (xTrain:Matrix<double>,yTrain:Vector<double>,lambda)=
        inherit model()
        let x,y=xTrain,yTrain
        let n= y.Count
        let Lambda=lambda 
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let beta0 = y.Sum()/(double n)

        let coordinateDescent (betaNew:Vector<double>) (co:int)=
            let yiTilde = beta0 + predictWith1 (betaNew, normalizedX) - betaNew.[co]*normalizedX.Column(co)
            let z=normalizedX.Column(co)* (y-yiTilde)/(double n)      
            STO z lambda

        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=
            for i in [0..beta.Count-1] do 
                beta.Item(i) <- (coordinateDescent beta i)
            let yiTilde = beta0 + predictWith1 (beta, normalizedX)
            let J = (y-yiTilde)*(y-yiTilde)/(double n)/2.0  
            let loss = J + (beta.Map (fun e -> abs(e))).Sum()*Lambda
            do printfn "Real loss: %10.15f \t penalized Loss: %10.15f" J  loss
            beta, loss
        
        override this.Family = "LM"
        override this.Penalty = "L1"

        member val eps = 1e-16 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount)) with get,set
        
        override this.Fit() = this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta,V x,mu,sigma ) + beta0
            match value with 
            | "link" -> link
            | "response" -> link
            | _ -> raiseExcetion "predict either link or response"

        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta, M x, mu, sigma) + beta0
            match value with 
            | "link" -> link
            | "response" -> link
            | _ -> raiseExcetion "predict either link or response"

   
