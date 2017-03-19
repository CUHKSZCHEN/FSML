module LogisticRegression
    
    open DataTypes
    open Utilities
    open GLM
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    
    type LR (xTrain:Matrix<double>,yTrain:Vector<double>)=
        inherit model()
        let x,y=xTrain,yTrain
        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        
        let reWeightedUpdate (beta:Vector<double>) updateF =
            let s = predictWith1 (beta, xWith1)
            let p = (s).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik= Loglik s y
            do loglik |> printfn "Loglikelihood: %10.15f"          
            let w= p .* (1.0 - p)
            let z= xWith1 * beta +  DiagonalMatrix.ofDiag(1.0/w) *  (y-p)
            updateF xWith1 z w, -loglik
        
        let reWeightedQRUpdate (beta:Vector<double>)  = reWeightedUpdate beta WeightedQRUpdate
        let reWeightedSVDUpdate (beta:Vector<double>)  = reWeightedUpdate beta WeightedSVDUpdate

                
        override this.Family = "binomial"
        override this.Penalty = "None"

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, (x.ToRowMatrix().InsertColumn(0, DenseVector.create x.Count 1.0)))
        
        member this.Predict (x:Matrix<double>) =
                    predictWith1 (this.Beta, (x.InsertColumn(0, DenseVector.create x.RowCount 1.0)))


        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinear(this.Beta.[1..],V x) + this.Beta.[0]
            predictMatchGLM link (this.Family) value


        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinear(this.Beta.[1..],M x) + this.Beta.[0]
            predictMatchGLM link (this.Family) value


        member val eps = 1e-16 with get,set
        member val maxIter = 1000 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        override this.Fit (?decomposition:string) = 
            let decomp = defaultArg decomposition "QR"
            let updateF = match decomp with 
                | InvariantEqual "QR" -> reWeightedQRUpdate
                | InvariantEqual "SVD" -> reWeightedSVDUpdate 
                | _ -> raiseException "select either QR or SVD"
            this.Beta <- (update updateF this.Beta this.eps 1 this.maxIter this.minIter 0.0)

     type LRRidge (xTrain:Matrix<double>,yTrain:Vector<double>,lambda)=
        inherit model()
        let x,y=xTrain,yTrain

        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseException "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXwith1= normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        let reWeightedUpdate (beta:Vector<double>) (updateF: Matrix<double> -> Vector<double> -> Vector<double> -> Vector<double>) =

            let eta = predictWith1 (beta, normalizedXwith1)
            let p = (eta ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w = p.*(1.0 - p)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            let pTilde = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)

            let wTilde =  Array.concat[wNew.AsArray(); [| for e in 1..normalizedXwith1.ColumnCount -> 1.0|]] |> DenseVector.ofArray
            let xTilde = DenseMatrix.stack [normalizedXwith1; DenseVector.create normalizedXwith1.ColumnCount (sqrt(double n * Lambda)) |> DenseMatrix.ofDiag ]
            do xTilde.Item(n,0) <- 0.0
            let z = eta + (y - pTilde)./wNew
            let zTilde = Array.concat[(z).AsArray(); Array.zeroCreate normalizedXwith1.ColumnCount] |> DenseVector.ofArray

            let beta_new =  (updateF xTilde zTilde wTilde).ToArray() |> DenseVector.ofArray

            let eta_new = predictWith1 (beta_new, normalizedXwith1)
            let loglik_new = Loglik eta_new y
            let loss_new = -loglik_new/(double n) + (beta_new.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            do printfn "Real Loglikelihood: %10.15f \t loss: %10.15f" (loglik_new) loss_new
            beta_new, loss_new            
        
        let reWeightedQRUpdate (beta:Vector<double>)  = reWeightedUpdate beta WeightedQRUpdate
        let reWeightedSVDUpdate (beta:Vector<double>)  = reWeightedUpdate beta WeightedSVDUpdate

        member val eps = 1e-16 with get,set
        member val maxIter = 1000 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        override this.Family = "binomial"
        override this.Penalty = "L2"
        override this.Fit (?decomposition:string) = 
            let decomp = defaultArg decomposition "QR"
            let updateF = match decomp with 
                | InvariantEqual "QR" -> reWeightedQRUpdate
                | InvariantEqual "SVD" -> reWeightedSVDUpdate 
                | _ -> raiseException "select either QR or SVD"
            this.Beta <- (update updateF this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        member this.Predict (x:Vector<double>) = 
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((V x), mu, sigma))
        
        member this.Predict (x:Matrix<double>) =
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((M x), mu, sigma))

        
        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],V x,mu,sigma ) + this.Beta.[0] 
            predictMatchGLM link (this.Family) value


        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],M x,mu,sigma ) + this.Beta.[0] 
            predictMatchGLM link (this.Family) value


    type LRLasso (xTrain:Matrix<double>,yTrain:Vector<double>,lambda)=
        inherit model()
        let x,y=xTrain,yTrain
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseException "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXwith1= normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        let mutable response = DenseVector.create n 0.0
        let mutable weight= DenseVector.create n 1.0

        let coordinateDescent (beta:Vector<double>) (co:int)=            

            let yiTilde = predictWith1 (beta, normalizedXwith1) - beta.[co]*normalizedXwith1.Column(co)
            let z=normalizedXwith1.Column(co) .*weight * (response-yiTilde) / (double n) 
            let d= weight.*normalizedXwith1.Column(co)*normalizedXwith1.Column(co)/(double n)
            if co = 0 then z/d
            else (STO z lambda)/d

        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=
            let eta = predictWith1 (beta, normalizedXwith1)
            let p = (eta ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w =  p.*(1.0 - p)

            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            do weight <- Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            do response <- eta + (y - pNew)./weight

            [0..beta.Count-1] |> List.iter (fun i ->
                beta.Item(i) <- (coordinateDescent beta i)
                )
                
            let eta_new = predictWith1 (beta, normalizedXwith1)

            let loglik_new = Loglik eta_new y
            let loss_new = -loglik_new/(double n) + (beta.[1..].Map (fun e -> abs(e))).Sum()*Lambda
            do printfn "Real Loglikelihood: %10.15f \t loss: %10.15f" (loglik_new) loss_new
            beta, loss_new

        override this.Family = "binomial"
        override this.Penalty = "L1"
        member val eps = 1e-16 with get,set
        member val maxIter = 1000 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        override this.Fit (?decomposition:string) = 
            this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        override this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],V x,mu,sigma ) + this.Beta.[0] 
            predictMatchGLM link (this.Family) value

        override this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = predictLinearScale(this.Beta.[1..],M x,mu,sigma ) + this.Beta.[0] 
            predictMatchGLM link (this.Family) value
