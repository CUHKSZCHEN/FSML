module LogisticRegression
    
    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    
    type Lr (x:Matrix<double>,y:Vector<double>)=

        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        
        let reWeightedUpdate (beta:Vector<double>)  =
            let s = predictWith1 (beta, xWith1)
            let p = (s).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik= Loglik s y
            do loglik |> printfn "Loglikelihood: %A"          
            let w= p .* (1.0 - p)
            let z= xWith1 * beta +  DiagonalMatrix.ofDiag(1.0/w) *  (y-p)
            WeightedQRUpdate xWith1 z w, -loglik
        
        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, (x.ToRowMatrix().InsertColumn(0, DenseVector.create x.Count 1.0)))
        
        member this.Predict (x:Matrix<double>) =
                    predictWith1 (this.Beta, (x.InsertColumn(0, DenseVector.create x.RowCount 1.0)))

        member val eps = 1e-12 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        member this.Fit () = this.Beta <- (update reWeightedUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

     type LrRidge (x:Matrix<double>,y:Vector<double>,lambda)=
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXwith1= normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        let reWeightedUpdate (beta:Vector<double>) =

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

            let beta_new =  (WeightedQRUpdate xTilde zTilde wTilde).ToArray() |> DenseVector.ofArray

            let eta_new = predictWith1 (beta_new, normalizedXwith1)
            let loglik_new = Loglik eta_new y
            let loss_new = -loglik_new/(double n) + (beta_new.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            do printfn "Real Loglikelihood: %A \t loss: %A" (loglik_new) loss_new
            beta_new, loss_new            
        
        member val eps = 1e-16 with get,set
        member val maxIter = 1000 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        member this.Fit () = 
            this.Beta <- (update reWeightedUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        member this.Predict (x:Vector<double>) = 
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((V x), mu, sigma))
        
        member this.Predict (x:Matrix<double>) =
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((M x), mu, sigma))

    type LrLasso (x:Matrix<double>,y:Vector<double>,lambda)=
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let mutable response = DenseVector.create n 0.0
        let mutable weight= DenseVector.create n 1.0
        let coordinateDescent (beta:Vector<double>) (co:int)=            

            let yiTilde = beta.[0] + predictWith1 (beta.[1..], normalizedX) - beta.[co]*normalizedX.Column(co-1)
            let z=normalizedX.Column(co-1) .*weight * (response-yiTilde) / (double n)    
            (STO z lambda)/(weight.*normalizedX.Column(co-1)*normalizedX.Column(co-1)/(double n))

        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=
            let eta = beta.[0] + predictWith1 (beta.[1..], normalizedX)
            let p = (eta ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w =  p.*(1.0 - p)

            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            do weight <- Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            do response <- eta + (y - pNew)./weight

            do beta.Item(0) <- response.Sum()/(double n)

            for i in [1..beta.Count-1] do 
                beta.Item(i) <- (coordinateDescent beta i)

            let eta_new = beta.[0] + predictWith1 (beta.[1..], normalizedX)

            let loglik_new = Loglik eta_new y
            let loss_new = -loglik_new/(double n)/2.0 + (beta.[1..].Map (fun e -> abs(e))).Sum()*Lambda
            do printfn "Real Loglikelihood: %A \t loss: %A" (loglik_new) loss_new
            beta, loss_new

        member val eps = 1e-16 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 20 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        member this.Fit () = 
            this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        member this.Predict (x:Vector<double>) = 
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((V x), mu, sigma))
        
        member this.Predict (x:Matrix<double>) =
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((M x), mu, sigma))
