module LogisticRegression
    
    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    

    //let predictWith1 (beta, xWith1:Matrix<double>)= xWith1 * beta


    
    type LR (x:Matrix<double>,y:Vector<double>)=

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

     type RIDGE (x:Matrix<double>,y:Vector<double>,lambda)=
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)

        let reWeightedUpdate (beta:Vector<double>) =
            
            let betaj = beta.[1..]
            let eta = beta.[0] + predictWith1 (betaj, normalizedX)
            let pTilde = (eta ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w = pTilde.*(1.0 - pTilde)
            let wTilde =  Array.concat[w.AsArray(); [| for e in 1..normalizedX.ColumnCount -> 1.0|]] |> DenseVector.ofArray
            let xTilde = DenseMatrix.stack [normalizedX; DenseVector.create normalizedX.ColumnCount (sqrt(Lambda)) |> DenseMatrix.ofDiag ]
            let z = eta + (y - pTilde)./w
            let beta0 = z.Sum()/(double n)
            let zTilde = Array.concat[(z- beta0).AsArray(); Array.zeroCreate normalizedX.ColumnCount] |> DenseVector.ofArray
            let loglik = Loglik eta y
            let j1= -w.*(z-eta) *(z-eta)/2.0/(double n) + (betaj.Map (fun e -> e*e)).Sum()*Lambda/2.0

            let loss = -loglik/(double n)/2.0 + (betaj.Map (fun e -> e*e)).Sum()*Lambda/2.0
            do printfn "before Real Loglikelihood: %A \t loss: %A \t j:%A" (Loglik eta y) loss j1
            printfn "before beta: %A" (beta.[0])
            let np = Array.concat[Array.create 1 beta0; (WeightedQRUpdate xTilde zTilde wTilde).ToArray()] |> DenseVector.ofArray
            let loglik1 = Loglik (np.[0] + predictWith1 (np.[1..], normalizedX)) y
            let loss1 = -loglik1/(double n)/2.0 + (np.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            let j2= -w.*(z- beta0 - predictWith1 (np.[1..], normalizedX))*(z-beta0- predictWith1 (np.[1..], normalizedX))/2.0/(double n) + (np.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            
            do printfn "after Real Loglikelihood: %A \t loss: %A \t j:%A" (loglik1) loss1 j2
            printfn "after beta: %A" (np.[0])

            Array.concat[Array.create 1 beta0; (WeightedQRUpdate xTilde zTilde wTilde).ToArray()] |> DenseVector.ofArray ,loss            
        
        member val eps = 1e-16 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        member this.Fit () = 
            this.Beta <- (update reWeightedUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        member this.Predict (x:Vector<double>) = 
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((V x), mu, sigma))
        
        member this.Predict (x:Matrix<double>) =
            this.Beta.[0] + predictWith1 (this.Beta.[1..], normalize ((M x), mu , sigma))


    type LASSO (x:Matrix<double>,y:Vector<double>,lambda)=
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXWith1=normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        let coordinateDescent (betaNew:Vector<double>) (co:int)=

            let s=predictWith1 (betaNew, normalizedXWith1)
            let p=(s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)          
            let w=p .* (p.Negate().Add(1.0))
            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            let zNew= normalizedX * betaNew + (DiagonalMatrix.ofDiag wNew).Inverse() *  (y-pNew)
            if co = 0 then
                let g= -wNew*((y-pNew)./wNew)/(double n)
                let h= wNew.Sum()/(double n)
                betaNew.Item(0)-g/h
            else
                let sto=(normalizedX.Column(co).*wNew.*(zNew-s)).Sum()
                let scale=(normalizedX.Column(co).*normalizedX.Column(co).*wNew).Sum()
                if (sto>0.0 && Lambda<sto) then (sto - Lambda)/scale else 
                    if (sto<0.0 && Lambda < -sto) then (sto + Lambda)/scale else 0.0
//                let g1 = -(normalizedXWith1.Column(co).*((z-pNew)./wNew).*wNew).Sum()/(double n) + Lambda
//                let h = wNew*(normalizedXWith1.Column(co).*normalizedXWith1.Column(co))/(double n)
//                if betaNew.Item(co) - g1/h> 0.0 then betaNew.Item(co) - g1/h else
//                    let g2 = g1 - 2.0 * Lambda
//                    if betaNew.Item(co) - g2/h< 0.0 then betaNew.Item(co) - g2/h else 0.0
   
        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=
            for i in [0..beta.Count-1] do 
                beta.Item(i) <- (coordinateDescent beta i)
            let s=predictWith1 (beta, normalizedX)
            let p=(s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w=p .* (p.Negate().Add(1.0))
            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            let zNew= normalizedX * beta + (DiagonalMatrix.ofDiag wNew).Inverse() *  (y-pNew)
            let approLik= -(((y-pNew))*((y-pNew)./w))/(double n)/2.0  
            let loss =approLik + (beta.[1..].Map (fun e -> abs(e))).Sum()*Lambda
            do printfn "Real Loglikelihood: %A \t Approx scaled Loglikelihood: %A \t loss: %A" (Loglik s y) approLik  loss
            beta,-loss

        member val eps = 1e-16 with get,set
        member val maxIter = 100000 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        member this.Fit() = this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, ((normalize ((V x), mu, sigma)).InsertColumn(0, DenseVector.create 1 1.0)))
        
        member this.Predict (x:Matrix<double>) =
            predictWith1 (this.Beta, ((normalize ((M x), mu , sigma)).InsertColumn(0, DenseVector.create x.RowCount 1.0)))
