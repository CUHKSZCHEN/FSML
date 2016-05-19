module LogisticRegression
    
    open DataTypes
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Statistics

    let AUC (y:Vector<double>) (score:Vector<double>)=
        let n1= y.Sum()
        let n0= (double) y.Count-n1
        let sortedScore = Array.zip (score.ToArray()) (y.Negate().ToArray()) |> Array.sortBy (fun e-> e|> snd) |> Array.map fst
        let rank= (sortedScore |> DenseVector.ofArray) .Ranks RankDefinition.Average
        ((rank.[0..(int)n1-1] |> Array.sum )- n1* (n1+1.0)/2.0)/n0/n1

    let Loglik (s:Vector<double>) (y:Vector<double>)=
        y*s - s.PointwiseExp().Add(1.0).PointwiseLog().Sum()

    let logloss (p:Vector<double>) (y:Vector<double>)=
        let EPSLow=1e-15
        let EPSHigh=1.0-EPSLow 
        let pNew=p.Map (fun e -> if e<EPSLow then EPSLow else if e>EPSHigh then EPSHigh else e)
        -(y*pNew.PointwiseLog() + (y.Negate().Add(1.0))*pNew.Negate().Add(1.0).PointwiseLog()) /(double p.Count)

    let SoftThresholdingOperator (b:double, lambda:double) = 
        if b >0.0 then b-lambda else
            if b<0.0 then b+lambda else 0.0 
    
    let getNormalizeParameter (x:Matrix<double>) =
        x.ColumnSums().Divide(double x.RowCount),x.ToColumnArrays() |> Array.map (fun col -> col.StandardDeviation() ) |> DenseVector.ofArray

    let normalize (x:VectorOrMatrix,mu:Vector<double>,sd:Vector<double>) =
        match x with
        | V v -> (v.Add(mu.Negate())./sd).ToRowMatrix()
        | M m -> m.ToRowArrays() |> Array.map (fun row -> ((row |> DenseVector.ofArray).Add(mu.Negate())./(sd)).ToArray() ) |> DenseMatrix.ofRowArrays
    
    let predictWith1 (beta, xWith1:Matrix<double>)= xWith1 * beta

    let rec update f (parameter:Vector<double>) (eps:double) (iter:int) (maxIter:int) (minIter:int) (lossOld:double) =
        if iter >= maxIter then parameter
        else
            let paramNew,lossNew = f parameter
            if lossOld - lossNew <  eps && iter >= minIter then parameter
            else update f paramNew eps (iter+1) maxIter minIter lossOld
    
    type LR (x:Matrix<double>,y:Vector<double>)=

        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        let reWeightedUpdate (beta:Vector<double>) =
            let s=predictWith1 (beta, xWith1)
            let p = (s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik = Loglik s y
            do loglik |> printfn "Loglikelihood: %A"          
            let w= DiagonalMatrix.ofDiag (p .* p.Negate().Add(1.0))
            let h = -xWith1.Transpose() * w *xWith1
            let g=  xWith1.Transpose()*(y-p)
            beta-h.Inverse()*g,-loglik

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, (x.ToRowMatrix().InsertColumn(0, DenseVector.create x.Count 1.0)))
        
        member this.Predict (x:Matrix<double>) =
                    predictWith1 (this.Beta, (x.InsertColumn(0, DenseVector.create x.RowCount 1.0)))

        member val eps = 1e-6 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        member this.Update (beta:Vector<double>)  =
            let s = predictWith1 (beta, xWith1)
            let p = (s).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik= Loglik s y
            do loglik |> printfn "Loglikelihood: %A"          
            let w= DiagonalMatrix.ofDiag (p .* p.Negate().Add(1.0))
            let z= xWith1 * beta + w.Inverse() *  (y-p)
            (xWith1.Transpose() * w *xWith1).Inverse() * xWith1.Transpose() * w*z,-loglik

        member this.Fit = this.Beta <- (update reWeightedUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

     type RIDGE (x:Matrix<double>,y:Vector<double>,lambda,algorithm:string)=
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-6
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXWith1=normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        let reWeightedUpdate (beta:Vector<double>) =

            let s=predictWith1 (beta, normalizedXWith1)
            let p = (s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik = Loglik s y
            let loss = -loglik/(double n)/2.0 + (beta.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            //do loglik |> printfn "Loglikelihood: %A"          
            let w= DiagonalMatrix.ofDiag (p .* p.Negate().Add(1.0))
            //let approLik= -(((y-p))*((y-p)./(p .* p.Negate().Add(1.0))))/(double n)/2.0  
            //let loss =approLik - (beta.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            do printfn "Real Loglikelihood: %A \t loss: %A" (Loglik s y) loss
  
            let m = DenseVector.create beta.Count Lambda
            m.Item(0) <- 0.0
            let h = -(normalizedXWith1.Transpose() * w *normalizedXWith1).Divide(double n) - (DiagonalMatrix.ofDiag (m))
            let k =Lambda*beta
            k.Item(0) <- 0.0
            let g= ((normalizedXWith1.Transpose()*(y-p)).Divide(double n))-k
            beta-h.Inverse()*g,loss

        let coordinateDescent (betaNew:Vector<double>) (co:int)=
            let s=predictWith1 (betaNew, normalizedXWith1)
            let p=(s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)          
            let w=p .* (p.Negate().Add(1.0))
            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            if co = 0 then
                let g= -wNew*((y-pNew)./wNew)/(double n)
                let h= wNew.Sum()/(double n)
                betaNew.Item(0)-g/h
            else
                let g = -(normalizedXWith1.Column(co).*((y-pNew)./wNew).*wNew).Sum()/(double n) + Lambda*betaNew.At(co)
                let h = wNew*(normalizedXWith1.Column(co).*normalizedXWith1.Column(co))/(double n) + Lambda   
                betaNew.Item(co) - g/h
                               
        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=
            for i in [0..beta.Count-1] do 
                beta.Item(i) <- (coordinateDescent beta i)
            let s=predictWith1 (beta, normalizedXWith1)
            let p=(s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w=p .* (p.Negate().Add(1.0))
            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            let zNew= normalizedXWith1 * beta + (DiagonalMatrix.ofDiag wNew).Inverse() *  (y-pNew)
            let approLik= -(((y-pNew))*((y-pNew)./w))/(double n)/2.0  
            let loss =approLik + (beta.[1..].Map (fun e -> e*e)).Sum()*Lambda/2.0
            do printfn "Real Loglikelihood: %A \t Approx scaled Loglikelihood: %A\t loss: %A" (Loglik s y) approLik  loss
            beta,-loss
        
        new (x,y,lambda) = RIDGE(x,y,lambda,"exact")
        member val eps = 1e-16 with get,set
        member val maxIter = 100 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        member this.Fit = 
            match algorithm with
            | "cd" ->
                this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)
            | "exact" -> this.Beta <- (update reWeightedUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)
            | _ -> raiseExcetion "please choose either \"cd\" or \"exact\" algorithm"

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, ((normalize ((V x), mu, sigma)).InsertColumn(0, DenseVector.create 1 1.0)))
        
        member this.Predict (x:Matrix<double>) =
            predictWith1 (this.Beta, ((normalize ((M x), mu , sigma)).InsertColumn(0, DenseVector.create x.RowCount 1.0)))


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
            let zNew= s + (y-pNew)./wNew
   
            if co = 0 then
                betaNew.Item(0)+wNew*(y-pNew)/(double (n*n))
            else
                let a = -wNew*(normalizedXWith1.Column(co).*normalizedXWith1.Column(co))/(double n)
                let b = (normalizedXWith1.Column(co))*((y-pNew))/(double n)
                let d=a*(b-Lambda)       
                if betaNew.Item(co) > d then betaNew.Item(co) - d
                else
                    let e= a*(b+Lambda)
                    if betaNew.Item(co) < e then betaNew.Item(co) - e else 0.0
                     
        let cyclicCoordinateDescentUpdate (beta:Vector<double>)=       
            for i in [0..beta.Count-1] do 
                beta.Item(i) <- (coordinateDescent beta i)
            let s=predictWith1 (beta, normalizedXWith1)
            let p=(s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w=p .* (p.Negate().Add(1.0))
            let pNew = p.Map (fun e -> if e<EPS then 0.0 else if e+EPS>1.0 then 1.0 else e)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            let zNew= s + (y-pNew)./w
            let approLik= -(((y-pNew))*((y-pNew)./w))/(double n)/2.0  
            let loss =approLik + (beta.[1..].Norm(1.0))*Lambda
            do printfn "Real Loglikelihood: %A \t Approx scaled Loglikelihood: %A\t loss: %A" (Loglik s y) approLik  loss
            beta,loss
        
        member val eps = 1e-16 with get,set
        member val maxIter = 100000 with get,set
        member val minIter = 10 with get,set
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        
        member this.Fit = this.Beta <- (update cyclicCoordinateDescentUpdate this.Beta this.eps 1 this.maxIter this.minIter 0.0)

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, ((normalize ((V x), mu, sigma)).InsertColumn(0, DenseVector.create 1 1.0)))
        
        member this.Predict (x:Matrix<double>) =
            predictWith1 (this.Beta, ((normalize ((M x), mu , sigma)).InsertColumn(0, DenseVector.create x.RowCount 1.0)))
            
