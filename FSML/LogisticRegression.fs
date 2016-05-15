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

    let Loglik (p:Vector<double>) (y:Vector<double>)=
            y*p.PointwiseLog() + (y.Negate().Add(1.0))*p.Negate().Add(1.0).PointwiseLog()
        
    let getNormalizeParameter (x:Matrix<double>) =
        x.ColumnSums().Divide(double x.RowCount),x.ToColumnArrays() |> Array.map (fun col -> col.StandardDeviation() ) |> DenseVector.ofArray

    let normalize (x:VectorOrMatrix,mu:Vector<double>,sd:Vector<double>) =
        match x with
        | V v -> (v.Add(mu.Negate())./sd).ToRowMatrix()
        | M m -> m.ToRowArrays() |> Array.map (fun row -> ((row |> DenseVector.ofArray).Add(mu.Negate())./(sd)).ToArray() ) |> DenseMatrix.ofRowArrays
    
    let predictWith1 (beta, xWith1:Matrix<double>)= xWith1 * beta

    let reWeightedUpdate (beta:Vector<double>)  (xWith1:Matrix<double>) (y:Vector<double>) =
            let p = (predictWith1 (beta, xWith1) ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik= Loglik p y
            do loglik |> printfn "Loglikelihood: %A"          
            let w= DiagonalMatrix.ofDiag (p .* p.Negate().Add(1.0))
            let z= xWith1 * beta + w.Inverse() *  (y-p)
            (xWith1.Transpose() * w *xWith1).Inverse() * xWith1.Transpose() * w*z,loglik

    let rec update f (parameter:Vector<double>) (x:Matrix<double>) (y:Vector<double>) (eps:double) (iter:int) (maxIter:int) =
        if iter > maxIter then parameter
        else
            let param0,loss0 = f parameter x y
            let param1,loss1 = f param0 x y
            if loss1-loss0 <  eps then parameter
            else update f param1 x y eps (iter+1) maxIter

    type LR (x:Matrix<double>,y:Vector<double>)=
       
        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Beta, (x.ToRowMatrix().InsertColumn(0, DenseVector.create x.Count 1.0)))
        
        member this.Predict (x:Matrix<double>) =
            predictWith1 (this.Beta, (x.InsertColumn(0, DenseVector.create x.RowCount 1.0)))

        member val eps = 1e-6 with get,set
        member val maxIter = 100 with get,set
   
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        member this.Update (beta:Vector<double>)  =
            let p = (predictWith1 (beta, xWith1) ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik= Loglik p y
            do loglik |> printfn "Loglikelihood: %A"          
            let w= DiagonalMatrix.ofDiag (p .* p.Negate().Add(1.0))
            let z= xWith1 * beta + w.Inverse() *  (y-p)
            (xWith1.Transpose() * w *xWith1).Inverse() * xWith1.Transpose() * w*z,loglik

        member this.Fit = this.Beta <- (update reWeightedUpdate this.Beta xWith1 y this.eps 0 this.maxIter)
    
    //https://core.ac.uk/download/files/153/6287975.pdf
   type LASSO (x:Matrix<double>,y:Vector<double>,lambda)=
        let n= y.Count
        let Lambda=lambda 
        let EPS = 1e-5
        let checkLambda = if Lambda < 0.0 then raiseExcetion "lambda should be positive"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let normalizedXWith1=normalizedX.InsertColumn(0, DenseVector.create x.RowCount 1.0)

        let coordinateDescent (betaNew:Vector<double>) (co:int)=
            let s=predictWith1 (betaNew, normalizedXWith1)
            let p=(s ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let w=p .* p.Negate().Add(1.0)
            let pNew = p.Map (fun e -> if abs(e)<EPS then 0.0 else if abs(e)+EPS>1.0 then 1.0 else e)
            let wNew = Vector.map2 (fun e k -> if abs(e)<EPS then EPS else if abs(e)+EPS>1.0 then EPS else k) p w
            let zNew= normalizedXWith1 * betaNew + (DiagonalMatrix.ofDiag wNew).Inverse() *  (y-pNew)
            let denominator=wNew*(normalizedXWith1.Column(co).PointwisePower(2.0))
            let a = ((zNew - s + normalizedXWith1.Column(co).Multiply(betaNew.At(co)) ).*wNew.*normalizedXWith1.Column(co)).Sum()
            let b = double n*Lambda
            if a>b then a-b/denominator else 
                if a + b < 0.0 then a+b/denominator else 0.0


        member val eps = 1e-6 with get,set
        member val maxIter = 100 with get,set
   
        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set
        member this.CyclicCoordinateDescentUpdate (beta:Vector<double>)=
            let betaNew,loglik = reWeightedUpdate beta normalizedXWith1 y
            for i in [1..beta.Count-1] do 
                betaNew.Item(i) <- (coordinateDescent betaNew i)
            betaNew
        
        member this.Fit (k:int) = for i in [1..k] do 
                                    this.Beta <- (this.CyclicCoordinateDescentUpdate this.Beta)

