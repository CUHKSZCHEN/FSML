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
    
    let SoftThresholdingOperator (b:double, lambda:double) = 
        if b >0.0 then b-lambda else
            if b<0.0 then b+lambda else 0.0 
    
    let rec update f (parameter:Vector<double>) (eps:double) (iter:int) (maxIter:int) =
        if iter > maxIter then parameter
        else
            let param0,loss0 = f parameter
            let param1,loss1 = f param0
            if loss1-loss0 <  eps then parameter
            else update f param1 eps (iter+1) maxIter

    type LR (x:Matrix<double>,y:Vector<double>)=
       
        let xWith1= x.InsertColumn(0, DenseVector.create x.RowCount 1.0)
        member val eps = 1e-6 with get,set
        member val maxIter = 100 with get,set

        member this.Predict (x:Vector<double>) =
            let x1= DenseVector.create (x.Count+1) 1.0
            do (x1.SetSubVector(1,x.Count,x))
            [this.PredictWith1 (this.Beta,x1) ] |> DenseVector.ofSeq 

        member this.Predict (x:Matrix<double>) =
            this.PredictWith1 (this.Beta, x.InsertColumn(0, DenseVector.create x.RowCount 1.0))
   
        member private this.PredictWith1 (beta:Vector<double>, xWith1:Vector<double>)=xWith1 * beta 

        member private this.PredictWith1 (beta:Vector<double>, xWith1:Matrix<double>)=xWith1 * beta

        member val Beta = (DenseVector.zero (x.ColumnCount+1)) with get,set

        member this.Update (beta:Vector<double>)  =
            let p = (this.PredictWith1 (beta, xWith1) ).Negate().PointwiseExp().Add(1.0).DivideByThis(1.0)
            let loglik= Loglik p y
            do loglik |> printfn "Loglikelihood: %A"          
            let w= DiagonalMatrix.ofDiag (p .* p.Negate().Add(1.0))
            let z= xWith1 * beta + w.Inverse() *  (y-p)
            (xWith1.Transpose() * w *xWith1).Inverse() * xWith1.Transpose() * w*z,loglik

        member this.Fit = this.Beta <- (update this.Update this.Beta this.eps 0 this.maxIter)
