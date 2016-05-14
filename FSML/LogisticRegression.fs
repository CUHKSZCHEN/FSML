module LogisticRegression
    
    open DataTypes
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type LR (x:Matrix<double>,y:Vector<double>)=
       
        member val EPS=1e-6 with get,set

        member this.Predict (x:VectorOrMatrix) = match x with
                | V v -> let v1= Vector<double>.One v.Count+1 
                         [this.PredictSingle this.Beta v. ] |> DenseVector.ofSeq 
                | M m ->  this.PredictMultiple this.Beta (m.InsertColumn(0, (seq{for i in [1..m.RowCount] -> 1.0} |> DenseVector.ofSeq)))

        member private this.XWith1= x.InsertColumn(0, (seq{for i in [1..y.Count] -> 1.0} |> DenseVector.ofSeq))

        member private this.PredictWith1 (beta:Vector<double>) (xWith1:VectorOrMatrix) = match xWith1 with
                | V v ->  [this.PredictSingle beta v] |> DenseVector.ofSeq 
                | M m ->  this.PredictMultiple beta m
        
        member private this.PredictSingle (beta:Vector<double>) (xWith1:Vector<double>)=xWith1 * beta 

        member private this.PredictMultiple (beta:Vector<double>) (xWith1:Matrix<double>)=xWith1 * beta

        member val Beta =seq{for i in [1..x.ColumnCount+1] -> 0.0} |> DenseVector.ofSeq with get,set

        member private this.Update beta  =

            let o = seq{for i in [1..y.Count] -> 1.0} |> DenseVector.ofSeq
            let pScore=this.PredictMultiple beta this.XWith1
            let p = o./(o+ (-1. *pScore).PointwiseExp())
            let loglikNew=this.Loglikelihood p y
            
            let w= DiagonalMatrix.ofDiag (p .* (o - p))
            let z= this.XWith1 * beta + w.Inverse() *  (y-p)
            let betaNew= (this.XWith1.Transpose() * w *this.XWith1).Inverse() * this.XWith1.Transpose() * w*z          
            betaNew

        member this.Loglikelihood (p:Vector<double>) (y:Vector<double>)=
            let o = seq{for i in [1..p.Count] -> 1.0} |> DenseVector.ofSeq
            y*p.PointwiseLog() + (o-y)*(o-p).PointwiseLog()

        member this.Fit k= for i in [1..k] do  this.Beta <- (this.Update this.Beta)
                            
 
