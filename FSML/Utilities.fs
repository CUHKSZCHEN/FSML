module Utilities

    open MathNet.Numerics.Statistics
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open DataTypes

    let predictWith1 (beta:Vector<double>, xWith1:Matrix<double>)= xWith1 * beta

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

    let getNormalizeParameter (x:Matrix<double>) =
        x.ColumnSums().Divide(double x.RowCount),x.ToColumnArrays() |> Array.map (fun col -> col.StandardDeviation() ) |> DenseVector.ofArray

    let normalize (x:VectorOrMatrix,mu:Vector<double>,sd:Vector<double>) =
        match x with
        | V v -> (v.Add(mu.Negate())./sd).ToRowMatrix()
        | M m -> m.ToRowArrays() |> Array.map (fun row -> ((row |> DenseVector.ofArray).Add(mu.Negate())./(sd)).ToArray() ) |> DenseMatrix.ofRowArrays
    

