module Utilities

    open MathNet.Numerics.Statistics
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open DataTypes
    open System
    
    let (|InvariantEqual|_|) (str:string) arg =
        if String.Compare(str,arg,StringComparison.OrdinalIgnoreCase) =0 then Some() else None
    
    let shuffle (n:int) (rnd:Random)=
        let arr = [|0..n-1|]
        for i = n-1 downto 1 do
            let j = rnd.Next() % (i+1)
            let temp = arr.[i]
            arr.[i] <- arr.[j]
            arr.[j] <- temp
        arr
    
    let predictWith1 (beta:Vector<double>, xWith1:Matrix<double>)= xWith1 * beta

    let normalize (x:VectorOrMatrix,mu:Vector<double>,sd:Vector<double>) =
        match x with
        | V v -> (v.Add(mu.Negate())./sd).ToRowMatrix()
        | M m -> m.ToRowArrays() |> Array.map (fun row -> ((row |> DenseVector.ofArray).Add(mu.Negate())./(sd)).ToArray() ) |> DenseMatrix.ofRowArrays

    let predictLinear (beta:Vector<double>, x:VectorOrMatrix) = 
        match x with 
        | V x -> [x*beta] |> DenseVector.ofList
        | M x -> x* beta
    
    let predictLinearScale (beta:Vector<double>, x:VectorOrMatrix,mu:Vector<double>,sigma:Vector<double>) = 
        match x with 
        | V x -> normalize ((V x), mu, sigma)*beta
        | M x -> normalize ((M x), mu, sigma)*beta

    let logistic (x:Vector<double>) = 1.0/((-x).PointwiseExp() + 1.0)

    let RMSE (y:Vector<double>) (p:Vector<double>)=
        sqrt(((y-p) * (y-p))/(double y.Count))

    let AUC (y:Vector<double>) (score:Vector<double>)=
        let n1= y.Sum()
        let n0= (double) y.Count-n1
        let sortedScore = Array.zip (score.ToArray()) (y.Negate().ToArray()) |> Array.sortBy (fun e-> e|> snd) |> Array.map fst
        let rank= (sortedScore |> DenseVector.ofArray) .Ranks RankDefinition.Average
        ((rank.[0..(int)n1-1] |> Array.sum )- n1* (n1+1.0)/2.0)/n0/n1

    let Loglik (s:Vector<double>) (y:Vector<double>)=
        y*s - s.PointwiseExp().Add(1.0).PointwiseLog().Sum()

    let logloss (y:Vector<double>) (p:Vector<double>)=
        let EPSLow=1e-15
        let EPSHigh=1.0-EPSLow 
        let pNew=p.Map (fun e -> if e<EPSLow then EPSLow else if e>EPSHigh then EPSHigh else e)
        -(y*pNew.PointwiseLog() + (y.Negate().Add(1.0))*pNew.Negate().Add(1.0).PointwiseLog()) /(double p.Count)

    let getNormalizeParameter (x:Matrix<double>) =
        x.ColumnSums().Divide(double x.RowCount),x.ToColumnArrays() |> Array.map (fun col -> sqrt(col.Variance() *((double col.Length)-1.0)/(double col.Length) )) |> DenseVector.ofArray


    let QRUpdate (x:Matrix<double>) (y:Vector<double>)=
        let qr=x.QR()
        qr.R.Solve(qr.Q.Transpose()*y)

    let SVDUpdate (x:Matrix<double>) (y:Vector<double>)=
        let svd=x.Svd()
        let D= svd.W
        let W = DiagonalMatrix.ofDiagArray2 D.RowCount D.ColumnCount  (D.Diagonal().ToArray() |> Array.map (fun e -> if e < 1e-9 then 0.0 else 1.0/e) )
        svd.VT.Transpose() * W.Transpose() * svd.U.Transpose() *y

    let WeightedQRUpdate (x:Matrix<double>) (y:Vector<double>) (w:Vector<double>)=
        let wSqrt= w.PointwiseSqrt()
        let xTilde = DiagonalMatrix.ofDiag(wSqrt) * x
        let yTilde = wSqrt.*y
        QRUpdate xTilde yTilde

    let WeightedSVDUpdate (x:Matrix<double>) (y:Vector<double>) (w:Vector<double>)=
        let wSqrt= w.PointwiseSqrt()
        let xTilde = DiagonalMatrix.ofDiag(wSqrt) * x
        let yTilde = wSqrt.*y
        SVDUpdate xTilde yTilde

    let STO (z) (gamma) =
        if gamma >= abs(z) then 0.0
        else if z>0.0 then z-gamma
        else z+gamma

    let rec update f (parameter:Vector<double>) (eps:double) (iter:int) (maxIter:int) (minIter:int) (lossOld:double) =
        if iter >= maxIter then parameter
        else
            let paramNew,lossNew = f parameter
            if lossOld - lossNew <  eps && iter >= minIter then parameter
            else update f paramNew eps (iter+1) maxIter minIter lossOld
