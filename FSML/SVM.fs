module SVM

    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Statistics

    type SVM (x:Matrix<double>,y:Vector<double>,C:double)=
        let n= y.Count
        let Y = y*2.0-1.0
        let tol = 1e-3
        let EPS = 1e-6
        let checkC = if C < 0.0 then raiseExcetion "C should be non-negative"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)

        let seed=1
        let rnd= System.Random(seed)

        member private this.f (i:int) = (normalizedX*normalizedX.Column(i)).* Y*this.Alpha

        member private this.LH (i:int,j:int,alphaI,alphaJ) =
            if Y.[i]=Y.[j] then  max 0.0 alphaJ-alphaI, min C C+alphaJ-alphaI
            else max 0.0 alphaJ+alphaI-C, min C alphaI+alphaJ

        member private this.eta(i:int,j:int)=
            2.0*normalizedX.Column(i)*normalizedX.Column(j) - normalizedX.Column(i)*normalizedX.Column(i)-normalizedX.Column(j)*normalizedX.Column(j)

        member val Alpha = (DenseVector.zero (n)) with get,set
        member val b=0.0 with get,set

        member this.Predict (x:Vector<double>) = 
            predictWith1 (this.Alpha, (x.ToRowMatrix().InsertColumn(0, DenseVector.create x.Count 1.0)))
        
        member this.Predict (x:Matrix<double>) =
            predictWith1 (this.Alpha, (x.InsertColumn(0, DenseVector.create x.RowCount 1.0)))

        member private this.E (i:int)= this.f i - Y.At(i)

        member this.checkCondition (i:int)=
            if (y.[i]*(this.E i) < -tol && this.Alpha.[i]<C) || (y.[i]*(this.E i) > tol && this.Alpha.[i]>0.0) then false else true
        
        member this.update (i:int)=
            let mutable j = rnd.Next() % (n-1)
            if j >=i then j <- j+1 
            let Ej= this.E j
            let alphaOldI,alphaOldJ = (this.Alpha.[i],this.Alpha.[j])
            let L,H= this.LH (i,j,alphaOldI,alphaOldJ)
            if L=H then
                this.update i+1
            else
                let eta = this.eta (i,j)
                if eta>0.0 then
                    this.update i+1
                else
                    let Ei= this.E i
                    let mutable alphaJ = (alphaOldJ-Y.[j]*(Ei - Ej)/eta)
                    if alphaJ > H then alphaJ <- H else if alphaJ<L then alphaJ<-L
                    if abs (alphaJ-alphaOldJ) < EPS then
                        this.update i+1
                    else
                        let mutable alphaI = alphaOldI + Y.[i]*Y.[j]*(alphaOldJ-alphaJ)
                        let b1 = this.b - Ei - Y.[i]*(alphaI-alphaOldI)*normalizedX.Column(i)*normalizedX.Column(i) -  Y.[j]*(alphaJ-alphaOldJ)*normalizedX.Column(i)*normalizedX.Column(j)
                        let mutable b = b1
                        let b2 = this.b - Ej - Y.[i]*(alphaI-alphaOldI)*normalizedX.Column(i)*normalizedX.Column(j) -  Y.[j]*(alphaJ-alphaOldJ)*normalizedX.Column(j)*normalizedX.Column(j)
                        if alphaI>0.0 && alphaI<C then b <- b1 else if  alphaJ>0.0 && alphaJ<C then b<- b2 else b <- (b1+b2)/2.0
                        this.b<-b
                        this.Alpha.[i] <- alphaI
                        this.Alpha.[j] <- alphaJ
                        this.update i+1
