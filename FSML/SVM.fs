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
        let EPS = 1e-5
        let checkC = if C < 0.0 then raiseExcetion "C should be non-negative"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let maxPass=5
        let seed=1
        let rnd= System.Random(seed)

        member private this.f (i:int) = (normalizedX*normalizedX.Row(i)).* Y*this.Alpha + this.b

        member private this.LH (i:int,j:int,alphaI,alphaJ) =
            if Y.[i]<>Y.[j] then  max 0.0 (alphaJ-alphaI), min C (C+alphaJ-alphaI)
            else max 0.0 (alphaJ+alphaI-C), min C (alphaI+alphaJ)

        member private this.eta(i:int,j:int)=
            2.0*normalizedX.Row(i)*normalizedX.Row(j) - normalizedX.Row(i)*normalizedX.Row(i) - normalizedX.Row(j)*normalizedX.Row(j)

        member val Alpha = (DenseVector.zero (n)) with get,set
        member val b=0.0 with get,set

        member this.Predict (x:Vector<double>) = 
            ((normalizedX*(normalize ((V x), mu, sigma)).Transpose()).Transpose()*(Y.*this.Alpha)+this.b).Map (fun e-> if e>0.0 then 1.0 else -1.0)
        
        member this.Predict (x:Matrix<double>) =
            ((normalizedX*(normalize ((M x), mu, sigma)).Transpose()).Transpose()*(Y.*this.Alpha)+this.b).Map (fun e-> if e>0.0 then 1.0 else -1.0)

        member private this.E (i:int)= this.f i - Y.[i]

        member private this.checkCondition (i:int,Ei:double)=
            if (Ei < -tol && this.Alpha.[i]<C) || (Ei > tol && this.Alpha.[i]>0.0) then true else false
        
        member private this.update (iter:int)=
            printfn "%A" iter
            let mutable pass=0
            for i in [0..n-1] do
                pass<- pass + ( this.updateSingle i)
            if pass = 0 then iter+1 else 0
        
        member private this.updateAlphaI (i:int,j:int,alphaOldI,alphaOldJ,alphaJ,Ei,Ej)=
            let mutable alphaI = alphaOldI + Y.[i]*Y.[j]*(alphaOldJ-alphaJ)
            let b1 = this.b - Ei - Y.[i]*(alphaI-alphaOldI)*(normalizedX.Row(i)*normalizedX.Row(i)) -  Y.[j]*(alphaJ-alphaOldJ)*(normalizedX.Row(i)*normalizedX.Row(j))    
            let b2 = this.b - Ej - Y.[i]*(alphaI-alphaOldI)*(normalizedX.Row(i)*normalizedX.Row(j)) -  Y.[j]*(alphaJ-alphaOldJ)*(normalizedX.Row(j)*normalizedX.Row(j))
            this.Alpha.[i] <- alphaI
            this.Alpha.[j] <- alphaJ
            this.b <- (b1+b2)/2.0

        member private this.updateSingle (i:int)=
            let Ei= this.E i
            if this.checkCondition (i,Ei) then
                let mutable j = rnd.Next() % (n-1)
                if j >=i then j <- j+1 
                let Ej= this.E j
                let alphaOldI,alphaOldJ = (this.Alpha.[i],this.Alpha.[j])
                let L,H= this.LH (i,j,alphaOldI,alphaOldJ)
                if L=H then
                    0
                else
                    let eta = this.eta (i,j)
                    if eta=0.0 then
                        let Lobj=eta/2.0*L*L+(Y.[j]*(Ei-Ej)-eta*alphaOldJ)*L
                        let Hobj=eta/2.0*H*H+(Y.[j]*(Ei-Ej)-eta*alphaOldJ)*H
                        if Lobj> (Hobj+EPS) then
                            this.Alpha.[j] <- L
                            this.updateAlphaI (i,j,alphaOldI,alphaOldJ,L,Ei,Ej)
                            1
                        else 
                            if Lobj< (Hobj-EPS) then 
                                this.Alpha.[j] <- H 
                                this.updateAlphaI (i,j,alphaOldI,alphaOldJ,H,Ei,Ej)
                                1
                            else 0
                    else
                        let mutable alphaJ = (alphaOldJ-Y.[j]*(Ei - Ej)/eta)
                        if alphaJ > H then alphaJ <- H else if alphaJ<L then alphaJ<-L
                        if abs (alphaJ-alphaOldJ) < EPS*(alphaJ+alphaOldJ+EPS) then
                            0
                        else
                            this.Alpha.[j] <- alphaJ
                            this.updateAlphaI (i,j,alphaOldI,alphaOldJ,alphaJ,Ei,Ej)
                            1
            else 0
        
        member this.Fit () = 
            let mutable iter=0
            while iter<maxPass
                do iter <- (this.update iter)
