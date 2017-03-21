module SVM

    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Statistics

    type SVM (x:Matrix<double>,y:Vector<double>,C:double,K:string,var:double)=
        let n= y.Count
        let Y = y*2.0-1.0
        let tol = 1e-3
        let EPS = 1e-5
        let checkC = if C < 0.0 then raiseException "C should be non-negative"
        let mu,sigma = x|> getNormalizeParameter
        let normalizedX=normalize ((M x), mu , sigma)
        let maxPass=5
        let seed=1
        let rnd= System.Random(seed)
        let innerProduct=normalizedX*normalizedX.Transpose()
        let linearKernel (i,j)= innerProduct.[i,j]
        let rbfKernel (i,j) = exp (-(innerProduct.[i,i]+innerProduct.[j,j]-2.0*innerProduct.[i,j])/2.0/var)
        let KernelFunc = match K with
                            | InvariantEqual "linear" -> linearKernel
                            | InvariantEqual "rbf" -> rbfKernel
                            |_ -> raiseException "please choose either linear or rbf kernel"
        
        let KernelMatrix = DenseMatrix.create n n 0.0

        let initKernal = [0..n-1] |> List.iter (fun i ->
            [0..n-1] |> List.iter (fun j ->
                KernelMatrix.[i,j] <- KernelFunc (i,j)
                )
            )

        new (x,y,C) = SVM(x,y,C,"linear",1.0)

        member private this.f (i:int) = (KernelMatrix.Column(i)).* Y*this.Alpha + this.b

        member private this.LH (i:int,j:int,alphaI,alphaJ) =
            if Y.[i]<>Y.[j] then  max 0.0 (alphaJ-alphaI), min C (C+alphaJ-alphaI)
            else max 0.0 (alphaJ+alphaI-C), min C (alphaI+alphaJ)

        member private this.eta(i:int,j:int)=
            2.0*KernelMatrix.[i,j] - KernelMatrix.[i,i] - KernelMatrix.[j,j]

        member val Alpha = (DenseVector.zero (n)) with get,set
        member val b=0.0 with get,set
        member val var=1.0 with get

        member this.Predict (x:Vector<double>) = 
            this.Predict (x.ToRowMatrix())

        member this.Predict (x:Matrix<double>) =
            let testMatrix = DenseMatrix.create x.RowCount n 0.0
            let testNormalized=normalize ((M x), mu, sigma)
			let kernelCalculation i j = match K with
                        | InvariantEqual "linear" -> normalizedX.Row(j)*testNormalized.Row(i)
                        | InvariantEqual "rbf" -> exp (-(normalizedX.Row(j)*normalizedX.Row(j)+testNormalized.Row(i)*testNormalized.Row(i)-2.0*testNormalized.Row(i)*normalizedX.Row(j))/2.0/var)
                        | _ -> raiseException "please choose either linear or rbf kernel"
            [0..x.RowCount-1] |> List.iter (fun i ->
                [0..n-1] |> List.iter (fun j ->
                        testMatrix.[i,j] <- kernelCalculation i j
                )
            )
            (testMatrix*(Y.*this.Alpha)+this.b).Map (fun e-> if e>0.0 then 1.0 else -1.0)
            
        member private this.E (i:int)= this.f i - Y.[i]

        member private this.checkCondition (i:int,Ei:double)=
            if (Ei < -tol && this.Alpha.[i]<C) || (Ei > tol && this.Alpha.[i]>0.0) then true else false
        
        member private this.update (iter:int)=
            //printfn "%A" iter
            let mutable pass=0
            [0..n-1] |> List.iter (fun i -> pass<- pass + ( this.updateSingle i))
            if pass = 0 then iter+1 else 0
        
        member private this.updateAlphaI (i:int,j:int,alphaOldI,alphaOldJ,alphaJ,Ei,Ej)=
            let mutable alphaI = alphaOldI + Y.[i]*Y.[j]*(alphaOldJ-alphaJ)
            let b1 = this.b - Ei - Y.[i]*(alphaI-alphaOldI)*(KernelMatrix .[i,i]) -  Y.[j]*(alphaJ-alphaOldJ)*(KernelMatrix .[i,j])    
            let b2 = this.b - Ej - Y.[i]*(alphaI-alphaOldI)*(KernelMatrix .[i,j]) -  Y.[j]*(alphaJ-alphaOldJ)*(KernelMatrix .[j,j])
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
                            this.updateAlphaI (i,j,alphaOldI,alphaOldJ,L,Ei,Ej)
                            1
                        else 
                            if Lobj< (Hobj-EPS) then 
                                this.updateAlphaI (i,j,alphaOldI,alphaOldJ,H,Ei,Ej)
                                1
                            else 0
                    else
                        let mutable alphaJ = (alphaOldJ-Y.[j]*(Ei - Ej)/eta)
                        if alphaJ > H then alphaJ <- H else if alphaJ<L then alphaJ<-L
                        if abs (alphaJ-alphaOldJ) < EPS*(alphaJ+alphaOldJ+EPS) then
                            0
                        else
                            this.updateAlphaI (i,j,alphaOldI,alphaOldJ,alphaJ,Ei,Ej)
                            1
            else 0
        
        member this.Fit () = 
            let mutable iter=0
            while iter<maxPass
                do iter <- (this.update iter)
