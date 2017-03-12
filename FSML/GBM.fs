module GBM

    open DataTypes
    open Utilities
    open Tree
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type GBM (xTrain:Matrix<double>,yTrain:Vector<double>,depth:int,eta:double,lambda:double,gamma:double,sub_sample:double,sub_feature:double)=
        let x,y = xTrain,yTrain.AsArray()
        let n = y.Length
        //let tol = 1e-3
        let EPS = 1e-5

        let checkDepth = if depth < 1 then raiseExcetion "the minimum depth is 1"
        let checkEta = if eta < 0.0 then raiseExcetion "please choose a positive learning rate eta"
        let checkLambda = if lambda < 0.0 then raiseExcetion "please choose a positive lambda"
        let checkGamma = if gamma < 0.0 then raiseExcetion "please choose a positive gamm"
        let checkSub_sample = if sub_sample <= 0.0 || sub_sample>1.0 then raiseExcetion "please choose sub_sample from 0.0 to 1.0"
        let checkSub_feature = if sub_feature <= 0.0 || sub_sample>1.0 then raiseExcetion "please choose sub_feature from 0.0 to 1.0"

        let seed=1
        let rnd= System.Random(seed)

        let forest = Array.empty<tree<node>>

        let yTilde = Array.zeroCreate yTrain.Count
        let gTilde,hTilde = [for i in 0..(n-1) -> gh_lm y.[i] yTilde.[i]] |> List.unzip |> fun (g,h) -> Array.ofList(g) ,Array.ofList(h)
        let xValueSorted,xIndexSorted = x.EnumerateColumns() |> Seq.map (fun col -> col.ToArray() |> Array.mapi (fun e t -> (t,e))  |> Array.sort |> Array.toList |> List.unzip |> fun(e,i)-> Array.ofList(e),Array.ofList(i) ) |> List.ofSeq |> List.unzip |> fun(e,i) -> Array.ofList(e),Array.ofList(i)
        let featureIndex = [|0..x.ColumnCount-1|]
        let fInTree = Array.create x.ColumnCount true
        let xInNode = Array.create n true
        let ttt= growTree Empty  fInTree  xInNode depth xValueSorted xIndexSorted y yTilde gTilde hTilde eta lambda gamma
        member this.tree =  ttt
        //let a = buildTree(featureIndex,xValueSorted,xIndexSorted,y,yTilde,(gTilde,hTilde),depth,eta,lambda,gamma,sub_sample,sub_feature)
