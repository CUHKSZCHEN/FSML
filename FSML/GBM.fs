module GBM

    open DataTypes
    open Utilities
    open Tree
    open System.Threading.Tasks
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type GBM (xTrain:Matrix<double>,yTrain:Vector<double>,family:string,maxTrees:int,depth:int,eta:double,lambda:double,gamma:double,sub_sample:double,sub_feature:double, ?seed:int)=
        let x,y = xTrain,yTrain.AsArray()
        let n = y.Length

        let gh = match family with
                 | "gaussian" -> gh_lm
                 | "Gaussian" -> gh_lm
                 | "binomial" -> gh_lr
                 | _ -> raiseExcetion "Please choose either Gaussian or binomial!"


        let checkDepth = if depth < 1 then raiseExcetion "the minimum depth is 1"
        let checkMaxTrees = if maxTrees < 1 then raiseExcetion "the minimum number of trees is 1"

        let checkEta = if eta < 0.0 then raiseExcetion "please choose a positive learning rate eta"
        let checkLambda = if lambda < 0.0 then raiseExcetion "please choose a positive lambda"
        let checkGamma = if gamma < 0.0 then raiseExcetion "please choose a positive gamm"
        let checkSub_sample = if sub_sample <= 0.0 || sub_sample>1.0 then raiseExcetion "please choose sub_sample from 0.0 to 1.0"
        let checkSub_feature = if sub_feature <= 0.0 || sub_sample>1.0 then raiseExcetion "please choose sub_feature from 0.0 to 1.0"

        let seed= defaultArg seed 1
        let rnd= System.Random(seed)

        let forest = Array.create maxTrees Empty

        let yTilde = Array.create yTrain.Count (y |> Array.average)
        let gTilde,hTilde = [for i in 0..(n-1) -> gh y.[i] yTilde.[i]] |> List.unzip |> fun (g,h) -> Array.ofList(g) ,Array.ofList(h)
        let xValueSorted,xIndexSorted = x.EnumerateColumns() |> Seq.map (fun col -> col.ToArray() |> Array.mapi (fun e t -> (t,e))  |> Array.sort |> Array.toList |> List.unzip |> fun(e,i)-> Array.ofList(e),Array.ofList(i) ) |> List.ofSeq |> List.unzip |> fun(e,i) -> Array.ofList(e),Array.ofList(i)

        let predictMatch (link:Vector<double>) (value:string)= 
            match value,family with
            | "link",_ -> link
            | "response", "gaussian" -> link
            | "response", "Gaussian" -> link
            | "response", "binomial" -> logistic(link)
            | _ -> raiseExcetion "predict either link or response"      

        member this.Fit() =
            for i in [0..maxTrees-1] do
                let xInNode = Array.init n (fun _ -> rnd.NextDouble() <= sub_sample)
                let fInTree = Array.init x.ColumnCount (fun _ -> rnd.NextDouble() <= sub_feature)
                forest.[i] <- growTree Empty fInTree xInNode gh depth xValueSorted xIndexSorted y yTilde gTilde hTilde eta lambda gamma

        member this.CVFit (metric:string, nTrees:int ,?nFolds:int, ?earlyStopRounds:int)=
            let nFolds = defaultArg nFolds 5
            let cvForests = Array2D.create nTrees nFolds Empty

            let foldArray = Array.init n (fun _ -> rnd.Next() % nFolds )
            let nRounds = defaultArg earlyStopRounds 10
            let metricArray = Array2D.create nTrees nFolds 0.0

            let metricFun = match metric with
                            | "rmse" -> RMSE
                            | "RMSE" -> RMSE
                            | "auc" -> AUC
                            | "AUC" -> AUC
                            | "logloss" -> logloss
                            | _ -> raiseExcetion "Please choose either RMSE, AUC or logloss"

            let yInFold = [|0..nFolds-1|] |> Array.map (fun f-> (y |> Array.indexed |> Array.filter (fun (index, _) -> foldArray.[index]=f) |> Array.map (fun (_, e) -> e)) |> DenseVector.ofArray)
            let yHatInFold = [| for f in 0..nFolds-1 do yield (Array.create (yInFold.[f]).Count 0.0 |> DenseVector.ofArray) |]  
            for i in [0..nTrees-1] do
                for f in [0..nFolds-1] do
                    let xInNode = Array.init n (fun index -> (rnd.NextDouble() <= sub_sample) && (foldArray.[index]<>f))
                    let fInTree = Array.init x.ColumnCount (fun _ -> rnd.NextDouble() <= sub_feature)

                    cvForests.[i,f] <- growTree Empty fInTree xInNode gh depth xValueSorted xIndexSorted y yTilde gTilde hTilde eta lambda gamma
                    yHatInFold.[f] <- yHatInFold.[f] + this.Predict(x, cvForests.[i,f],foldArray, f) 
                    metricArray.[i,f] <- metricFun yInFold.[f] (predictMatch yHatInFold.[f] "response")
                let mu= metricArray.[i,*] |> Array.average
                let sd= sqrt((metricArray.[i,*] |> Array.map (fun e -> e*e) |> Array.average) - mu*mu)
                printfn "Iter: %5d \t CV %s \t %10.15f \t %10.15f" i metric mu sd
        
        member private this.Predict(x:Matrix<double>, oneTree: tree<node>, foldArray: int[], fold: int) = 
            DenseVector.ofArray( predictTreeforMatrixInFold oneTree x foldArray fold)

        member this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = DenseVector.ofArray([|predictForestforVector forest x|])
            predictMatch link value

        member this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = DenseVector.ofArray( predictForestforMatrix forest x)
            predictMatch link value

        
