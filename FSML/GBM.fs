module GBM

    open DataTypes
    open Utilities
    open Tree
    open System.Threading.Tasks
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type GBM (xTrain:Matrix<double>,yTrain:Vector<double>,family:string,depth:int,eta:double,lambda:double,gamma:double,sub_sample:double,sub_feature:double, ?seed:int)=
        let x,y = xTrain,yTrain.AsArray()
        let n = y.Length

        let gh = match family with
                 | InvariantEqual "Gaussian" -> gh_lm
                 | InvariantEqual "binomial" -> gh_lr
                 | _ -> raiseException "Please choose either Gaussian or binomial!"


        let checkDepth = if depth < 1 then raiseExcetion "the minimum depth is 1"

        let checkEta = if eta < 0.0 then raiseExcetion "please choose a positive learning rate eta"
        let checkLambda = if lambda < 0.0 then raiseExcetion "please choose a positive lambda"
        let checkGamma = if gamma < 0.0 then raiseExcetion "please choose a positive gamm"
        let checkSub_sample = if sub_sample <= 0.0 || sub_sample>1.0 then raiseExcetion "please choose sub_sample from 0.0 to 1.0"
        let checkSub_feature = if sub_feature <= 0.0 || sub_sample>1.0 then raiseExcetion "please choose sub_feature from 0.0 to 1.0"

        let seed= defaultArg seed 1
        let rnd= System.Random(seed)

        
        let w0 = y |> Array.average
        let yTilde = Array.create yTrain.Count w0

        let gTilde,hTilde = [for i in 0..(n-1) -> gh y.[i] yTilde.[i]] |> List.unzip |> fun (g,h) -> Array.ofList(g) ,Array.ofList(h)
        let xValueSorted,xIndexSorted = x.EnumerateColumns() |> Seq.map (fun col -> col.ToArray() |> Array.mapi (fun e t -> (t,e))  |> Array.sort |> Array.toList |> List.unzip |> fun(e,i)-> Array.ofList(e),Array.ofList(i) ) |> List.ofSeq |> List.unzip |> fun(e,i) -> Array.ofList(e),Array.ofList(i)

        let predictMatch (link:Vector<double>) (value:string)= 
            match value,family with
            | "link",_ -> link
            | "response", "gaussian" -> link
            | "response", "Gaussian" -> link
            | "response", "binomial" -> logistic(link)
            | _ -> raiseExcetion "predict either link or response"      
        
        member val Forest = Array.empty with get,set

        member this.Fit(maxTrees:int) =
            let checkMaxTrees = if maxTrees < 1 then raiseExcetion "the minimum number of trees is 1"
            this.Forest <- Array.create maxTrees Empty
            let wTilde = Array.create yTrain.Count 0.0

            for i in [0..maxTrees-1] do
                printfn "tree %A" i
                let nodeId = [|0|]

                let xInNode = Array.init n (fun _ -> rnd.NextDouble() <= sub_sample)
                let fInTree = Array.init x.ColumnCount (fun _ -> rnd.NextDouble() <= sub_feature)
                this.Forest.[i] <- growTree Empty nodeId fInTree xInNode depth xValueSorted xIndexSorted y wTilde gTilde hTilde eta lambda gamma
                for j in [0..n-1] do
                    yTilde.[j] <- yTilde.[j] + wTilde.[j]
                    let gt,ht= gh y.[j] yTilde.[j]
                    gTilde.[j] <- gt 
                    hTilde.[j] <- ht 
                let yt:Vector<double> = (this.Predict xTrain )
                let delta:Vector<double>= (yt- DenseVector.ofArray(yTilde))
                printfn "delta %A \t %A" (delta.AbsoluteMaximum()) (delta.AbsoluteMaximumIndex())
                printfn "Iter: %5d \t %10.15f"  i  (RMSE yTrain (DenseVector.ofArray(yTilde)))

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

            let s = Array.zip foldArray y |> Seq.ofArray |> Seq.sortBy fst 
            let ySum = y |> Array.sum
            let foldSums = s|> Seq.groupBy fst |> Seq.toList |> List.map (fun (_,s) -> Seq.sumBy snd s ) |> Array.ofList |> Array.map (fun e -> ySum-e )
            let w0s = [| for f in 0..nFolds-1 do yield foldSums.[f]/(double (y.Length - (yInFold.[f]).Count))|]
            
            let yHatInFold = [| for f in 0..nFolds-1 do yield (Array.create (yInFold.[f]).Count w0s.[f] |> DenseVector.ofArray) |]

            let wTildeCVs = [| for f in 0..nFolds-1 do yield (Array.create yTrain.Count 0.0)|]
            let yTildeCVs = [|for f in 0..nFolds-1 do yield (foldArray |> Array.map (fun e -> if e <> f then w0s.[f] else 0.0 ))|]
            let gTildeCVs,hTildeCVs = [| for f in 0..nFolds-1 do yield ([for i in 0..(n-1) -> gh y.[i] yTildeCVs.[f].[i]] |> List.unzip |> fun (g,h) -> Array.ofList(g) ,Array.ofList(h) )|] |> Array.unzip
            
            for i in [0..nTrees-1] do
                for f in [0..nFolds-1] do
                    let nodeId = [|0|]
                    let xInNode = Array.init n (fun index -> (rnd.NextDouble() <= sub_sample) && (foldArray.[index]<>f))
                    let fInTree = Array.init x.ColumnCount (fun _ -> rnd.NextDouble() <= sub_feature)
                    cvForests.[i,f] <- growTree Empty nodeId fInTree xInNode depth xValueSorted xIndexSorted y wTildeCVs.[f] gTildeCVs.[f] hTildeCVs.[f] eta lambda gamma

                    for j in [0..n-1] do
                        if foldArray.[j] <>f then
                            yTildeCVs.[f].[j] <- yTildeCVs.[f].[j] + wTildeCVs.[f].[j]
                            let gt,ht= gh y.[j] yTildeCVs.[f].[j]
                            gTildeCVs.[f].[j] <- gt 
                            hTildeCVs.[f].[j] <- ht 
                    
                    yHatInFold.[f] <- yHatInFold.[f] + this.Predict(x, cvForests.[i,f],foldArray, f) 
                    metricArray.[i,f] <- metricFun yInFold.[f] (predictMatch yHatInFold.[f] "response")

                    //let yt:Vector<double> = (this.Predict xTrain )
                    //let delta:Vector<double>= (yt- DenseVector.ofArray(yTilde))
                    //printfn "delta %A \t %A" (delta.AbsoluteMaximum()) (delta.AbsoluteMaximumIndex())
                    //printfn "Iter: %5d \t %10.15f"  i  (RMSE yTrain (DenseVector.ofArray(yTilde)))

                let mu= metricArray.[i,*] |> Array.average
                let sd= sqrt((metricArray.[i,*] |> Array.map (fun e -> e*e) |> Array.average) - mu*mu)
                printfn "Iter: %5d \t CV %s \t %10.15f \t %10.15f" i metric mu sd
        
        member private this.Predict(x:Matrix<double>, oneTree: tree<node>, foldArray: int[], fold: int) = 
            DenseVector.ofArray( predictTreeforMatrixInFold oneTree x foldArray fold)

        member this.Predict(x:Vector<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = w0+ DenseVector.ofArray([|predictForestforVector this.Forest x|])
            predictMatch link value

        member this.Predict(x:Matrix<double>,?value:string) = 
            let value = defaultArg value "link"
            let link = w0 + DenseVector.ofArray( predictForestforMatrix this.Forest x)
            predictMatch link value
