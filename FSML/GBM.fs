module GBM

    open DataTypes
    open Utilities
    open Tree
    open System.Threading.Tasks
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open System

    type GBM (xTrain:Matrix<double>,yTrain:Vector<double>,family:string,depth:int,eta:double,lambda:double,gamma:double,sub_sample:double,sub_feature:double)=
        let x,y = xTrain,yTrain.AsArray()
        let n = y.Length

        let gh = match family with
                 | InvariantEqual "Gaussian" -> gh_lm
                 | InvariantEqual "binomial" -> gh_lr
                 | _ -> raiseException "Please choose either Gaussian or binomial!"


        let checkDepth = if depth < 1 then raiseException "the minimum depth is 1"

        let checkEta = if eta < 0.0 then raiseException "please choose a positive learning rate eta"
        let checkLambda = if lambda < 0.0 then raiseException "please choose a positive lambda"
        let checkGamma = if gamma < 0.0 then raiseException "please choose a positive gamm"
        let checkSub_sample = if sub_sample <= 0.0 || sub_sample>1.0 then raiseException "please choose sub_sample from 0.0 to 1.0"
        let checkSub_feature = if sub_feature <= 0.0 || sub_sample>1.0 then raiseException "please choose sub_feature from 0.0 to 1.0"

        let nf = min (max 1 (int (sub_feature * double x.ColumnCount))) x.ColumnCount
        
        let w0 = y |> Array.average
        let yTilde = Array.create yTrain.Count w0

        let gTilde,hTilde = [for i in 0..(n-1) -> gh y.[i] yTilde.[i]] |> List.unzip |> fun (g,h) -> Array.ofList(g) ,Array.ofList(h)
        let xValueSorted,xIndexSorted = x.EnumerateColumns() |> Seq.map (fun col -> col.ToArray() |> Array.mapi (fun e t -> (t,e))  |> Array.sort |> Array.toList |> List.unzip |> fun(e,i)-> Array.ofList(e),Array.ofList(i) ) |> List.ofSeq |> List.unzip |> fun(e,i) -> Array.ofList(e),Array.ofList(i)

        let predictMatch (link:Vector<double>) (value:string)= 
            match value,family with
            | "link",_ -> link
            | "response", InvariantEqual "gaussian" -> link
            | "response", InvariantEqual "binomial" -> logistic(link)
            | _ -> raiseException "predict either link or response"      
               
        member val Forest = Array.empty with get,set
        member val RND = System.Random
        member this.Fit(maxTrees:int,?seed:int) =
            let seed= defaultArg seed 1
            let rnd= this.RND seed
            let checkMaxTrees = if maxTrees < 1 then raiseException "the minimum number of trees is 1"
            this.Forest <- Array.create maxTrees Empty
            let wTilde = Array.create yTrain.Count 0.0

            for i in [0..maxTrees-1] do
                let nodeId = [|0|]

                let xInNode = Array.init n (fun _ -> rnd.NextDouble() <= sub_sample)
                let fInTreeArray = shuffle x.ColumnCount rnd |> Seq.take nf
                let fInTree = Array.create x.ColumnCount false 
                for e in fInTreeArray do fInTree.[e] <- true                
                this.Forest.[i] <- growTree Empty nodeId fInTree xInNode depth xValueSorted xIndexSorted y wTilde gTilde hTilde eta lambda gamma
                match this.Forest.[i] with
                    | Empty -> ()
                    | _ -> 
                        for j in [0..n-1] do
                            if xInNode.[j] then
                                yTilde.[j] <- yTilde.[j] + wTilde.[j]
                            else yTilde.[j] <- yTilde.[j] + predictTree this.Forest.[i] (x.Row(j))
                            let gt,ht= gh y.[j] yTilde.[j]
                            gTilde.[j] <- gt 
                            hTilde.[j] <- ht 
            
        member this.CVFit (metric:string, nTrees:int ,?nFolds:int, ?earlyStopRounds:int,?seed:int)=
            let seed= defaultArg seed 1
            let rnd= this.RND seed
            let nFolds = defaultArg nFolds 5
            let cvForests = Array2D.create nTrees nFolds Empty

            let foldArray = Array.init n (fun _ -> rnd.Next() % nFolds )
            let nRounds = defaultArg earlyStopRounds 10

            let inMetricArray = Array2D.create nTrees nFolds 0.0

            let outMetricArray = Array2D.create nTrees nFolds 0.0

            let metricFun = match metric with
                            | InvariantEqual "RMSE" -> RMSE
                            | InvariantEqual "AUC" -> AUC
                            | InvariantEqual "logloss" -> logloss
                            | _ -> raiseException "Please choose either RMSE, AUC or logloss"

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
                    let fInTreeArray = shuffle x.ColumnCount rnd |> Seq.take nf
                    let fInTree = Array.create x.ColumnCount false 
                    for e in fInTreeArray do fInTree.[e] <- true

                    cvForests.[i,f] <- growTree Empty nodeId fInTree xInNode depth xValueSorted xIndexSorted y wTildeCVs.[f] gTildeCVs.[f] hTildeCVs.[f] eta lambda gamma
                    match cvForests.[i,f] with
                        | Empty -> ()
                        | _ -> 
                            for j in [0..n-1] do
                                if foldArray.[j] <>f then
                                    if xInNode.[j] then
                                        yTildeCVs.[f].[j] <- yTildeCVs.[f].[j] + wTildeCVs.[f].[j]
                                    else yTildeCVs.[f].[j] <- yTildeCVs.[f].[j] + predictTree cvForests.[i,f] (x.Row(j))
          
                                    let gt,ht= gh y.[j] yTildeCVs.[f].[j]
                                    gTildeCVs.[f].[j] <- gt 
                                    hTildeCVs.[f].[j] <- ht 
                    inMetricArray.[i,f] <- metricFun (y |>Array.indexed |> Array.filter (fun (ii, _) -> foldArray.[ii] <>f) |> Array.map snd |> DenseVector.ofArray) (predictMatch (yTildeCVs.[f] |>Array.indexed |> Array.filter (fun (ii, _) -> foldArray.[ii] <>f) |> Array.map snd |> DenseVector.ofArray)  "response")

                    yHatInFold.[f] <- yHatInFold.[f] + this.Predict(x, cvForests.[i,f],foldArray, f) 
                    outMetricArray.[i,f] <- metricFun yInFold.[f] (predictMatch yHatInFold.[f] "response")
                
                let muIn = inMetricArray.[i,*] |> Array.average
                let sdIn = sqrt((inMetricArray.[i,*] |> Array.map (fun e -> e*e) |> Array.average) - muIn*muIn)

                let muOut = outMetricArray.[i,*] |> Array.average
                let sdOut = sqrt((outMetricArray.[i,*] |> Array.map (fun e -> e*e) |> Array.average) - muOut*muOut)
                printfn "Iter: %5d \t %s \t Train: %10.10f+%5.5f \t Test: %10.10f+%5.5f" i metric muIn sdIn muOut sdOut
        
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
