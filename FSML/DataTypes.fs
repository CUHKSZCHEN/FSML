module DataTypes

    open System.IO
    open System.Collections.Generic
    open System
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    let raiseExcetion (x:string) = raise (System.Exception x)

    type VectorOrMatrix =
        | V of Vector<double>
        | M of Matrix<double>

    type Variable = 
        | Y of double
        | X of int*double  

    let parseClassificationVariable (e:string) = 
        if e.StartsWith "+"  then Y 1.
        else if e.StartsWith "-" then Y 0.
        else let split = e.Split [|':'|]
             X (int split.[0],double split.[1])

    let parseRegressionVariable (e:string) =
        if e.StartsWith "+"  then raiseExcetion "incorrect response variable type"
        else if e.Contains(":") 
        then 
            let split = e.Split [|':'|]
            X (int split.[0],double split.[1])
        else Y (double e)

    type Imputation = Mean=0|Median=1|Min=2|Max=3
        

    type data (foldedData:Map<int,seq<double*Map<int,double>>>, features:Set<int>)  =
        member this.FoldNumber= foldedData.Count

        member this.GetTrain (fold:int)= 
            let temp= foldedData |> Map.filter (fun key _ -> key <>fold) |> Map.toSeq |> Seq.map snd 
            seq {for i in temp do
                    for j in i do yield j}

        member this.GetTest (fold: int)= foldedData .[fold]

        member this.GetColumn (f:int) (rows:seq<Map<int,double>>)=seq {for row in rows -> (if row.ContainsKey f then row.[f] else raiseExcetion "missing value" )}

        member this.Train (fold:int)=
            this.GetTrain fold |> List.ofSeq |> List.unzip |> 
                        (fun (a,b) -> (seq{for f in (Set.toList features |> List.sort) ->  (f,b |> this.GetColumn f)} |> Map.ofSeq |> Map.toSeq |> Seq.map snd |> DenseMatrix.ofColumnSeq ,a|> DenseVector.ofSeq))      

        
        member this.Test (fold:int)=
            this.GetTest fold |> List.ofSeq |> List.unzip |> 
                        (fun (a,b) -> (seq{for f in (Set.toList features |> List.sort) ->  (f,b |> this.GetColumn f)} |> Map.ofSeq |> Map.toSeq |> Seq.map snd |> DenseMatrix.ofColumnSeq ,a|> DenseVector.ofSeq))



    type readData (filePath: string, response: string)=

        let parseVariable = 
            match response with
                | "binary" -> parseClassificationVariable
                | "continuous" -> parseRegressionVariable
                | _ -> raiseExcetion "resposne variable must be either binary for classification for continuous for regression"
        
        let parseLine (line : string) =
            line.Split ([|' ';'\t'|], System.StringSplitOptions.RemoveEmptyEntries) |> Array.map parseVariable
         
        let parseEntry (row: array<Variable>) = 
            if row.Length=1 
            then (match row.[0] with 
                 |Y y -> y 
                 |_ ->  raiseExcetion "missing label" ), Map.empty 
            else (match row.[0] with 
                 |Y y -> y 
                 |_ -> raiseExcetion "missing label" ), (row.[1..] |> Seq.ofArray |> Seq.choose 
                    (fun x -> match x with     
                                |X (feature,value) -> Some (feature,value)
                                |_ -> None ) |> Map.ofSeq)
        
        let allRows = seq {
            use sr= new StreamReader (filePath)
            while not sr.EndOfStream do
            yield sr.ReadLine()} |> Seq.map parseLine |> Seq.map parseEntry
        
        let getColumn (f:int) (rows:seq<Map<int,double>>)=seq {for row in rows -> (if row.ContainsKey f then (Some row.[f]) else None)}

        member private this.AddToXDict ((xDict:Dictionary<int,List<Option<double>>>)) (entry:Map<int,double>)=
            for f in this.Features do 
                if entry.ContainsKey f then xDict.[f].Add (Some entry.[f]) else xDict.[f].Add None

        member this.Features =
            allRows |> Seq.map snd |> Seq.fold (fun acc elem -> acc + (elem |> Map.toSeq |> Seq.map fst |> Set.ofSeq)) Set.empty<int>              
               
        member this.XByColumns = 
            let rows=allRows|> Seq.map snd
            seq{
            for f in this.Features ->  (f,rows |> getColumn f)} |> Map.ofSeq

        member this.N= 
            Seq.length allRows

        member this.CreateFold k seed  =
            if k>=this.N then raiseExcetion "too many folds"
            else
                let rnd = System.Random(seed)
                allRows |> Seq.groupBy (fun _ -> rnd.Next() % k) |> Map.ofSeq
