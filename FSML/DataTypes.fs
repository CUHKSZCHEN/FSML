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

    type Imputation = Mean=0|Median=1|Min=2|Max=3

    type data (foldedData:Map<int,seq<double*Map<int,double>>>, features:Set<int>)  =
        member this.FoldNumber= foldedData.Count

        member this.GetTrain (fold:int)= 
            let temp= foldedData |> Map.filter (fun key _ -> key <>fold) |> Map.toSeq |> Seq.map snd 
            seq {for i in temp do
                    for j in i do yield j}

        member this.GetTest (fold: int)= foldedData .[fold]

        member this.XByColumns rows = 
            let Rows=rows|> Seq.map snd
            seq{
            for f in (Set.toList features |> List.sort) ->  (f,Rows |> this.GetColumn f)} |> Map.ofSeq

        member this.GetColumn (f:int) (rows:seq<Map<int,double>>)=seq {for row in rows -> (if row.ContainsKey f then row.[f] else raiseExcetion "missing value" )}

        member this.Train (fold:int)=
            let xy=this.GetTrain fold
            let x = this.XByColumns xy
            let y = xy |> Seq.map fst |> DenseVector.ofSeq  
            let columnSeq = x |> Map.toSeq |> Seq.map snd 
            let xMatrix = DenseMatrix.ofColumnSeq columnSeq         
            xMatrix,y
        
        member this.Test (fold:int)=
            let xy=this.GetTest fold
            let x = this.XByColumns xy
            let y = xy |> Seq.map fst |> DenseVector.ofSeq 
            let columnSeq = x |> Map.toSeq |> Seq.map snd 
            let xMatrix = DenseMatrix.ofColumnSeq columnSeq         
            xMatrix,y


    type readData (filePath: string)=

        member this.AddToXDict ((xDict:Dictionary<int,List<Option<double>>>)) (entry:Map<int,double>)=
            for f in this.Features do 
                if entry.ContainsKey f then xDict.[f].Add (Some entry.[f]) else xDict.[f].Add None

        member this.ParseVariable (e:string) = 
            if e.StartsWith "+"  then Y 1.
            else if e.StartsWith "-" then Y 0.
            else let split = e.Split [|':'|]
                 X (int split.[0],double split.[1])

        member this.ParseLine (line : string) =
            line.Split ([|' ';'\t'|], System.StringSplitOptions.RemoveEmptyEntries) |> Array.map this.ParseVariable
        
        
        
        member this.ParseEntry (row: array<Variable>) = 
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
             
        member this.Rows = seq {
            use sr= new StreamReader (filePath)
            while not sr.EndOfStream do
            yield sr.ReadLine()} |> Seq.map this.ParseLine |> Seq.map this.ParseEntry

        member this.Features =
            this.Rows |> Seq.map snd |> Seq.fold (fun acc elem -> acc + (elem |> Map.toSeq |> Seq.map fst |> Set.ofSeq)) Set.empty<int>              
        
        member this.GetColumn (f:int) (rows:seq<Map<int,double>>)=seq {for row in rows -> (if row.ContainsKey f then (Some row.[f]) else None)}
        
        member this.XByColumns = 
            let rows=this.Rows|> Seq.map snd
            seq{
            for f in this.Features ->  (f,rows |> this.GetColumn f)} |> Map.ofSeq

        member this.N= 
            Seq.length this.Rows

        member this.CreateFold k seed  =
            let rnd = System.Random(seed)
            this.Rows |> Seq.groupBy (fun _ -> rnd.Next() % k) |> Map.ofSeq
