module DataTypes

    open System.IO
    open System.Collections.Generic

    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type Variable = 
        | Y of int
        | X of int*double  

    type Imputation = Mean=0|Median=1|Min=2|Max=3

    type readData (filePath: string)=
        member this.ParseVariable (e:string) = 
            if e.StartsWith "+"  then Y 1
            else if e.StartsWith "-" then Y 0
            else let split = e.Split [|':'|]
                 X (int split.[0],double split.[1])

        member this.ParseLine (line : string) =
            line.Split ([|' ';'\t'|], System.StringSplitOptions.RemoveEmptyEntries) |> Array.map this.ParseVariable
        
        member this.ParseEntry (row: array<Variable>) = 
            if row.Length=1 
            then (match row.[0] with 
                 |Y y -> y 
                 |_ ->  raise (System.Exception("missing label")) ), Map.empty 
            else (match row.[0] with 
                 |Y y -> y 
                 |_ -> raise (System.Exception("missing label")) ), (row.[1..] |> Seq.ofArray |> Seq.choose 
                    (fun x -> match x with     
                                |X (feature,value) -> Some (feature,value)
                                |_ -> None ) |> Map.ofSeq)
             
        member this.Rows = seq {
            use sr= new StreamReader (filePath)
            while not sr.EndOfStream do
            yield sr.ReadLine()} |> Seq.map this.ParseLine |> Seq.map this.ParseEntry

        member this.Features =
            this.Rows |> Seq.map snd |> Seq.fold (fun acc elem -> acc + (elem |> Map.toSeq |> Seq.map fst |> Set.ofSeq)) Set.empty<int> 
        
        member this.XByColumns=
            let xByColumn = new Dictionary<int,List<Option<double>>>()
            for f in this.Features 
                do xByColumn.Add(f, new List<Option<double>>())
            let rows= (this.Rows |> Seq.map snd) 
            for row in rows do 
                for f in this.Features do 
                    if row.ContainsKey f then xByColumn.[f].Add (Some row.[f]) else xByColumn.[f].Add None
            xByColumn
        
        member this.Impute (imputation:Imputation)= 
            match imputation with
            | Imputation.Mean -> 

        member this.

        member this.XMatrix(features) =
            if Set.isSuperset this.Features (Set.ofList features) then
                let xVectors = new Dictionary<int,vector<double>>()
                for f in features do
                    this.XByColumns.[f] |> List.map (fun x-> match x with
                        | Some value -> value
                        | _ -> raise (System.Exception("missing value for feature"+f)))
            else raise System.Exception("feature not found")





