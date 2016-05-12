module DataTypes

    open System.IO

    type Variable = 
        | Y of int
        | X of int*double  

    let parseVariable (e:string) = 
        if e.StartsWith "+"  then Y 1
        else if e.StartsWith "-" then Y 0
        else let split = e.Split [|':'|]
             X (int split.[0],double split.[1])

    let private parseLine (line : string) =
        let split = line.Split ([|' ';'\t'|], System.StringSplitOptions.RemoveEmptyEntries)
        split |> Array.map parseVariable
    
    let readTable (filePath:string) = seq {
        use sr= new StreamReader (filePath)
        while not sr.EndOfStream do
        yield sr.ReadLine()
    }

    let parseRow (row: array<Variable>) = 
        if row.Length=1 
        then (match row.[0] with 
             |Y y -> Some y 
             |_ -> None), Map.empty 
        else (match row.[0] with 
             |Y y -> Some y 
             |_ -> None), (row.[1..] |> Seq.ofArray |> Seq.choose 
              (fun x -> match x with     
                        |X (feature,value) -> Some (feature,value)
                        |_ -> None ) |> Map.ofSeq)


    let getDict table = table|> Seq.map parseLine |> Seq.map parseRow

    let getAllFeatures seqDict =
        let featureSet = Set.empty<int>
        seqDict |> Seq.map snd |> Seq.fold (fun acc elem -> acc + (elem |> Map.toSeq |> Seq.map fst |> Set.ofSeq)) featureSet 
