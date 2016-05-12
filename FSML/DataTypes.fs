module PrepareData =

    open FSharp.Data
    type csvProvider=CsvProvider<"/cd/j/bank.csv">
    let bank = csvProvider.Load("/cd/j/bank/csv")

    let getHeader someHeader = match someHeader with
                               | None -> raise (System.Exception("csv file has no header"))
                               | Some header -> header |> Array.toList
    let header = bank.Headers |> getHeader
