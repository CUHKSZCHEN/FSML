module Tree

    open DataTypes
    open Utilities
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra

    type node = {mutable nodeId:int; mutable featureId:int; mutable splitValue:double; mutable leafValue:double; mutable isLeaf:bool}

    type tree<'node>  =
        | Empty
        | TreeNode of 'node * 'node tree* 'node tree

    let rec getHeight (tree: tree<'a>) =
        match tree with
        | Empty -> 0
        | TreeNode(head,left,right) -> 1 + max (getHeight left) (getHeight right)
    
    let getDepth (tree: tree<'a>) = -1 + getHeight tree

    let rec predictTree tree (x:Vector<double>)= 
        match tree, x with
        | Empty,_ -> 0.0
        | TreeNode(head,left,right),_ ->
            match head.isLeaf with
            | true -> head.leafValue
            | _ -> if x.[head.featureId]<= head.splitValue then predictTree left x else predictTree right x
    

    let predictForestforVector forest (x:Vector<double>)  = forest |> Array.map (fun tree -> predictTree tree x ) |> Array.sum
    let predictForestforMatrix forest (x:Matrix<double>)  = x.EnumerateRows() |> Seq.map (fun row -> predictForestforVector forest row) |> Array.ofSeq

    let predictTreeforMatrixInFold tree (x:Matrix<double>) (foldArray: int[]) (fold:int)  = x.EnumerateRowsIndexed() |> Seq.filter (fun (i,_) -> foldArray.[i]=fold) |> Seq.map (fun (_, row) -> predictTree tree row) |> Array.ofSeq

    let gh_lm (y:double) (pred:double) = pred - y, 1.0

    let gh_lr (y:double) (pred:double) =
        let prob= 1.0/(1.0+exp(-pred))
        prob - y , max (prob * (1.0 - prob)) 1e-16


    let splitNode (fInTree: bool[], xInNode: bool [], xValueSorted: double [][], xIndexSorted: int [][],gTilde: double [],hTilde: double [],lambda:double,gamma:double) = 
        let mutable score= 0.0
        let mutable bestFeature,bestBreak,bestIndex = 0,0.0,0
        let mutable wLeft, wRight = 0.0,0.0
        let mutable doSplit = false
        let nIn = xInNode |> Array.map (fun e -> if e then 1 else 0) |> Array.sum
        let ncol =fInTree.Length
        let nrow =xInNode.Length
        let g = xInNode |> Array.mapi (fun i e -> if e then gTilde.[i] else 0.0) |> Array.sum
        let h = xInNode |> Array.mapi (fun i e -> if e then hTilde.[i] else 0.0) |> Array.sum

        let refScore = (g*g)/(h+lambda)

        for k in [0..ncol-1] do
            if fInTree.[k] then
                let mutable nLeft = 0
                let mutable gLeft,gRight,hLeft,hRight = 0.0,0.0,0.0,0.0
                for i in [0..nrow-1] do
                    let index = xIndexSorted.[k].[i]
                    if (xInNode.[index] && (nLeft < nIn-1)) then
                        nLeft <- nLeft + 1 
                        gLeft <- gLeft + gTilde.[index]
                        gRight <- g - gLeft
                        hLeft <- hLeft + hTilde.[index]
                        hRight <- h - hLeft
                        let scoreNew = (gLeft * gLeft)/(hLeft+lambda) + (gRight*gRight)/(hRight+lambda) - refScore

                        if scoreNew > score then
                            doSplit <- true
                            score <- scoreNew
                            bestFeature <- k
                            bestBreak <- xValueSorted.[k].[i]
                            bestIndex <- i
                            wLeft <- (- gLeft/(hLeft + lambda))
                            wRight <- (- gRight/(hRight + lambda))

        doSplit && (0.5*score > gamma),bestFeature,bestBreak,bestIndex,wLeft,wRight,score


    let rec growTree (currentTree: tree<node>) (nodeId: int []) (fInTree: bool []) (xInNode: bool [])  (maxDepth:int) (xValueSorted: double [][]) (xIndexSorted: int [][]) (y: double []) (wTilde: double []) (gTilde: double []) (hTilde: double []) (eta:double) (lambda:double) (gamma:double)=
        let ncol = fInTree.Length
        let nrow = y.Length
        if maxDepth = 0 then currentTree
        else 
            match currentTree with
            | Empty -> 

                let doSplit,bestFeature,bestBreak,bestIndex,wLeft,wRight,score = splitNode (fInTree,xInNode,xValueSorted,xIndexSorted,gTilde,hTilde,lambda,gamma)
                let wLeftScaled= wLeft * eta
                let wRightScaled= wRight * eta
                if doSplit then
                    let xInLeftNode = Array.copy xInNode
                    let xInRightNode = Array.copy xInNode
        
                    for i in [0..nrow-1] do
                        let index = xIndexSorted.[bestFeature].[i]
                        if xInNode.[index] then
                            if i <= bestIndex then 
                                xInLeftNode.[index] <- true
                                wTilde.[index] <- wLeftScaled
                                xInRightNode.[index] <- false
                            else 
                                xInRightNode.[index] <- true
                                wTilde.[index] <- wRightScaled
                                xInLeftNode.[index] <- false
                                   
                    let currentNode = {nodeId=nodeId.[0]; featureId=bestFeature;splitValue=bestBreak;leafValue=0.0;isLeaf=false}
                    nodeId.[0] <- nodeId.[0]+1
                    let mutable leftTree = TreeNode({nodeId=nodeId.[0]; featureId= -1;splitValue=0.0;leafValue=wLeftScaled;isLeaf=true},Empty,Empty)
                    leftTree <- growTree leftTree nodeId fInTree xInLeftNode (maxDepth-1) xValueSorted xIndexSorted y wTilde gTilde hTilde eta lambda gamma

                    nodeId.[0] <- nodeId.[0]+1
                    let mutable rightTree = TreeNode({nodeId=nodeId.[0];featureId= -1;splitValue=0.0;leafValue=wRightScaled;isLeaf=true},Empty,Empty)
                    rightTree <- growTree rightTree nodeId fInTree xInRightNode (maxDepth-1) xValueSorted xIndexSorted y wTilde gTilde hTilde eta lambda gamma

                    TreeNode(currentNode,leftTree,rightTree)
                else Empty
            | TreeNode(head,left,right) ->
                match head.isLeaf with
                | true -> 
                    let newNode = growTree Empty nodeId fInTree xInNode maxDepth xValueSorted xIndexSorted y wTilde gTilde hTilde eta lambda gamma
                    match newNode with
                    | Empty -> currentTree
                    | _ -> newNode
                | _ -> raiseException "try to split a node which is not a leaf"
