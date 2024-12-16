import numpy as np

def dynamicProgram(unaryCosts, pairwiseCosts):
    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # define parent matrix - each element will contain the (vertical) index of
    # the node that preceded it on the path.  Since the first column has no
    # parents, we will leave it set to zeros.
    parents = np.zeros([nNodesPerPosition, nPosition])

    # FORWARD PASS

    # fill in first column of minimum cost matrix
    for cNode in range(nNodesPerPosition):
        minimumCost[cNode, 0] = unaryCosts[cNode, 0]

    # Now run through each position (column)
    for cPosition in range(1, nPosition):
        # run through each node (element of column)
        for cNode in range(nNodesPerPosition):
            # now we find the costs of all paths from the previous column to this node
            possPathCosts = np.zeros([nNodesPerPosition, 1])
            for cPrevNode in range(nNodesPerPosition):
                possPathCosts[cPrevNode, 0] = (
                    minimumCost[cPrevNode, cPosition-1] +  # Previous minimum cost
                    pairwiseCosts[cPrevNode, cNode] +      # Pairwise transition cost
                    unaryCosts[cNode, cPosition]           # Unary cost for current node
                )

            # find the minimum of the possible paths 
            minCost = np.min(possPathCosts)
            ind = np.argmin(possPathCosts)
            
            # store the minimum cost in the minimumCost matrix
            minimumCost[cNode, cPosition] = minCost
            
            # store the parent index in the parents matrix
            parents[cNode, cPosition] = ind

    # BACKWARD PASS

    # we will now fill in the bestPath vector
    bestPath = np.zeros([nPosition, 1])
    
    # find the index of the overall minimum cost from the last column 
    minCost = np.min(minimumCost[:, -1])
    minInd = np.argmin(minimumCost[:, -1])
    bestPath[-1] = minInd

    # find the parent of the node you just found
    bestParent = parents[minInd, -1]

    # run backwards through the cost matrix tracing the best patch
    for cPosition in range(nPosition-2, -1, -1):
        # work through matrix backwards, updating bestPath by tracing parents
        bestPath[cPosition] = bestParent
        bestParent = parents[int(bestParent), cPosition]

    return bestPath

def dynamicProgramVec(unaryCosts, pairwiseCosts):
    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # Initialize first column with unary costs
    minimumCost[:, 0] = unaryCosts[:, 0]

    # Precompute pairwise cost matrix for efficient broadcasting
    pairwiseCostTiled = np.tile(pairwiseCosts, (nPosition-1, 1, 1))

    # Forward pass using vectorized operations
    for cPosition in range(1, nPosition):
        # Create a matrix of all possible previous costs
        prevCostMatrix = np.tile(minimumCost[:, cPosition-1], (nNodesPerPosition, 1)).T
        
        # Compute transition costs using broadcasting
        transitionCosts = prevCostMatrix + pairwiseCosts
        
        # Add unary costs for current position
        totalCosts = transitionCosts + unaryCosts[:, cPosition]
        
        # Find minimum costs for each node
        minimumCost[:, cPosition] = np.min(totalCosts, axis=0)

    # Backward pass to find best path
    bestPath = np.zeros([nPosition], dtype=int)
    
    # Find the node with minimum cost in the last column
    bestPath[-1] = np.argmin(minimumCost[:, -1])

    # Trace back the path
    for cPosition in range(nPosition-2, -1, -1):
        # Find the parent node that minimized the cost
        prevCostMatrix = np.tile(minimumCost[:, cPosition], (nNodesPerPosition, 1)).T
        totalCosts = prevCostMatrix + pairwiseCosts
        bestPath[cPosition] = np.argmin(totalCosts[:, bestPath[cPosition+1]])

    return bestPath