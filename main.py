import numpy as np



#to find the number of min line cover
def fr(n, L, A):
    f0 = np.where(A[n] == 0)[0]
    diff = [x for x in f0 if x not in L]
    if len(diff) > 0:
        L = np.append(L, diff[0])
        a, b = np.shape(A)
        if n == a - 1:
            return L
        else:
            return fr(n + 1, L, A)
    else:
        for i in range(len(f0)):
            A1 = np.delete(A, f0[i], axis=1)
            nsz = np.array([])
            L1 = fr(0 , nsz, A1)
            a, b = np.shape(A1)
            if len(L1) == a:
                L = np.append(L1, f0[i])
                L = np.where(L >= f0[i], L + 1, L)
                a, b = np.shape(A)
                if n == a - 1:
                    return L
                else:
                    return fr(n + 1, L, A)
        a, b = np.shape(A)
        if n == a - 1:
            L = np.append(L, -1)
            return L
        else:
            L = np.append(L, -1)
            return fr(n + 1, L, A)



def hungarian_algorithm(cost_matrix, n_rows, n_cols, min_dim):
    '''
    Hungarian algorithm iteration with row priority (when n_rows <= n_cols)
    '''
    max_iter = 50
    iter_count = 0
    
    while iter_count < max_iter:
        ''' Use fr function to find assignment '''
        L = np.array([])
        result = fr(0, L, cost_matrix)
        
        # Check if we found a complete assignment
        complete_assignment = True
        assigned_count = 0
        for i, col_idx in enumerate(result):
            if col_idx == -1 and i < n_rows and col_idx < n_cols:
                complete_assignment = False
                break
            else:
                # if the column is assigned, increment the assigned count
                assigned_count += 1
        
        # Check if we have enough assignments (min_dim)
        if complete_assignment and assigned_count == min_dim:
            # Convert result to binary assignment matrix
            assignment_matrix = np.zeros((n_rows, n_cols), dtype=int)
            for i, col_idx in enumerate(result):
                if col_idx != -1 and i < n_rows and col_idx < n_cols:
                    assignment_matrix[i, int(col_idx)] = 1
            return assignment_matrix
        
        ''' Matrix adjustment: find minimum uncovered value and adjust '''
        # Find unmatched rows
        unmatched = [i for i in range(n_rows) if i >= len(result) or result[i] == -1]
        if not unmatched:
            break
        
        # Mark rows and columns for matrix adjustment
        marked_rows = set(unmatched)
        marked_cols = set()
        
        # Find all reachable rows and columns through alternating paths
        change = True
        while change:
            change = False
            new_rows = []
            for row in marked_rows:
                for col in range(n_cols):
                    if abs(cost_matrix[row, col]) < 1e-10 and col not in marked_cols:
                        marked_cols.add(col)
                        change = True
                        # Find rows which use this newly marked column
                        for r in range(n_rows):
                            if r < len(result) and result[r] == col and r not in marked_rows:
                                new_rows.append(r)
            # Add the new rows to the marked rows
            for row in new_rows:
                marked_rows.add(row)
        
        # Find minimum uncovered value
        min_val = float('inf')
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx not in marked_rows and col_idx not in marked_cols:
                    min_val = min(min_val, cost_matrix[row_idx, col_idx])
        
        if min_val == float('inf') or min_val == 0:
            break
        
        # Adjust matrix: add to covered rows, subtract from uncovered columns
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx in marked_rows:
                    cost_matrix[row_idx, col_idx] += min_val
                if col_idx not in marked_cols:
                    cost_matrix[row_idx, col_idx] -= min_val
        
        iter_count += 1
    
    # Return partial assignment if no complete assignment found
    assignment_matrix = np.zeros((n_rows, n_cols), dtype=int)
    partial_count = 0
    for i, col_idx in enumerate(result):
        if col_idx != -1 and i < n_rows and col_idx < n_cols:
            assignment_matrix[i, int(col_idx)] = 1
            partial_count += 1
    # Print warning for partial assignment
    if partial_count < min_dim:
        print(f"WARNING: Only {partial_count}/{min_dim} assignments found. Partial assignment returned.")
    
    return assignment_matrix



def assignment(A, c):
    '''
    Hungarian algorithm for assignment problem with column capacity constraints.
    
    Args:
        A: n_rows x n_cols cost matrix
        c: column capacity array, c[i] indicates how many times column i should be copied
    
    Returns:
        match_results: n_rows x n_cols binary matrix where match_results[i,j] = 1 indicates
                      row i is assigned to column j
    '''
    n_rows, n_cols = np.shape(A)
    
    # Create expanded matrix by duplicating columns according to capacity
    expanded_A = []
    col_mapping = []
    
    for col_idx in range(n_cols):
        for _ in range(c[col_idx]):
            expanded_A.append(A[:, col_idx])
            col_mapping.append(col_idx)
    
    expanded_A = np.array(expanded_A).T
    
    # Apply Hungarian algorithm with symmetric processing
    exp_n_rows, exp_n_cols = expanded_A.shape
    need_transpose = exp_n_rows > exp_n_cols
    
    # If n_rows > n_cols, transpose the matrix for processing
    if need_transpose:
        expanded_A = expanded_A.T
        exp_n_rows, exp_n_cols = exp_n_cols, exp_n_rows
    
    min_dim = exp_n_rows
    
    # Row reduction: subtract minimum from each row
    for row_idx in range(exp_n_rows):
        min_val = np.min(expanded_A[row_idx])
        expanded_A[row_idx] -= min_val
    
    # Use Hungarian algorithm
    assignment_matrix = hungarian_algorithm(expanded_A, exp_n_rows, exp_n_cols, min_dim)
    
    # If we transposed the input, transpose the result back
    if need_transpose:
        assignment_matrix = assignment_matrix.T
    
    # Convert assignment back to original matrix format
    match_results = np.zeros((n_rows, n_cols), dtype=int)
    
    for row_idx in range(n_rows):
        for col_idx in range(assignment_matrix.shape[1]):
            if assignment_matrix[row_idx, col_idx] == 1 and col_idx < len(col_mapping):
                original_col = col_mapping[col_idx]
                match_results[row_idx, original_col] = 1
    
    return match_results



if __name__ == "__main__":
    A = np.array([[3,7,1],[8,2,5], [9,1,4], [6,3,7], [2,8,6], [5,4,9], [1,7,2]])
    c = np.array([3,2,4])
    
    print("Original matrix:")
    print(A)
    print("Column capacities:", c)
    
    result = assignment(A, c)
    print("Assignment result:")
    print(result)
    print(f"Matches: {np.sum(result)}/7")