import numpy as np


# to find the number of min line cover
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
            A1 = A1[:n, :]
            nsz = np.array([])
            L1 = fr(0, nsz, A1)
            a, b = np.shape(A1)
            if np.sum(L1 != -1) == a:
                L1 = np.where(L1 >= f0[i], L1 + 1, L1)
                L = np.append(L1, f0[i])
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

# better fr (DFS)
def fr_plus(A):
    a, b = np.shape(A)
    r_able_c = [[] for _ in range(a)]
    for i in range(a):
        for j in range(b):
            if A[i, j] == 0:
                r_able_c[i].append(j)
    c_to_r = [-1] * b

    def DFS(r, dis_c):
        for c in r_able_c[r]:
            if dis_c[c]:
                continue
            dis_c[c] = True
            if c_to_r[c] == -1 or DFS(c_to_r[c], dis_c):
                c_to_r[c] = r
                return True
        return False

    for r in range(a):
        dis_c = [False] * b
        if DFS(r, dis_c):
            a = a

    L = np.full(a, -1, dtype=int)
    for c, r in enumerate(c_to_r):
        if r != -1:
            L[r] = c

    return L


def hungarian_algorithm(cost_matrix, n_rows, n_cols, min_dim):
    '''
    Hungarian algorithm iteration with row priority (when n_rows <= n_cols)
    '''
    max_iter = 42
    iter_count = 0

    while iter_count < max_iter:
        #print('\nIteration', iter_count)
        ''' Use fr function to find assignment '''
        L = np.array([])
        #print('Input cost matrix:\n', cost_matrix)
        #result = fr(0, L, cost_matrix)
        result = fr_plus(cost_matrix)
        #print('fr function result:\n', result)

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

        #print('Marked rows:\n', marked_rows)
        #print('Marked cols:\n', marked_cols)
        #print('Cost matrix:\n', cost_matrix)

        # Find minimum uncovered value
        min_val = float('inf')
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx in marked_rows and col_idx not in marked_cols:
                    min_val = min(min_val, cost_matrix[row_idx, col_idx])

        if min_val == float('inf') or min_val == 0:
            break

        #print('Minimum uncovered value:\n', min_val)

        # Adjust matrix: add to covered rows, subtract from uncovered columns
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx not in marked_rows:
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
    # Test 1
    print("=" * 60)
    print("Example 1")
    print("=" * 60)
    A = np.array([[3, 7, 1], [8, 2, 5], [9, 1, 4], [6, 3, 7], [2, 8, 6], [5, 4, 9], [1, 7, 2]])
    c = np.array([3, 2, 4])

    print("Simple matrix:")
    print(A)
    print("Column capacities:", c)

    result = assignment(A, c)
    print("Assignment result:")
    print(result)
    print(f"Matches: {np.sum(result)}/7")

    # Test 2
    print("\n" + "=" * 60)
    print("Example 2")
    print("=" * 60)

    A2 = np.array([
        [12, 8, 15, 6, 9, 11, 4, 7],
        [5, 13, 2, 10, 14, 3, 8, 12],
        [9, 4, 11, 7, 2, 15, 6, 13],
        [3, 14, 8, 5, 12, 1, 9, 4],
        [7, 2, 13, 9, 6, 8, 11, 3],
        [15, 6, 4, 12, 1, 7, 2, 10],
        [1, 9, 7, 3, 13, 5, 14, 8],
        [11, 3, 6, 14, 8, 2, 5, 15],
        [4, 12, 9, 1, 7, 13, 3, 6],
        [8, 5, 2, 11, 4, 9, 12, 1]
    ])

    c2 = np.array([2, 3, 1, 4, 2, 3, 1, 2])

    print("Complex matrix:")
    print(A2)
    print("Column capacities:", c2)

    result2 = assignment(A2, c2)
    print("Assignment result:")
    print(result2)
    print(f"Matches: {np.sum(result2)}/{A2.shape[0]}")

    # Test 3
    print("\n" + "=" * 60)
    print("Example 3")
    print("=" * 60)

    A3 = np.array([
        [4, 2, 6],
        [3, 5, 2],
        [1, 3, 4],
        [5, 1, 3],
        [2, 4, 1]
    ])

    c3 = np.array([1, 2, 1])

    print("Limited capacity matrix:")
    print(A3)
    print("Column capacities:", c3)

    result3 = assignment(A3, c3)
    print("Assignment result:")
    print(result3)
    print(f"Matches: {np.sum(result3)}/{A3.shape[0]}")

    # Test 4
    print("\n" + "=" * 60)
    print("Example 4")
    print("=" * 60)

    A4 = np.array([
        [100, 95, 90, 85],
        [80, 75, 70, 65],
        [60, 55, 50, 45],
        [40, 35, 30, 25],
        [20, 15, 10, 5]
    ])

    c4 = np.array([1, 2, 1, 1])

    print("High cost matrix:")
    print(A4)
    print("Column capacities:", c4)

    result4 = assignment(A4, c4)
    print("Assignment result:")
    print(result4)
    print(f"Matches: {np.sum(result4)}/{A4.shape[0]}")
