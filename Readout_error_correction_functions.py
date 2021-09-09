import numpy as np

# Inverse single-qubit readout error matrix, p_0, p_1
def inverse_D_mat( p, q ):
    return np.array([ [1 - p/(p + q -1), q/(p + q -1)], [p/(p + q -1), 1 - q/(p + q -1)] ])

# Convert a bit vector into a positive integer num 
def booltoint(state):
    return state.dot(2**np.arange(state.size)[::-1])

# Convert an integer number to a m-bit array
def bin_array(num, m):
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int16)

# Read nth bit of integer i
def ReadBit( i , n ):
    return ( i & ( 1 << n ) ) >> n


######################################################################################  
## Functions to measure the stabilizer operators (without readout error), 1D cluster state
# n is an even integer
# \prod_ (k = flag mod 2) (g_k + 1)/2
def Stab_prod_func( bit_string, n, flag ):
    stab_prod = 1
    for k in range(0, n, 1):
        if ( k % 2 == flag ):
            if (k == 0):
                stab_prod = stab_prod * (1 + (-1)**(bit_string[k] + bit_string[k+1])) / 2
            if (k != 0) and (k != n-1 ):
                stab_prod = stab_prod * (1 + (-1)**(bit_string[k-1] + bit_string[k] + bit_string[k+1])) / 2
            if (k == n-1):
                stab_prod = stab_prod * (1 + (-1)**(bit_string[k-1] + bit_string[k])) / 2
    return int(stab_prod)
######################################################################################  



# Exact readout error corrections
######################################################################################  
# Calculate the matrix element of kronecker product
def mat_element_kron(n, readout_error_0, readout_error_1, i, j):
    row_index = i
    col_index = j
    mat_element = 1
    for k in range(0, n, 1):
        mat_element *= inverse_D_mat( readout_error_0[k], readout_error_1[k] )[row_index // 2**(n-1-k)][col_index // 2**(n-1-k)]
        row_index = row_index % 2**(n-1-k)
        col_index = col_index % 2**(n-1-k)
        
    return mat_element

# The function to generate the array indices for 1 value, recurrent function
def one_index_func(n, flag): # k % 2 = flag
    if (flag == 0): # even case, 1101
        if (n == 2):
            return [0, 3]
        if (n > 2):
            output_arr = []
            for k in one_index_func(n-2, flag):
                if (ReadBit( k , 0 ) == 0):
                    output_arr.append( 4*k )
                    output_arr.append( 4*k + 3 )
                if (ReadBit( k , 0 ) == 1):
                    output_arr.append( 4*k + 1 )
                    output_arr.append( 4*k + 2 )
            return output_arr
    if (flag == 1): # odd case, 1011
        if (n == 2):
            return [0, 3]
        if (n > 2):
            output_arr = []
            for k in one_index_func(n-2, flag):
                if (ReadBit( k , n-3 ) == 0):
                    output_arr.append( k )
                    output_arr.append( k + 2**(n-2) + 2**(n-1) )
                if (ReadBit( k , n-3 ) == 1):
                    output_arr.append( k + 2**(n-2) )
                    output_arr.append( k + 2**(n-1) )
            return output_arr

# n is an even integer
# diag (\prod_ (k = flag mod 2) (g_k + 1)/2 * (inverse_D_k-1) ) * \otimes_k inverse_D_k
# Calculate the element of O^c row vector
def Stab_prod_exact_correct_func( bit_string, n, readout_error_0, readout_error_1, one_index ):
    col_index = booltoint(bit_string)
    
    element_sum = 0 # summation variable
    for row_index in one_index:
        element_sum += mat_element_kron(n, readout_error_0, readout_error_1, row_index, col_index)
        
    return element_sum
######################################################################################  

# Approximate readout error corrections
######################################################################################  
# Function to generate the check_table
# n is an even integer
def check_table_func( n, readout_error_0, readout_error_1 ):
    check_table = np.zeros([n, 8])
    
    # \prod_k \in even (g_k + 1)/2 * inverse_D_k \otimes inverse_D_k+1
    for k in range(0, n, 2):
        if (k == 0): # inverse_D_2 = inverse_D_k \otimes inverse_D_k+1
            inverse_D_2 = np.kron( inverse_D_mat(readout_error_0[k], readout_error_1[k]), inverse_D_mat(readout_error_0[k+1], readout_error_1[k+1]) )
            check_table[k][0:4:] = np.dot( np.array([1, 0, 0, 1]), inverse_D_2 )
        if (k != 0): # inverse_D_3 = I \otimes inverse_D_k \otimes inverse_D_k+1
            inverse_D_3 = np.kron( inverse_D_mat(readout_error_0[k], readout_error_1[k]), inverse_D_mat(readout_error_0[k+1], readout_error_1[k+1]) )
            inverse_D_3 = np.kron( np.eye(2), inverse_D_3 )
            check_table[k] = np.dot( np.array([1, 0, 0, 1, 0, 1, 1, 0]), inverse_D_3 )
    # \prod_k \in odd (g_k + 1)/2 * inverse_D_k-1 \otimes inverse_D_k
    for k in range(1, n+1, 2):
        if (k == n-1): # inverse_D_2 = inverse_D_k-1 \otimes inverse_D_k
            inverse_D_2 = np.kron( inverse_D_mat(readout_error_0[k-1], readout_error_1[k-1]), inverse_D_mat(readout_error_0[k], readout_error_1[k]) )
            check_table[k][0:4:] = np.dot( np.array([1, 0, 0, 1]), inverse_D_2 )
        if (k != n-1): # inverse_D_3 = inverse_D_k-1 \otimes inverse_D_k \otimes I  
            inverse_D_3 = np.kron( inverse_D_mat(readout_error_0[k-1], readout_error_1[k-1]), inverse_D_mat(readout_error_0[k], readout_error_1[k]) )
            inverse_D_3 = np.kron( inverse_D_3, np.eye(2) )
            check_table[k] = np.dot( np.array([1, 0, 0, 1, 0, 1, 1, 0]), inverse_D_3 )
    return check_table

# n is an even integer
# \prod_ (k = flag mod 2) (g_k + 1)/2 * (inverse_D_k-1) \otimes inverse_D_k (\otimes inverse_D_k+1)
def Stab_prod_approx_correct_func( bit_string, n, flag, check_table ):
    stab_value = []
    for k in range(0, n, 1):
        if ( k % 2 == flag ):
            if (k == 0):
                stab_value.append( check_table[k][ booltoint( np.array([bit_string[k], bit_string[k+1]]) ) ] )
            if (k != 0) and (k != n-1 ):
                stab_value.append( check_table[k][ booltoint( np.array([bit_string[k-1], bit_string[k], bit_string[k+1]]) ) ] )
            if (k == n-1):
                stab_value.append( check_table[k][ booltoint( np.array([bit_string[k-1], bit_string[k]]) ) ] )
  
    # factorization product
    prod_val = np.prod(stab_value)
    if np.all( np.array(stab_value) > 0 ):
        return prod_val # if individually Stab=1, then it is the direct product value
    else:
        return -np.abs(prod_val) # otherwies there is some 0 value
######################################################################################