import numpy as np
from Clifford_circuit_operations import *
from Readout_error_correction_functions import *
import sys

#####################################################################################################
## Error probability for single- and two-qubit gates
# 1D cluster state, 50 qubits
single_qubit_error = np.array([18, 15, 9, 12, 12, 16, 10, 15, 15, 15, 15, 10, 16, 12, 21, 12, 12,13, 15, 15, 18, 12, 8, 8, 18, 12, 10, 11, 11, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]) / 10000
two_qubit_error = np.array([42, 53, 29, 33, 49, 53, 51, 54, 97, 87, 58, 58, 47, 50, 43, 59, 55, 92, 63, 82, 53, 67, 46, 77, 35, 51, 63, 54, 44, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59]) / 10000
readout_error = np.array([191, 397, 218, 342, 367, 330, 409, 270, 342, 494, 483, 455, 318, 553, 495, 521, 533, 475, 451, 353, 457, 413, 506, 448, 350, 370, 397, 517, 541, 532, 446, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452, 452]) / 10000
readout_error_0 = readout_error - 100 / 10000
readout_error_1 = readout_error + 100 / 10000 

#####################################################################################################

## Generate the random circuit ensemble

# The number of spins. We take n as an even integer
n = int(sys.argv[1])

# The readout error correction mode, 
# "Both" for the exact and approximate readout error correction
# "Exact" for only the exact readout error correction
# "Approx" for only the approximate readout error correction
# "None" for no readout error correction
Correct_Mode = sys.argv[2]

# Check the input
if ( np.all( np.isin(["Both" ,"Exact", "Approx", "None" ], Correct_Mode) == False ) == True ):
    print( "Correction Mode Wrong! (Both, Exact, Approx, None)" )
    exit()

# Number of the random circuit ensemble
Num_ensemble = n * 3000

col_num = 3 # For "Exact" and "Approx" mode
if (Correct_Mode == "Both"):
    col_num = 4
if (Correct_Mode == "None"):
    col_num = 2
# Stabilizer product array, 2 rows for k % 2 = 0,1, 4 columns for Stab_prod, Stab_prod_error, Stab_prod_exact_correct, Stab_prod_approx_correct
Stab_prod_arr = np.zeros([2, col_num])

if (Correct_Mode == "Both") or (Correct_Mode == "Approx"):
    # check table for approximate readout error correction
    check_table = check_table_func( n, readout_error_0, readout_error_1 )

for flag in range(0, 2, 1): # k % 2 = flag
    if (Correct_Mode == "Both") or (Correct_Mode == "Exact"):
        # Generate the array indices for 1 value, flag = 0, 1, exact readout error correction
        one_index = one_index_func(n, flag)
    
    for round_num in range(0, Num_ensemble):
        if ( round_num % 1000 == 0 ): 
            print( "{}: {}/{}".format(flag, round_num, Num_ensemble) )
            
            # Save the Intermediate states
            file_name = "data/1D_cluster_state_n={}_{}mod2_{}".format(n, flag, Correct_Mode)
            with open(file_name + ".csv", "a") as opt:
                np.savetxt(opt, Stab_prod_arr[flag] / (round_num + 1) )
        # A tableau consisting of binary variables xij , zij for all i ∈ {0, . . . , 2n}, 
        # j ∈ {0, . . ., n-1}, and ri for all i ∈ {0, . . ., 2n}
        Stab_mat = np.zeros([2*n+1, 2*n+1], dtype="int8")

        # Initialization to |000...>, i ∈ {0, . . . , 2n-1},
        for i in range(0, 2*n):
            Stab_mat[i][i] = 1

        # Hadamard gates on all the qubits
        for a in range(0, n):
            Stab_mat = Hadamard( Stab_mat, n, a )
        #########################################################################################   
        # Single-qubit noise channel
        operation_num_arr = []
        for a in range(0, n):
            operation_num_arr.append( int( np.random.choice(4, 1, p=[1 - single_qubit_error[a], single_qubit_error[a]/3, single_qubit_error[a]/3, single_qubit_error[a]/3]) ) )
        for a in range(0, n):
            Stab_mat = single_qubit_noise( Stab_mat, n, a, operation_num_arr[a] )
        #########################################################################################    
        # Consider the 1D cluster state, CZ gate = H CNOT H
        for a in range(0, n-1):
            # Hadamard gates on the target a+1 qubit
            Stab_mat = Hadamard( Stab_mat, n, a+1 )
            # CNOT gates from control a to target a+1
            Stab_mat = CNOT( Stab_mat, n, a, a+1 )
            # Hadamard gates on the target a+1 qubit
            Stab_mat = Hadamard( Stab_mat, n, a+1 )

        # Two-qubit noise channel
        operation_num_arr = []
        for a in range(0, n-1):
            operation_num_arr.append( int( np.random.choice(16, 1, p=[1 - two_qubit_error[a], two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15, two_qubit_error[a]/15] ) ) )
        for a in range(0, n-1):
            Stab_mat = two_qubit_noise( Stab_mat, n, a, a+1, operation_num_arr[a] )

        #########################################################################################
        # Measurement output arrays
        Output = np.zeros(n, dtype="int8")
        
        if (flag % 2 == 0): # The even measurement setting
            for a in range(0, n, 2):  
                # Hadamard gates on the even-site qubits
                Stab_mat = Hadamard( Stab_mat, n, a )
            # Single-qubit noise channel, only for the even-site qubits
            operation_num_arr = []
            for a in range(0, n):
                operation_num_arr.append( int( np.random.choice(4, 1, p=[1 - single_qubit_error[a], single_qubit_error[a]/3, single_qubit_error[a]/3, single_qubit_error[a]/3]) ) )
            for a in range(0, n, 2):
                Stab_mat = single_qubit_noise( Stab_mat, n, a, operation_num_arr[a] )
            # Measurement of qubit a in standard basis.
            for a in range(0, n): 
                [Stab_mat, Output[a]] = Measurement( Stab_mat, n, a )

        if (flag % 2 == 1): # The odd measurement setting
            for a in range(1, n+1, 2):
                # Hadamard gates on the odd-site qubits
                Stab_mat = Hadamard( Stab_mat, n, a )
            # Single-qubit noise channel, only for the odd-site qubits
            operation_num_arr = []
            for a in range(0, n):
                operation_num_arr.append( int( np.random.choice(4, 1, p=[1 - single_qubit_error[a], single_qubit_error[a]/3, single_qubit_error[a]/3, single_qubit_error[a]/3]) ) )
            for a in range(1, n+1, 2):
                Stab_mat = single_qubit_noise( Stab_mat, n, a, operation_num_arr[a] )
            # Measurement of qubit a in standard basis.
            for a in range(0, n):
                [Stab_mat, Output[a]] = Measurement( Stab_mat, n, a )
        #############################################################          
        # 1. Stabilizer product operators for (k % 2 = flag) sites, without readout errors, Stab_prod_arr[flag][0]
        Stab_prod_arr[flag][0] += Stab_prod_func( Output, n, flag )

        #####################################################
        # Readout noise channel
        operation_num_arr = []
        for a in range(0, n):
            if (Output[a] == 0): # p_0
                operation_num_arr.append( int( np.random.choice(2, 1, p=[1 - readout_error_0[a], readout_error_0[a]]) ) )
            if (Output[a] == 1): # p_1
                operation_num_arr.append( int( np.random.choice(2, 1, p=[1 - readout_error_1[a], readout_error_1[a]]) ) )
        Output = (Output + np.array(operation_num_arr)) % 2
        #####################################################
        # 2. Stabilizer product operators for (k % 2 = flag) sites, with readout errors, Stab_prod_arr[flag][1]
        Stab_prod_arr[flag][1] += Stab_prod_func( Output, n, flag )

        if (Correct_Mode == "Both") or (Correct_Mode == "Exact"):
            # 3. Stabilizer product operators for (k % 2 = flag) sites, exactly correcting readout error, Stab_prod_arr[flag][2]
            Stab_prod_arr[flag][2] += Stab_prod_exact_correct_func( Output, n, readout_error_0, readout_error_1, one_index )
        
        if (Correct_Mode == "Both") or (Correct_Mode == "Approx"):
            # 4. Stabilizer product operators for (k % 2 = flag) sites, approximately correcting readout error by the factorization approximation, Stab_prod_arr[flag][-1]
            Stab_prod_arr[flag][-1] += Stab_prod_approx_correct_func( Output, n, flag, check_table )
        
# Save the final results
for flag in range(0, 2, 1): # k % 2 = flag
    file_name = "data/1D_cluster_state_n={}_{}mod2_{}".format(n, flag, Correct_Mode)
    with open(file_name + ".csv", "a") as opt:
        np.savetxt(opt, Stab_prod_arr[flag] / Num_ensemble )

print(Stab_prod_arr / Num_ensemble)