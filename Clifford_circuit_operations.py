import numpy as np

# Clifford circuit operations
######################################################################################              
# Hadamard gate on qubit a ∈ {0, . . ., n-1}
# For all i ∈ {0, . . ., 2n-1}, set ri:= ri⊕x_ia * z_ia and swap x_ia with z_ia
def Hadamard( Stab_mat, n, a ):
    for i in range(0, 2*n):
        Stab_mat[i][-1] = (Stab_mat[i][-1] + Stab_mat[i][a] * Stab_mat[i][n+a])%2
        
        tmp = Stab_mat[i][a] # x_ia
        Stab_mat[i][a] = Stab_mat[i][n+a] # z_ia
        Stab_mat[i][n+a] = tmp
    
    return Stab_mat

# Phase on qubit a ∈ {0, . . ., n-1}
# For all i ∈ {0, . . ., 2n-1}, set ri:=ri⊕x_ia * z_ia and then set z_ia := z_ia ⊕ x_ia
def Phase( Stab_mat, n, a ):
    for i in range(0, 2*n):
        Stab_mat[i][-1] = (Stab_mat[i][-1] + Stab_mat[i][a] * Stab_mat[i][n+a])%2
        
        Stab_mat[i][n+a] = ( Stab_mat[i][n+a] + Stab_mat[i][a] )%2 # z_ia
    
    return Stab_mat

# CNOT from control a to target b
# For all i ∈ {0, . . ., 2n-1}, set ri:= ri ⊕ xia * zib (xib ⊕ zia ⊕ 1), 
# xib :=xib ⊕ xia, and zia := zia ⊕ zib
def CNOT( Stab_mat, n, a, b ):
    for i in range(0, 2*n):
        Stab_mat[i][-1] = ( Stab_mat[i][-1] + Stab_mat[i][a] * Stab_mat[i][n+b] 
                           * ( (1 + Stab_mat[i][b] + Stab_mat[i][n+a])%2 ) )% 2
        
        Stab_mat[i][b] = (Stab_mat[i][b] + Stab_mat[i][a]) % 2 # x_ib
        Stab_mat[i][n+a] = (Stab_mat[i][n+a] + Stab_mat[i][n+b]) % 2 # z_ia
    
    return Stab_mat

# g(x1,z1,x2,z2) function
def g_func(x1, z1, x2, z2):
    if (x1 == 0) and (z1 == 0):
        return 0
    if (x1 == 1) and (z1 == 1): 
        return z2 - x2
    if (x1 == 1) and (z1 == 0):
        return z2 * (2 * x2 - 1)
    if (x1 == 0) and (z1 == 1): 
        return x2 * (1 - 2 * z2)

# rowsum(h,i), as representing the group multiplication of Pauli operators
def rowsum(Stab_mat, n, h, i):
    # For all j ∈ {0, . . ., n-1}, set xhj := xij ⊕ xhj and set zhj := zij ⊕ zhj
    for j in range(0, n):
        Stab_mat[h][j] = (Stab_mat[i][j] + Stab_mat[h][j]) % 2
        Stab_mat[h][n+j] = (Stab_mat[i][n+j] + Stab_mat[h][n+j]) % 2
    # rh = 0 if 2rh + 2ri + sum j=0^n-1 g(xij , zij , xhj , zhj ) ≡ 0 (mod 4)
    factor_i = 2 * (Stab_mat[h][-1] + Stab_mat[i][-1])
    for j in range(0, n):
        factor_i += g_func(Stab_mat[i][j], Stab_mat[i][n+j], Stab_mat[h][j], Stab_mat[h][n+j])
    
    if (factor_i % 4 == 0):
        Stab_mat[h][-1] = 0
    if (factor_i % 4 == 2):
        Stab_mat[h][-1] = 1
        
    return Stab_mat

# Measurement of qubit a in standard basis. 
def Measurement( Stab_mat, n, a ):
# First check whether there exists a p ∈ {n , . . ., 2n-1} such that x_pa = 1.
    p = 0
    for i in range(n, 2*n):
        if (Stab_mat[i][a] == 1):
            p = i
            break
    
    if (p!=0): # Case I: Such a p exists (if more than one exists, then let p be the smallest).
        # In this case the measurement outcome is random, so the state needs to be updated. This is done as follows.
        # First call rowsum (i, p) for all i ∈ {0, . . ., 2n-1} such that i != p and x_ia = 1. 
        for i in range(0, 2*n):
            if (i != p) and (Stab_mat[i][a] == 1):
                Stab_mat = rowsum(Stab_mat, n, i, p)
        # Second, set entire the (p − n)th row equal to the pth row. 
        Stab_mat[p-n] = Stab_mat[p]
        # Third, set the pth row to be identically 0, except that r_p is 0 or 1 with equal probability, 
        # and z_pa = 1. 
        Stab_mat[p] = np.zeros(2*n+1, dtype="int8")
        Stab_mat[p][n+a] = 1
        Stab_mat[p][-1] = np.random.randint(0, 2)
        
        # Finally, return r_p as the measurement outcome
        return [Stab_mat, Stab_mat[p][-1]]
    
    if (p == 0): # Case II: Such an p does not exist. 
        # In this case the outcome is determinate, so measuring the state will not change it;
        # the only task is to determine whether 0 or 1 is observed. This is done as follows. 
        # First set the 2n_th row to be identically 0. 
        Stab_mat[2*n] = np.zeros(2*n+1, dtype="int8")
        # Second, call rowsum (2n, i + n) for all i ∈ {0, . . ., n-1} such that xia = 1. 
        for i in range(0, n):
            if (Stab_mat[i][a] == 1):
                Stab_mat = rowsum(Stab_mat, n, 2*n, i + n)
        # Finally return r_2n as the measurement outcome
        return [Stab_mat, Stab_mat[2*n][-1]]
    
# Z-gate on qubit a ∈ {0, . . ., n-1}
# Z = Phase^2
def Zgate( Stab_mat, n, a ):
    Stab_mat = Phase( Stab_mat, n, a )
    Stab_mat = Phase( Stab_mat, n, a )
    return Stab_mat

# X-gate on qubit a ∈ {0, . . ., n-1}
# X = H Z H
def Xgate( Stab_mat, n, a ):
    Stab_mat = Hadamard( Stab_mat, n, a )
    Stab_mat = Zgate( Stab_mat, n, a )
    Stab_mat = Hadamard( Stab_mat, n, a )
    return Stab_mat

# Y-gate on qubit a ∈ {0, . . ., n-1}
# Y = i X Z
def Ygate( Stab_mat, n, a ):
    Stab_mat = Zgate( Stab_mat, n, a )
    Stab_mat = Xgate( Stab_mat, n, a )
    return Stab_mat

# Single-qubit noise channel, operation_num = 0 for I, 1 for X, 2 for Y, 3 for Z
def single_qubit_noise( Stab_mat, n, a, operation_num ):
    if (operation_num == 0):
        return Stab_mat
    if (operation_num == 1):
        return Xgate( Stab_mat, n, a )
    if (operation_num == 2):
        return Ygate( Stab_mat, n, a )
    if (operation_num == 3):
        return Zgate( Stab_mat, n, a )

# Two-qubit noise channel, operation_num = 0 for II, 1 for IX, 2 for IY, 3 for IZ, 4 for XI, 5 for XX
# 6 for XY, 7 for XZ, 8 for YI, 9 for YX, 10 for YY, 11 for YZ, 12 for ZI, 13 for ZX, 14 for ZY, 15 for ZZ
def two_qubit_noise( Stab_mat, n, a, b, operation_num ):
    if (operation_num == 1): # IX
        Stab_mat = Xgate( Stab_mat, n, b )
    if (operation_num == 2): # IY
        Stab_mat = Ygate( Stab_mat, n, b )
    if (operation_num == 3): # IZ
        Stab_mat = Zgate( Stab_mat, n, b )
    if (operation_num == 4): # XI
        Stab_mat = Xgate( Stab_mat, n, a )
    if (operation_num == 5): # XX
        Stab_mat = Xgate( Stab_mat, n, a )
        Stab_mat = Xgate( Stab_mat, n, b )
    if (operation_num == 6): # XY
        Stab_mat = Xgate( Stab_mat, n, a )
        Stab_mat = Ygate( Stab_mat, n, b )
    if (operation_num == 7): # XZ
        Stab_mat = Xgate( Stab_mat, n, a )
        Stab_mat = Zgate( Stab_mat, n, b )
    if (operation_num == 8): # YI
        Stab_mat = Ygate( Stab_mat, n, a )
    if (operation_num == 9): # YX
        Stab_mat = Ygate( Stab_mat, n, a )
        Stab_mat = Xgate( Stab_mat, n, b )
    if (operation_num == 10): # YY
        Stab_mat = Ygate( Stab_mat, n, a )
        Stab_mat = Ygate( Stab_mat, n, b )
    if (operation_num == 11): # YZ
        Stab_mat = Ygate( Stab_mat, n, a )
        Stab_mat = Zgate( Stab_mat, n, b )
    if (operation_num == 12): # ZI
        Stab_mat = Zgate( Stab_mat, n, a )
    if (operation_num == 13): # ZX
        Stab_mat = Zgate( Stab_mat, n, a )
        Stab_mat = Xgate( Stab_mat, n, b )
    if (operation_num == 14): # ZY
        Stab_mat = Zgate( Stab_mat, n, a )
        Stab_mat = Ygate( Stab_mat, n, b )
    if (operation_num == 15): # ZZ
        Stab_mat = Zgate( Stab_mat, n, a )
        Stab_mat = Zgate( Stab_mat, n, b )
    return Stab_mat
######################################################################################  