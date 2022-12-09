import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sci

def create_v_vector(N,x):
    v_vector = -1.0/x
    return v_vector

def create_a_matrix(x,h):
    N = len(x)
    amatrix = np.zeros([N,N])
    beta0 = h*h/6.0
    beta1 = -5.0*beta0
    vvector = create_v_vector(N,x) 
    amatrix[0,0] = -2.0*(1.0-beta1*vvector[0])
    amatrix[0,1] = 1.0-beta0*vvector[1]
    amatrix[N-1,N-2] = 1.0-beta0*vvector[N-2]
    amatrix[N-1,N-1] = -2.0*(1.0-beta1*vvector[N-1])
    for i in range(1,N-1):
        amatrix[i,i-1] = 1.0-beta0*vvector[i-1]
        amatrix[i,i] = -2.0*(1.0-beta1*vvector[i])
        amatrix[i,i+1] = 1.0-beta0*vvector[i+1]
    return amatrix 

def create_inverse_b_matrix(x,h):
    N = len(x)
    invbmatrix = np.zeros([N,N])
    beta0 = h*h/6.0
    beta1 = -5.0*beta0
    invbmatrix[0,0] = 2.0*beta1
    invbmatrix[0,1] = -beta0
    invbmatrix[N-1,N-2] = -beta0
    invbmatrix[N-1,N-1] = 2.0*beta1
    for i in range(1,N-1):
        invbmatrix[i,i-1] = -beta0
        invbmatrix[i,i] = 2.0*beta1
        invbmatrix[i,i+1] = -beta0

    return sci.linalg.inv(invbmatrix)

x_array,h = np.linspace(0.0001,50.0,8000,retstep = True)
print("h is: ",h)
a = create_a_matrix(x_array,h)
print("A created")
binv = create_inverse_b_matrix(x_array,h)
print("B created")
bigmatrix = np.matmul(binv,a)
print("Big matrix created")
eigenvalues = sci.linalg.eigvals(bigmatrix)
print("eigenvalues found")
eigenvalues = np.sort(eigenvalues)
print(eigenvalues[0:10])