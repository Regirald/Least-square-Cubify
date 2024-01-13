from mesh import Mesh
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsmr
from ifp1.attributes import *

model = "ifp1"
m = Mesh(model + "/slice.obj")
nboundary = sum(m.boundary)

print(nboundary, m.ncorners, m.nverts, np.max(fault_opposite), np.count_nonzero(is_fault))



for dim in range(2): # solve for x first, then for y
    A = scipy.sparse.lil_matrix((m.ncorners*4 , m.nverts))
    b = [0]*A.shape[0]

    if dim==1:
        L= [[0, 0, i] for i in range(np.max(horizon_id)+1)]
        for row in range(m.ncorners):
            if horizon_id[row]!=-1:
                L[horizon_id[row]][0]+=1
                L[horizon_id[row]][dim]+= (m.V[m.dst(row)][dim]+m.V[m.org(row)][dim])/2

        Mean_horizon=[L[i][1]/L[i][0] for i in range(len(L))]
        print(Mean_horizon)


    for row in range(m.ncorners):
        i = m.org(row)
        j = m.dst(row)
        A[row, i] = -1
        A[row, j] =  1
        b[row] = m.V[j][dim]-m.V[i][dim]

        if horizon_id[row]!=-1 and dim==1:  # flatten the right dimension of each half-edge
            A[row+m.ncorners, i] = 1*100
            A[row+m.ncorners, j] = 1*100
            b[row+m.ncorners] = 2*Mean_horizon[horizon_id[row]]*100
        
        if is_fault[row]:
            row2=fault_opposite[row]
            i_opp=m.org(row2)
            j_opp=m.dst(row2)

            A[row+m.ncorners*2, i_opp] = -1
            A[row+m.ncorners*2, i] = 1
            A[row+m.ncorners*2, j_opp] = -1
            A[row+m.ncorners*2, j] = 1
            b[row+m.ncorners*2] = 0

            if dim ==0:
                A[row+m.ncorners, i] = -1*100
                A[row+m.ncorners, j] = 1*100
                b[row+m.ncorners] = 0
            
 

            
    # for row in range(m.ncorners):
    #     i = m.org(row)
    #     j = m.dst(row)

    #     if m.on_border(i) and not(is_fault[row]):
    #         A[row+m.ncorners*3, i] = 1*100 # quadratic penalty to lock boundary vertices
    #         A[row+m.ncorners*3, j] = 1*100
    #         b[row+m.ncorners*3] = (m.V[i][dim]+m.V[j][dim]) *100

    
    # l=[]
    # for (i,v) in enumerate(m.V):
        
    #     fault_id=m.v2c[i]
    #     if fault_opposite[fault_id]!=-1:
    #         l.append(fault_opposite[fault_id]) 

    #     if m.on_border(i) :
            
    #         A[i+m.ncorners*3, i] = 1 # quadratic penalty to lock boundary vertices
    #         b[i+m.ncorners*3] = v[dim] 

    # print(np.sort(l))


    A = A.tocsr() # convert to compressed sparse row format for faster matrix-vector muliplications
    x = lsmr(A, b)[0] # call the least squares solver
    for i in range(m.nverts): # apply the computed flattening
        m.V[i][dim] = x[i]

m.write_vtk("output_cubi.vtk")
print("done.")
#print(m) # output the deformed mesh

