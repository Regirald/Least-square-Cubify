from mesh import Mesh
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsmr
from ifp1.attributes import *

model = "ifp1"
m = Mesh(model + "/slice.obj")
#attr = importlib.import_module(model+ "/attributes")
nboundary = sum(m.boundary)

print(nboundary, m.ncorners, m.nverts, len(fault_opposite))

def nearest_axis(n):
    return np.argmax([np.abs(np.dot(n, a)) for a in [[1,0],[0,1]]])

for dim in range(2): # solve for x first, then for y
    A = scipy.sparse.lil_matrix((m.ncorners*2, m.nverts))
    # b = [0]*(m.ncorners*2)

    # for c in range(m.ncorners):
    #     b[c] = m.V[m.dst(c)][dim]-m.V[m.org(c)][dim]

    #     if is_fault[c] and dim ==1 :
    #         b[m.ncorners+c] = 0
    #     else:
    #         b[m.ncorners+c] = m.V[m.dst(c)][dim]-m.V[m.org(c)][dim]

    b = [m.V[m.dst(c)][dim]-m.V[m.org(c)][dim] for c in range(m.ncorners)] + [0]*m.ncorners

    for row in range(m.ncorners):
        i = m.org(row)
        j = m.dst(row)
        A[row, j] =  1
        A[row, i] = -1

        if is_fault[row] and dim==1:  # flatten the right dimension of each half-edge
            A[row+m.ncorners, i] = -2
            A[row+m.ncorners, j] =  2


    # row=0
    # for (i,v) in enumerate(m.V):
    #     if m.on_border(i):
    #         A[row, i] = 1*100 # quadratic penalty to lock boundary vertices
    #         b[row] = v[dim] *100
    #         row += 1
    A = A.tocsr() # convert to compressed sparse row format for faster matrix-vector muliplications
    x = lsmr(A, b)[0] # call the least squares solver
    for i in range(m.nverts): # apply the computed flattening
        m.V[i][dim] = x[i]

m.write_vtk("output_cubi.vtk")
print("done.")
#print(m) # output the deformed mesh

