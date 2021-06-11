# try: range = xrange
# except: pass

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import pcdeflation as pcdeflation

# grid size and spacing
m, n  = 32, 32
hx = 1.0/(m-1)
hy = 1.0/(n-1)

# create sparse matrix
A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes([m*n, m*n])
A.setType('aij') # sparse
A.setPreallocationNNZ(5)

# precompute values for setting
# diagonal and non-diagonal entries
diagv = 2.0/hx**2 + 2.0/hy**2
offdx = -1.0/hx**2
offdy = -1.0/hy**2

# loop over owned block of rows on this
# processor and insert entry values
Istart, Iend = A.getOwnershipRange()
for I in range(Istart, Iend) :
    A[I,I] = diagv
    i = I//n    # map row number to
    j = I - i*n # grid coordinates
    if i> 0  : J = I-n; A[I,J] = offdx
    if i< m-1: J = I+n; A[I,J] = offdx
    if j> 0  : J = I-1; A[I,J] = offdy
    if j< n-1: J = I+1; A[I,J] = offdy
A.assemblyBegin()
A.assemblyEnd()

x,b = A.createVecs()
x.set(0)
b.set(1)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
pc = ksp.getPC()
pc.setType('deflation')
ksp.setFromOptions()

nrows = b.getSize()

dense = True
transpose = True

W = PETSc.Mat()
W.create(PETSc.COMM_WORLD)

if transpose:
    W.setSizes([1,nrows])
else:
    W.setSizes([nrows,1])

if dense:
    W.setType('dense')
else:
    W.setType('aij')
    W.setPreallocationNNZ(1)

W.setUp()
    
if transpose:
    W[0,nrows//2] = 1.0
else:
    W[nrows//2,0] = 1.0 # std basis deflation using a single vector somewhere in the middle

W.assemblyBegin()
W.assemblyEnd()
pcdeflation.setDeflationMat(pc,W,transpose);

ksp.solve(b, x)

ksp.view()

