from petsc4py.PETSc cimport Mat,  PetscMat
from petsc4py.PETSc cimport PC, PetscPC

from petsc4py.PETSc import Error

cdef extern from "petscpc.h":
    int PCDeflationSetSpace(PetscPC pc,PetscMat W,int transpose)

def setDeflationMat(PC pc,Mat mat, transpose=False):
    cdef int ierr
    ierr = PCDeflationSetSpace(pc.pc,mat.mat,transpose)
    if ierr != 0: raise Error(ierr)

