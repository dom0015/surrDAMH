from collections import namedtuple, Iterable
from libcpp cimport bool

from libc.stdint cimport uintptr_t
from cpython cimport PyLong_AsVoidPtr, PyLong_FromVoidPtr

cimport mpi4py.MPI as MPI
#from mpi4py.mpi_c cimport *
from mpi4py.libmpi cimport *

cdef extern from "mpi_stats.hpp":
    bool check_mpi()
    int get_rank()
    int get_comm_size()
    int thread_level()
    bool has_thread_multiple()
    int test()
    void* inicializace()
    void printni(void* pole)
    void sayhello(MPI_Comm)
    void setAAA()
    void printAAA()

#cdef MPI_Comm communicator
#sayhello(&communicator)

#cdef MPI.Comm comm = MPI.Comm()
#comm.ob_mpi = communicator

MPIStats = namedtuple("MPIStats", ["check_mpi", "rank", "size",
                                   "thread_level", "has_thread_multiple"])

def mpi_stats():
    return MPIStats(check_mpi(), get_rank(), get_comm_size(),
                    thread_level(), has_thread_multiple())

def mpi_test():
    return test()

def Inicializace( ):
    return PyLong_FromVoidPtr(inicializace())

def Printni( uintptr_t pole ):
    printni( PyLong_AsVoidPtr(pole))
    return

def Sayhello( MPI.Comm C ):
    sayhello(C.ob_mpi)
    return

def SetAAA():
    setAAA()
    return

def PrintAAA():
    printAAA()
    return