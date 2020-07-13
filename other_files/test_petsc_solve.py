#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:19:56 2020

@author: domesova
"""

import numpy as np
from petsc4py import PETSc

dof = 10

M0 = np.eye(dof)
b0 = np.ones((dof,))
x0 = np.linalg.solve(M0,b0)
print(x0)

M = PETSc.Mat().create()
M.setSizes([dof, dof])
M.setType("aij")
M.setUp()
M.setPreallocationNNZ(10)

b = PETSc.Vec().create()
b.setSizes(dof)
b.setType("mpi")
b.set(0)
b.setUp()

x = PETSc.Vec().create()
x.setSizes(dof)
x.setType("mpi")
x.set(0)
x.setUp()

for i in range(dof):
    M.setValues
    M.setValues(i, i, 1, 2)
    b.setValues(i, 1, 2)

M.assemble()
b.assemble()
x.assemble()

M.mult(b,x)
print(np.array(b))
print(np.array(x))

ksp = PETSc.KSP().create()
ksp.setOperators(M)
ksp.setType('cg')
ksp_pc = ksp.getPC()
ksp_pc.setType("none")
ksp.setFromOptions()
# ksp.setUp()
ksp.solve(b, x)
print(np.array(x))
