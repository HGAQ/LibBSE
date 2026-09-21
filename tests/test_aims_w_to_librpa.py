#!/usr/bin/env python3
"""Independent Fourier sign/normalization and real-cell ordering checks."""
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from aims_w_to_librpa import complete_inverse, to_real_space
for n in (3,4):
    mesh=(n,1,1)
    matrices={}
    for iq in range(n//2+1):
        q=iq/n
        matrices[iq,0,0]=np.array([[2+np.cos(2*np.pi*q),1j*np.sin(2*np.pi*q)],[-1j*np.sin(2*np.pi*q),4]])
    wr=to_real_space(complete_inverse(matrices,mesh))[:,0,0]
    cells=np.arange(-(n//2),(n-1)//2+1)
    for q in np.arange(5)/5:
        value=np.einsum('r,rij->ij',np.exp(2j*np.pi*q*cells),wr)
        expected=np.array([[2+np.cos(2*np.pi*q),1j*np.sin(2*np.pi*q)],[-1j*np.sin(2*np.pi*q),4]])
        np.testing.assert_allclose(value,expected,atol=1e-13)
try:
    complete_inverse({(0,0,0):np.eye(2)},(3,1,1))
except ValueError:
    pass
else:
    raise AssertionError('missing q coverage accepted')
print('Python W Fourier tests passed')
