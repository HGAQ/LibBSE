"""Check that regression comparisons accept gauge freedom and detect real changes."""
import unittest
import numpy as np
from compare_outputs import compare_eigenpairs,compare_screening

class EigenpairComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference_energies=np.array([1.,2.,2.,4.])
        self.reference_amplitudes=np.eye(4,dtype=complex)

    def test_phase_and_degenerate_rotation(self):
        candidate_amplitudes=self.reference_amplitudes.copy()
        candidate_amplitudes[:,0]*=np.exp(.7j)
        candidate_amplitudes[:,1:3]=candidate_amplitudes[:,1:3]@np.array([[1,1j],[1j,1]])/np.sqrt(2)
        comparison=compare_eigenpairs(self.reference_energies,self.reference_amplitudes,self.reference_energies,candidate_amplitudes)
        self.assertTrue(comparison['pass'])

    def test_energy_change_fails(self):
        candidate_energies=self.reference_energies.copy()
        candidate_energies[0]+=.001
        self.assertFalse(compare_eigenpairs(self.reference_energies,self.reference_amplitudes,candidate_energies,self.reference_amplitudes)['pass'])

    def test_mixing_distinct_energies_fails(self):
        candidate_amplitudes=self.reference_amplitudes.copy()
        candidate_amplitudes[:,[0,3]]=candidate_amplitudes[:,[0,3]]@np.array([[1,1],[-1,1]])/np.sqrt(2)
        self.assertFalse(compare_eigenpairs(self.reference_energies,self.reference_amplitudes,self.reference_energies,candidate_amplitudes)['pass'])

    def test_normalization_change_fails(self):
        self.assertFalse(compare_eigenpairs(self.reference_energies,self.reference_amplitudes,self.reference_energies,2*self.reference_amplitudes)['pass'])

    def test_duplicate_degenerate_vectors_fail(self):
        candidate_amplitudes=self.reference_amplitudes.copy()
        candidate_amplitudes[:,2]=candidate_amplitudes[:,1]
        self.assertFalse(compare_eigenpairs(self.reference_energies,self.reference_amplitudes,self.reference_energies,candidate_amplitudes)['pass'])

    def test_W_change_and_missing_q_fail(self):
        reference_screening={(0.,0.,0.,0.):(np.eye(2),np.diag([.5,.8]))}
        candidate_screening={(0.,0.,0.,0.):(np.eye(2),np.diag([.6,.8]))}
        self.assertFalse(compare_screening(reference_screening,candidate_screening)['pass'])
        with self.assertRaises(ValueError): compare_screening(reference_screening,{})

if __name__=='__main__':
    unittest.main()
