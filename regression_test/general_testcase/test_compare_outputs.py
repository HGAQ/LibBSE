"""Check that regression comparisons accept gauge freedom and detect real changes."""
import unittest
import tempfile
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import numpy as np
from compare_outputs import compare_eigenpairs,compare_screening
from run_case import screening_to_cache

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

class FullWCacheTests(unittest.TestCase):
    def test_static_cache_matches_cpp_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sqrt_cut = np.diag([2., 3.]).astype(np.complex128)
            inverse = np.array([[.5, .1j], [-.1j, .8]], dtype=np.complex128)
            with (root / 'screening_q_0.bin').open('wb') as handle:
                np.array([2], dtype=np.int32).tofile(handle)
                np.zeros(4, dtype=np.float64).tofile(handle)
                sqrt_cut.tofile(handle)
                inverse.tofile(handle)
            cache = root / 'cache'
            result = screening_to_cache(root, cache, 0.)
            self.assertEqual(result['format'], 'full_w_binary')
            data = (cache / 'full_w_q_000001.bin').read_bytes()
            # Check the fixed header offsets used by the C++ reader.
            self.assertEqual(len(data), 64 + 4 * 16)
            self.assertEqual(data[:16], b'LIBBSE_FULL_W_V1')
            self.assertEqual(struct.unpack_from('=4i', data, 16), (2, 1, 1, 1))
            self.assertEqual(struct.unpack_from('=4d', data, 32), (0., 0., 0., 0.))
            matrix = np.frombuffer(data, dtype=np.complex128, offset=64).reshape(2, 2)
            np.testing.assert_allclose(matrix, sqrt_cut @ inverse @ sqrt_cut,
                                       rtol=0, atol=1.e-14)

if __name__=='__main__':
    unittest.main()
