"""Restore and independently check the bundled Si 222 full-symmetry W fixture."""
from pathlib import Path
import numpy as np


def load_w_operators(screening_directory):
    w_operators = {}
    w_frequencies = {}
    for compact_path in sorted(screening_directory.glob('periodic_gw_w_*.npz')):
        with np.load(compact_path, allow_pickle=False) as compact_w:
            w_entries = compact_w['entries']
            w_header = compact_w['header']
        dimension_line = next(line for line in w_header if 'n_basbas n_freq q_index:' in line)
        naux, nfreq, q_id = map(int, dimension_line.split(':')[1].split())
        if naux != 218 or nfreq != 1 or w_entries.shape != (naux * naux, 6):
            raise ValueError('Invalid Si 222 W dimensions/frequency count')
        if not np.isfinite(w_entries).all() or not np.all(w_entries[:, 0] == 1):
            raise ValueError('Invalid W entries/frequency index')
        w_indices = w_entries[:, 2:4].astype(int) - 1
        if (not np.array_equal(w_indices + 1, w_entries[:, 2:4])
                or np.any(w_indices < 0) or np.any(w_indices >= naux)
                or len(np.unique(w_indices[:, 0] * naux + w_indices[:, 1])) != naux * naux):
            raise ValueError('Invalid or duplicate W matrix indices')
        if q_id in w_operators or np.unique(w_entries[:, 1]).size != 1:
            raise ValueError('Duplicate W q point or mixed frequencies')
        w_operator = np.empty((naux, naux), dtype=complex)
        w_operator[w_indices[:, 0], w_indices[:, 1]] = w_entries[:, 4] + 1j * w_entries[:, 5]
        w_operators[q_id] = w_operator
        w_frequencies[q_id] = float(w_entries[0, 1])
    return w_operators, w_frequencies


def restore_and_check_symmetry(fixture_root):
    symmetry_directory = fixture_root / 'screening_all'
    symmetry_lines = (symmetry_directory / 'aims_symmetry_all.dat').read_text().splitlines()
    if (symmetry_lines[0].split() != ['AIMS_SYMMETRY_ALL', '1']
            or symmetry_lines[1].split() != ['218', '8']
            or symmetry_lines[2].split() != ['FULL_W', 'D_W_DAGGER']):
        raise ValueError('Expected Si 222 full-symmetry manifest')
    symmetry_mappings = [line.split() for line in symmetry_lines[3:] if line.strip()]
    if len(symmetry_mappings) != 8 or any(len(mapping) != 11 for mapping in symmetry_mappings):
        raise ValueError('Expected eight complete q mappings')
    w_reference, reference_frequencies = load_w_operators(fixture_root / 'screening')
    w_representatives, representative_frequencies = load_w_operators(symmetry_directory)
    representative_ids = {int(mapping[1]) for mapping in symmetry_mappings}
    if len(representative_ids) != 3 or representative_ids != set(w_representatives):
        raise ValueError('Expected three actual representatives, not full-grid identity mappings')
    target_ids = [int(mapping[0]) for mapping in symmetry_mappings]
    if len(set(target_ids)) != 8 or set(target_ids) != set(w_reference):
        raise ValueError('Missing or duplicate target q point')

    w_relative_errors = []
    for mapping in symmetry_mappings:
        target_id, representative_id, operation_id, conjugate_flag = map(int, mapping[:4])
        if operation_id < 1 or conjugate_flag not in (0, 1):
            raise ValueError('Invalid symmetry operation or conjugation flag')
        delta_filename = Path(mapping[-1])
        if delta_filename.name != mapping[-1]:
            raise ValueError('Delta must be in the screening directory')
        with np.load(symmetry_directory / delta_filename.with_suffix('.npz'), allow_pickle=False) as compact_delta:
            symmetry_delta = compact_delta['symmetry_delta']
            delta_header = compact_delta['header']
        expected_header = ['AIMS_DELTA 1', f'218 {target_id} {representative_id}']
        if ([line.split() for line in delta_header] != [line.split() for line in expected_header]
                or symmetry_delta.shape != (218, 218) or not np.isfinite(symmetry_delta).all()):
            raise ValueError('Invalid Delta metadata or entries')
        # Preserve float64/complex128 exactly when recreating the native text interface.
        delta_rows, delta_columns = np.indices(symmetry_delta.shape)
        delta_entries = np.column_stack((delta_rows.ravel() + 1, delta_columns.ravel() + 1,
                                        symmetry_delta.real.ravel(), symmetry_delta.imag.ravel()))
        np.savetxt(symmetry_directory / delta_filename, delta_entries,
                   fmt=['%d', '%d', '%.17e', '%.17e'], header='\n'.join(delta_header), comments='')
        delta_reloaded = np.loadtxt(symmetry_directory / delta_filename, skiprows=2)
        if not np.array_equal(delta_entries, delta_reloaded):
            raise ValueError('Delta text roundtrip changed entries')
        if abs(reference_frequencies[target_id] - representative_frequencies[representative_id]) > 1e-14:
            raise ValueError('W frequency changed between full and representative inputs')
        representative_w = w_representatives[representative_id]
        if conjugate_flag:
            representative_w = representative_w.conj()
        # Independent NumPy reconstruction checks every matrix before the C++ BSE path.
        restored_w = symmetry_delta @ representative_w @ symmetry_delta.conj().T
        w_relative_error = float(np.linalg.norm(restored_w - w_reference[target_id])
                                 / np.linalg.norm(w_reference[target_id]))
        if not np.isfinite(w_relative_error) or w_relative_error > 2e-8:
            raise ValueError(f'W restoration differs at q={target_id}: {w_relative_error}')
        w_relative_errors.append(w_relative_error)
    return {'representatives': 3, 'q_points': 8, 'relative_W_errors': w_relative_errors,
            'relative_W_tolerance': 2e-8}


def dielectric_spectrum(output_root, geometry_path):
    lattice_vectors = []
    for geometry_line in geometry_path.read_text().splitlines():
        geometry_tokens = geometry_line.split()
        if geometry_tokens and geometry_tokens[0] == 'lattice_vector':
            lattice_vectors.append([float(component) for component in geometry_tokens[1:4]])
    if np.shape(lattice_vectors) != (3, 3):
        raise ValueError('Expected three lattice vectors in Angstrom')
    volume_bohr3 = abs(np.linalg.det(np.asarray(lattice_vectors) / 0.529177210903))
    momentum_strength = np.loadtxt(output_root / 'momentum_strength_singlet_tda.dat')
    if (momentum_strength.shape != (32, 5) or not np.isfinite(momentum_strength).all()
            or np.any(momentum_strength[:, 1] <= 0) or volume_bohr3 <= 0):
        raise ValueError('Invalid Si 222 2v2c momentum strengths or cell volume')
    excitation_ev = momentum_strength[:, 1]
    photon_ev = np.linspace(0, 12, 4801)
    # Singlet per-cell convention: f = 4 |P|^2 / (Nk Omega_Ha), Nk = 8.
    oscillator_per_cell = 4 * momentum_strength[:, 2:5] / (8 * excitation_ev[:, None] / 27.211386245988)
    lorentz_kernel = 0.10 / np.pi / ((photon_ev[:, None] - excitation_ev[None, :])**2 + 0.10**2)
    dielectric_xyz = (2 * np.pi**2 / volume_bohr3 * 27.211386245988**2
                      * (lorentz_kernel @ (oscillator_per_cell / excitation_ev[:, None])))
    if not np.isfinite(dielectric_xyz).all():
        raise ValueError('Non-finite dielectric spectrum')
    return photon_ev, dielectric_xyz
