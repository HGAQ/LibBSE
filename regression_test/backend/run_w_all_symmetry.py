"""Real Si 333 regression: representative W + Delta versus un-reduced W."""
import argparse
from pathlib import Path
import pprint
import time

import numpy as np

from compare_outputs import compare_eigenpairs, read_eigenpairs
from run_case import file_digest, input_manifest, run_bse

HA_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903


def validate_fixture(fixture_root):
    symmetry_manifest = fixture_root / 'aims_all/aims_symmetry_all.dat'
    symmetry_lines = symmetry_manifest.read_text().splitlines()
    if (symmetry_lines[0].split() != ['AIMS_SYMMETRY_ALL', '1']
            or symmetry_lines[1].split() != ['694', '27']
            or symmetry_lines[2].split() != ['FULL_W', 'D_W_DAGGER']):
        raise ValueError('Expected the real Si 333, 694-auxiliary-function FULL_W fixture')
    symmetry_mappings = [line.split() for line in symmetry_lines[3:] if line.strip()]
    if len(symmetry_mappings) != 27:
        raise ValueError('Expected 27 target q points')
    representative_ids = {int(mapping[1]) for mapping in symmetry_mappings}
    if len(representative_ids) != 4:
        raise ValueError('Expected four representatives, not a full-grid identity mapping')
    for symmetry_mapping in symmetry_mappings:
        if len(symmetry_mapping) != 11:
            raise ValueError('Invalid symmetry mapping')
        if not (symmetry_manifest.parent / symmetry_mapping[-1]).is_file():
            raise FileNotFoundError(symmetry_mapping[-1])
    if (fixture_root / 'aims_none/aims_symmetry_all.dat').exists():
        raise ValueError('The un-reduced reference must not use symmetry restoration')
    return symmetry_manifest


def dielectric(output_root, geometry_path):
    lattice_vectors = []
    for geometry_line in geometry_path.read_text().splitlines():
        geometry_tokens = geometry_line.split()
        if geometry_tokens and geometry_tokens[0] == 'lattice_vector':
            lattice_vectors.append([float(component) for component in geometry_tokens[1:4]])
    if np.shape(lattice_vectors) != (3, 3):
        raise ValueError('Expected three lattice vectors in Angstrom')
    volume_bohr3 = abs(np.linalg.det(np.asarray(lattice_vectors) / BOHR_ANGSTROM))
    momentum_strength = np.loadtxt(output_root / 'momentum_strength_singlet_tda.dat')
    if (momentum_strength.shape != (324, 5) or not np.isfinite(momentum_strength).all()
            or np.any(momentum_strength[:, 1] <= 0) or volume_bohr3 <= 0):
        raise ValueError('Invalid momentum strengths, excitation energies or cell volume')
    excitation_ev = momentum_strength[:, 1]
    photon_ev = np.linspace(0, 12, 4801)
    # Spin-degenerate singlet, per primitive cell: f = 4 |P|^2 / (Nk Omega_Ha).
    oscillator_per_cell = 4 * momentum_strength[:, 2:5] / (27 * excitation_ev[:, None] / HA_EV)
    lorentz_kernel = 0.10 / np.pi / ((photon_ev[:, None] - excitation_ev[None, :])**2 + 0.10**2)
    dielectric_xyz = (2 * np.pi**2 / volume_bohr3 * HA_EV**2
                      * (lorentz_kernel @ (oscillator_per_cell / excitation_ev[:, None])))
    if not np.isfinite(dielectric_xyz).all():
        raise ValueError('Non-finite dielectric response')
    return photon_ev, dielectric_xyz


def compare_runs(reference_root, candidate_root, geometry_path, work_root, name):
    reference_energies, reference_amplitudes = read_eigenpairs(reference_root)
    candidate_energies, candidate_amplitudes = read_eigenpairs(candidate_root)
    if (reference_energies.shape != (324,) or reference_amplitudes.shape != (324, 324)
            or candidate_energies.shape != (324,) or candidate_amplitudes.shape != (324, 324)):
        raise ValueError('Expected all 324 Si 333 4v3c eigenpairs')
    eigenpair_metrics = compare_eigenpairs(reference_energies, reference_amplitudes,
                                          candidate_energies, candidate_amplitudes)
    photon_ev, reference_dielectric = dielectric(reference_root, geometry_path)
    _, candidate_dielectric = dielectric(candidate_root, geometry_path)
    dielectric_norms = np.linalg.norm(reference_dielectric, axis=0)
    if np.any(dielectric_norms <= 0):
        raise ValueError('Zero reference dielectric spectrum')
    dielectric_relative_errors = (np.linalg.norm(candidate_dielectric - reference_dielectric, axis=0)
                                   / dielectric_norms)
    # Check every direction so averaging cannot cancel a symmetry error.
    comparison_metrics = {
        'eigenpairs': eigenpair_metrics,
        'dielectric_relative_error_xyz': dielectric_relative_errors.tolist(),
        'dielectric_relative_tolerance': 2e-5,
        'pass': bool(eigenpair_metrics['pass'] and np.all(dielectric_relative_errors <= 2e-5)),
    }
    np.savetxt(work_root / (name + '_dielectric.dat'),
               np.column_stack((photon_ev, reference_dielectric, candidate_dielectric)),
               header='energy_eV ref_eps2_x ref_eps2_y ref_eps2_z test_eps2_x test_eps2_y test_eps2_z')
    return comparison_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--ranks', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    fixture_root = args.fixtures.resolve()
    validate_fixture(fixture_root)
    work_root = (args.work / f'w_all_symmetry_mpi{args.ranks}_{time.time_ns()}').resolve()
    work_root.mkdir(parents=True)
    common_options = {
        'input_dir': str(fixture_root / 'aims_export'), 'input_format': 'fhi_aims',
        'qp_data': str(fixture_root / 'aims_none/energy_qp'), 'qp_format': 'energy_qp',
        'screened_format': 'fhi_aims_w', 'output_dir': 'libbse.d',
        'nocc': '4', 'nvirt': '3', 'bse_nstates': '-1', 'bse_solver': 'elpa',
        'bse_spin_types': 'singlet', 'bse_tda': 'tda', 'bse_ri_hartree': '1',
        'bse_use_fine_kgrid': '0', 'abs_gauge': 'velocity', 'wavefunction_gauge': 'native',
        'out_bse_eigenvectors': 'true', 'bse_plasma_energy_ev': '0',
        'spectrum_broadening_ev': '0.10', 'spectrum_energy_step_ev': '0.01',
        'spectrum_energy_min_ev': '0', 'spectrum_energy_max_ev': '12',
    }
    # Both paths use the same executable and KS/QP inputs; only the W source differs.
    fixture_hashes = {}
    for symmetry_mode in ('none', 'all'):
        run_options = dict(common_options)
        run_options['screened_dir'] = str(fixture_root / ('aims_' + symmetry_mode))
        fixture_hashes.update(input_manifest(run_options))
        run_bse(args.executable.resolve(), work_root / symmetry_mode, run_options, args.ranks)
    geometry_path = fixture_root / 'aims_export/geometry.in'
    comparison_metrics = {
        'all_vs_none': compare_runs(work_root / 'none/libbse.d', work_root / 'all/libbse.d',
                                   geometry_path, work_root, 'all_vs_none'),
        # A frozen independent result also catches regressions shared by both input paths.
        'none_vs_frozen': compare_runs(fixture_root / 'bse_none/libbse.d', work_root / 'none/libbse.d',
                                      geometry_path, work_root, 'none_vs_frozen'),
    }
    fixture_hashes.update({str(path): file_digest(path)
                           for path in (fixture_root / 'bse_none/libbse.d').glob('*.dat')})
    (work_root / 'inputs.sha256').write_text(''.join(
        f'{digest}  {path}\n' for path, digest in sorted(fixture_hashes.items())))
    (work_root / 'comparison.txt').write_text(pprint.pformat(comparison_metrics) + '\n')
    print(f'W all symmetry: 4 representatives -> 27 q points; ranks={args.ranks}; {work_root}')
    for comparison_name, comparison_entry in comparison_metrics.items():
        print(comparison_name, 'PASS' if comparison_entry['pass'] else 'FAIL',
              'max energy error (eV):', comparison_entry['eigenpairs']['max_energy_error_eV'],
              'relative dielectric error xyz:', comparison_entry['dielectric_relative_error_xyz'])
    return 0 if all(entry['pass'] for entry in comparison_metrics.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
