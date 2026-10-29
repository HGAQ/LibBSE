"""Run real Si fixtures against frozen references; never regenerate references in CI."""
from pathlib import Path
import ast
import pprint
import tomllib
import argparse
import hashlib
import tarfile
import time
import numpy as np
from compare_outputs import compare_directories, compare_eigenpairs, read_eigenpairs
from run_case import run_bse, screening_to_cache

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / 'physical_testcases/si222'


def extract_verified(archive_name, destination):
    archive_path = FIXTURE_ROOT / archive_name
    expected = ast.literal_eval((FIXTURE_ROOT / 'archives.txt').read_text())[archive_name]
    with archive_path.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'Fixture checksum mismatch: {archive_name}')
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile():
                raise ValueError(f'Invalid archive member: {member.name}')
        archive.extractall(destination, filter='data')


def restore_screening(fixture):
    for compact_path in (fixture / 'screening').glob('*.npz'):
        with np.load(compact_path, allow_pickle=False) as compact:
            entries = compact['entries']
            header = '\n'.join(str(line) for line in compact['header'])
        output_path = compact_path.with_suffix('.dat')
        np.savetxt(output_path, entries,
                   fmt=['%d', '%.17e', '%d', '%d', '%.17e', '%.17e'],
                   header=header, comments='')
        if not np.array_equal(entries, np.loadtxt(output_path)):
            raise ValueError('Screening text/binary roundtrip changed float64 entries')


def options_for(fixture, definition):
    options = {
        'input_dir': str(fixture / 'electronic'), 'input_format': 'fhi_aims',
        'qp_data': str(fixture / 'energy_qp'), 'qp_format': 'energy_qp',
        'screened_dir': str(fixture / 'screening'), 'output_dir': 'libbse.d',
        'nocc': '2', 'nvirt': '2', 'bse_nstates': '-1', 'bse_solver': 'elpa',
        'bse_spin_types': 'singlet', 'bse_tda': 'tda', 'bse_ri_hartree': '1',
        'bse_use_fine_kgrid': '0', 'abs_gauge': 'velocity',
        'wavefunction_gauge': 'native', 'chi0_coulomb_metric': 'full',
        'bse_memory_optimized': 'true', 'bse_ri_batch_blocks': '1',
        'bse_w_boundary_average': 'false', 'bse_plasma_energy_ev': '0',
        'out_bse_eigenvectors': 'true', 'out_screening_matrices': 'true',
        'spectrum_broadening_ev': '0.10', 'spectrum_energy_step_ev': '0.02',
        'spectrum_energy_min_ev': '0', 'spectrum_energy_max_ev': '12',
    }
    options.update(definition['options'])
    return options


def print_eigenstate_matches(title, eigenpair_metrics):
    print(f'\nEigenstate comparison: {title}', flush=True)
    print('Nondegenerate: Match = normalized |<reference|candidate>|^2.', flush=True)
    print('Degenerate: Match = smallest squared singular overlap of the two subspaces;', flush=True)
    print('the same group score is repeated for its members, not individual state overlaps.', flush=True)
    print(f"Degeneracy grouping: {eigenpair_metrics['degeneracy_eV']:.1e} eV; "
          f"energy tolerance: {eigenpair_metrics['energy_atol_eV']:.1e} eV.", flush=True)
    print(' State    E_ref (eV)    E_test (eV)       Delta (eV)       Match (%)    sin(theta)   Group', flush=True)
    for state in eigenpair_metrics['states']:
        group = str(state['group_first']) if state['group_first'] == state['group_last'] else f"{state['group_first']}-{state['group_last']}"
        print(f"{state['state']:6d} {state['reference_energy_eV']:13.8f} "
              f"{state['candidate_energy_eV']:14.8f} {state['energy_difference_eV']:+16.7e} "
              f"{100*state['match_squared']:15.9f} {state['subspace_sine']:13.5e}   {group}", flush=True)
    print(f"Eigenpair check: {'PASS' if eigenpair_metrics['pass'] else 'FAIL'}; "
          f"max |Delta E| = {eigenpair_metrics['max_energy_error_eV']:.7e} eV", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    definition = tomllib.loads((FIXTURE_ROOT / 'cases.toml').read_text())[args.case]
    work = (args.work / (args.case + '_' + str(time.time_ns()))).resolve()
    work.mkdir(parents=True)
    extract_verified('inputs.tar.xz', work / 'fixture')
    extract_verified('references.tar.gz', work / 'references')
    restore_screening(work / 'fixture')
    options = options_for(work / 'fixture', definition)
    run_bse(args.executable.resolve(), work / 'candidate', options, definition['ranks'])
    output = work / 'candidate/libbse.d'
    reference = work / 'references' / definition['reference']
    requires_screening = 'chi0' in options['screened_format']
    metrics = compare_directories(reference, output, requires_screening)
    print_eigenstate_matches(args.case + ' vs frozen reference', metrics['eigenpairs'])
    energies, amplitudes = read_eigenpairs(output)
    dimension = 8 * int(options['nocc']) * int(options['nvirt'])
    if len(energies) != dimension or amplitudes.shape != (dimension, dimension):
        raise ValueError('Eigenpair dimensions do not match the selected band window')
    reference_spectrum = np.loadtxt(reference / 'spectrum_singlet_tda.dat')
    candidate_spectrum = np.loadtxt(output / 'spectrum_singlet_tda.dat')
    if reference_spectrum.shape != candidate_spectrum.shape or not np.isfinite(candidate_spectrum).all():
        raise ValueError('Invalid spectrum dimensions or entries')
    if not np.allclose(reference_spectrum[:, 0], candidate_spectrum[:, 0], rtol=0, atol=1e-12):
        raise ValueError('Spectrum energy grid differs')
    spectrum_error = float(np.linalg.norm(candidate_spectrum[:, 1:] - reference_spectrum[:, 1:]) /
                           max(np.linalg.norm(reference_spectrum[:, 1:]), 1e-30))
    metrics['spectrum_relative_error'] = spectrum_error
    metrics['pass'] &= spectrum_error <= 2e-5
    if requires_screening:
        replay = screening_to_cache(output, work / 'replay_W', float(options['bse_plasma_energy_ev']))
        replay_options = dict(options)
        replay_options.update({'screened_format': replay['format'], 'screened_dir': str(work / 'replay_W'),
                               'bse_plasma_energy_ev': '0', 'out_screening_matrices': 'false'})
        run_bse(args.executable.resolve(), work / 'replay', replay_options, definition['ranks'])
        replay_energies, replay_amplitudes = read_eigenpairs(work / 'replay/libbse.d')
        metrics['W_replay'] = compare_eigenpairs(energies, amplitudes, replay_energies, replay_amplitudes)
        metrics['pass'] &= metrics['W_replay']['pass']
        print_eigenstate_matches('W replay vs chi0/dynamic path', metrics['W_replay'])
    (work / 'comparison.txt').write_text(pprint.pformat(metrics) + '\n')
    summary_metrics = dict(metrics)
    for key in ('eigenpairs', 'W_replay'):
        if key in summary_metrics:
            summary_metrics[key] = {name: value for name, value in metrics[key].items() if name != 'states'}
    print(pprint.pformat({'case': args.case, 'work': str(work), **summary_metrics}))
    return 0 if metrics['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
