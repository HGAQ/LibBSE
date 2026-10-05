"""Import LibBSE TDA rank-local amplitudes without changing their band gauge.

Usage: python libbse_to_exciview.py RUN_DIR --v-start 11 --c-start 15 -o excitons.npz
State indices in ExciView are zero based; absolute band IDs are one based.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def read_parameters(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        words = line.split('#', 1)[0].replace('=', ' ').split()
        if len(words) >= 2:
            result[words[0]] = words[1:]
    return result


def read_amplitudes(files, dimension, nstates):
    """LibBSE contiguous pair partitions, including unequal rank lengths."""
    chunks = []
    for rank, path in enumerate(files):
        count = dimension // len(files) + (rank < dimension % len(files))
        rows = []
        with Path(path).open() as stream:
            for state in range(nstates):
                line = stream.readline()
                if not line:
                    raise ValueError(f'{path}: missing state {state}')
                tokens = re.findall(r'\(([^(),]+),([^(),]+)\)', line)
                if len(tokens) != count or re.sub(r'\([^()]+\)', '', line).strip():
                    raise ValueError(f'{path}: state {state}: expected {count} complex amplitudes')
                rows.append([complex(float(r), float(i)) for r, i in tokens])
            if stream.read().strip():
                raise ValueError(f'{path}: unexpected extra states')
        chunks.append(np.asarray(rows, dtype=complex).reshape(nstates, count))
    amplitudes = np.concatenate(chunks, axis=1)
    if not np.isfinite(amplitudes).all():
        raise ValueError('Non-finite amplitudes')
    if not np.allclose(np.sum(abs(amplitudes)**2, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError('TDA state norm differs from one; check MPI files and output precision')
    return amplitudes


def find_mulliken_dir(run, data_dir, requested=None):
    """Prefer explicit/local outputs, then the established sibling run names."""
    if requested is not None:
        directory = Path(requested).resolve()
        if not list(directory.glob('bandmlk*.out')):
            raise ValueError(f'No bandmlk*.out files in {directory}')
        return directory
    for directory in (run, data_dir, run.parent/'DFT_muliken',
                      run.parent/'DFT_mulliken', run.parent/'aims_mulliken_333'):
        if list(directory.glob('bandmlk*.out')):
            return directory.resolve()
    return None


def read_mulliken(directory, kpoints, band_ids):
    """Read every k block, matching periodic coordinates rather than file order.

    Repeated endpoints are accepted only when the requested populations agree.
    The result contains total/s/p/... populations, with absent angular channels
    padded by zero. All requested bands and all BSE k points must be present.
    """
    files = sorted(Path(directory).glob('bandmlk*.out'))
    band_ids = list(band_ids)
    kpoints = np.asarray(kpoints, dtype=float)
    records = {}
    for path in files:
        blocks = re.split(r'(?=^\s*k point number:)', path.read_text(), flags=re.M)
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().splitlines()
            header = re.fullmatch(r'k point number:\s*\d+:\s*\(([^)]+)\)\s*', lines[0])
            if header is None:
                raise ValueError(f'Invalid Mulliken k-point header in {path}')
            coord = np.array([float(x.replace('D', 'E').replace('d', 'e'))
                              for x in header[1].split()])
            if coord.shape != (3,) or not np.isfinite(coord).all():
                raise ValueError(f'Invalid Mulliken coordinates in {path}')
            delta = kpoints - coord
            delta -= np.rint(delta)
            hits = np.flatnonzero(np.max(abs(delta), axis=1) < 1e-6)
            if len(hits) != 1:
                raise ValueError(f'Mulliken k point {coord} does not uniquely match BSE grid: {path}')
            k = int(hits[0])
            populations = {}
            for line in lines[1:]:
                words = line.split()
                if not words or not words[0].isdigit():
                    continue
                band = int(words[0])
                if band not in band_ids:
                    continue
                if len(words) < 6:
                    raise ValueError(f'Incomplete Mulliken row in {path}: {line}')
                atom = int(words[3])
                values = np.array([float(x.replace('D', 'E').replace('d', 'e'))
                                   for x in words[4:]])
                if atom < 1 or not np.isfinite(values).all() or (band, atom) in populations:
                    raise ValueError(f'Invalid/duplicate Mulliken row in {path}: {line}')
                populations[band, atom] = values
            if {band for band, atom in populations} != set(band_ids):
                raise ValueError(f'Missing requested Mulliken bands at k index {k}: {path}')
            if k in records:
                previous = records[k]
                if previous.keys() != populations.keys() or any(
                    previous[key].shape != values.shape or not np.allclose(
                        previous[key], values, atol=1e-7, rtol=0)
                    for key, values in populations.items()
                ):
                    raise ValueError(f'Conflicting duplicate Mulliken k point {k}: {path}')
            else:
                records[k] = populations
    if set(records) != set(range(len(kpoints))):
        raise ValueError(f'Incomplete Mulliken k grid: got {len(records)}, expected {len(kpoints)}')
    atoms = sorted({atom for population in records.values() for band, atom in population})
    expected = {(band, atom) for band in band_ids for atom in atoms}
    if any(set(population) != expected for population in records.values()):
        raise ValueError('Inconsistent Mulliken atom coverage between bands/k points')
    ncolumns = max(len(values) for population in records.values() for values in population.values())
    data = np.zeros((len(kpoints), len(band_ids), len(atoms), ncolumns))
    for k, population in records.items():
        for ib, band in enumerate(band_ids):
            for ia, atom in enumerate(atoms):
                values = population[band, atom]
                data[k, ib, ia, :len(values)] = values
    return {'mulliken_populations': data, 'mulliken_band_ids': np.array(band_ids),
            'mulliken_atom_ids': np.array(atoms)}, files


def convert(run_dir, output, v_start, c_start, spin='singlet', mulliken_dir=None):
    run = Path(run_dir).resolve()
    params = read_parameters(run / 'libbse.in')
    nv, nc = int(params['nocc'][0]), int(params['nvirt'][0])
    if v_start < 1 or c_start != v_start + nv:
        raise ValueError('Expected contiguous one-based occupied and virtual band windows')
    if params.get('qp_format', ['energy_qp'])[0] == 'energy_qp' and 'qp_data' in params:
        blocks = (run / params['qp_data'][0]).read_text().split('K_point')[1:]
        for block in blocks:
            rows = [line.split() for line in block.splitlines()[1:]
                    if len(line.split()) == 4 and line.split()[0].isdigit()]
            occupied = [int(row[0]) for row in rows if float(row[1]) > 1e-6]
            if not occupied or max(occupied) + 1 != c_start:
                raise ValueError('Band window does not match occupied bands in energy_qp')
    if params.get('bse_use_fine_kgrid', ['0'])[0] != '0':
        raise ValueError('This importer currently requires the regular SCF grid (bse_use_fine_kgrid 0)')
    data_dir = run / params['input_dir'][0]
    bz = data_dir / 'bz_sampling_out'
    lines = bz.read_text().splitlines()
    mesh = np.array([int(x) for x in lines[0].split()])
    nk = int(lines[1].split()[0])
    records = np.array([[float(x) for x in line.split()] for line in lines[2:2+nk]])
    if len(records) != nk or not np.array_equal(records[:, 0], np.arange(1, nk+1)):
        raise ValueError('Invalid k-point coverage/order')
    if not np.isfinite(records).all() or not np.allclose(records[:,1], 1/nk, atol=1e-10, rtol=0):
        raise ValueError('Expected finite k-point data and uniform weights')
    kpoints = records[:, 2:5]
    if len(np.unique(np.round(kpoints % 1, 9), axis=0)) != nk or np.prod(mesh) != nk:
        raise ValueError('Expected a complete regular k grid')
    outdir = run / params.get('output_dir', ['libbse.d'])[0]
    energy_file = outdir / f'Excitation_Energy_{spin}.dat'
    energies = np.loadtxt(energy_file).reshape(-1)
    if not np.isfinite(energies).all():
        raise ValueError('Non-finite excitation energies')
    # LibBSE uses ABACUS Rydberg units internally (including this text file),
    # whereas energy_qp input uses Hartree. Preserve both explicitly.
    optical_file = outdir / f'oscillator_strength_{spin}_tda.dat'
    if optical_file.exists():
        optical = np.atleast_2d(np.loadtxt(optical_file))
        if optical.shape[0] != len(energies) or not np.allclose(
                optical[:,1], energies * (27.211386245988 / 2), atol=1e-5, rtol=0):
            raise ValueError('Excitation energy units/order disagree with the eV optical table')
    files = sorted(outdir.glob(f'Excitation_Amplitude_{spin}_*.dat'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    if not files or [int(p.stem.rsplit('_', 1)[1]) for p in files] != list(range(len(files))):
        raise ValueError('Missing/non-contiguous MPI rank files')
    amplitudes = read_amplitudes(files, nk*nv*nc, len(energies))
    sources = [run/'libbse.in', bz, energy_file, *files]
    mulliken = {}
    directory = find_mulliken_dir(run, data_dir, mulliken_dir)
    if directory is not None:
        mulliken, mulliken_files = read_mulliken(
            directory, kpoints, range(v_start, c_start+nc))
        sources.extend(mulliken_files)
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    gauge = params.get('wavefunction_gauge', ['auto'])[0]
    if gauge == 'auto' and params.get('input_format', ['auto'])[0] == 'fhi_aims':
        gauge = 'native'
    np.savez(output, coefficients=amplitudes.reshape(-1,nk,nv,nc),
             energies_ha=energies/2, energies_ry=energies, kpoints=kpoints, kmesh=mesh,
             v_start=v_start, c_start=c_start,
             wavefunction_gauge=gauge,
             source='LibBSE TDA', provenance=json.dumps(provenance), **mulliken)
    return {'states': len(energies), 'dimension': nk*nv*nc, 'mpi_ranks': len(files),
            'mulliken_dir': str(directory) if directory is not None else None,
            'mulliken_kpoints': nk if mulliken else 0,
            'max_norm_error': float(np.max(abs(np.sum(abs(amplitudes)**2,axis=1)-1)))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--v-start', type=int, required=True)
    parser.add_argument('--c-start', type=int, required=True)
    parser.add_argument('--spin', choices=['singlet','triplet','rpa','ipa'], default='singlet')
    parser.add_argument('-o', '--output', default='excitons.npz')
    parser.add_argument('--mulliken-dir', help='Directory containing original bandmlk*.out files; '
                        'otherwise detect local outputs or sibling DFT_muliken/DFT_mulliken/aims_mulliken_333')
    args = parser.parse_args()
    print(json.dumps(convert(args.run_dir, args.output, args.v_start, args.c_start,
                             args.spin, args.mulliken_dir), indent=2))
