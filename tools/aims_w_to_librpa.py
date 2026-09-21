#!/usr/bin/env python3
"""Export FHI-aims full W as legacy LibRPA Wc(R) files (NumPy only).

Usage: python3 aims_w_to_librpa.py libbse.in output_directory

Wc(R) = FT[W_aims(q, i*omega_min)] - Re FT[v_cut(q)]. Legacy consumers
add the same bare v_cut(R) back. This converter never recomputes screening.
The native LibBSE W reader still calls LibRPA's FT_Wc_q2R directly; this
standalone Python utility implements its linear Fourier convention in NumPy.
"""
from pathlib import Path
import argparse
import itertools
import re
import struct
import numpy as np


def read_options(path):
    values = {}
    for line in path.read_text().splitlines():
        words = line.split('#', 1)[0].replace('=', ' ').split()
        if len(words) >= 2:
            values[words[0]] = words[1]
    if values.get('screened_format') != 'fhi_aims_w':
        raise ValueError('converter requires screened_format fhi_aims_w')
    return {k: (path.parent / values[k]).resolve() for k in ('input_dir', 'screened_dir')}


def read_layout(directory):
    words = (directory / 'basis_out').read_text().split()
    types = {int(words[4+3*i]): int(words[6+3*i]) for i in range(int(words[0]))}
    atoms = (directory / 'stru_out').read_text().split()
    sizes = [types[int(atoms[22+4*i])] for i in range(int(atoms[18]))]
    lines = (directory / 'bz_sampling_out').read_text().splitlines()
    mesh = tuple(map(int, lines[0].split()[:3]))
    nq = int(lines[1].split()[0])
    if np.prod(mesh) != nq or min(mesh) <= 0:
        raise ValueError('requires a complete uniform coarse q mesh')
    qs = {int(row.split()[0]): np.array(list(map(float, row.split()[2:5])))
          for row in lines[2:2+nq]}
    return sizes, mesh, qs


def grid_index(q, mesh):
    scaled = np.asarray(q)*mesh
    if not np.allclose(scaled, np.rint(scaled), atol=1e-7, rtol=0):
        raise ValueError(f'q point {q} is off the coarse mesh {mesh}')
    return tuple((np.rint(scaled).astype(int) % mesh).tolist())


def read_header(path, quantity="w"):
    header = {}
    with path.open() as stream:
        for line in stream:
            if not line.startswith('#'):
                break
            if ':' in line:
                k, v = line[1:].split(':', 1)
                header[k.strip()] = v.strip()
            if 'Original RI auxiliary basis; atomic units;' in line:
                header['units'] = True
    if header.get('quantity') != quantity or not header.get('units'):
        raise ValueError(f'expected full W in the original RI basis: {path}')
    return header


def read_w(directory, mesh, naux, quantity="w"):
    groups = {}
    for path in sorted(directory.glob(f'periodic_gw_{quantity}_q_*_rank_*.dat')):
        h = read_header(path, quantity)
        n, nf, iq = map(int, h['n_basbas n_freq q_index'].split())
        match = re.fullmatch(rf'periodic_gw_{quantity}_q_(\d+)_rank_\d+\.dat', path.name)
        if n != naux or nf < 1 or not match or iq != int(match[1]):
            raise ValueError(f'inconsistent W header: {path}')
        index = grid_index(list(map(float, h['q_fractional'].split())), mesh)
        groups.setdefault(index, []).append((path, nf))
    if not groups:
        raise ValueError('no aims W rank files found')
    matrices = {}; selected = None
    for index, paths in sorted(groups.items()):
        frequencies = {}
        for path, nf in paths:
            with path.open() as stream:
                for line in stream:
                    if not line.strip() or line.startswith('#'):
                        continue
                    fields = line.split(None, 2)
                    iw, omega = int(fields[0]), float(fields[1])
                    if not 1 <= iw <= nf or not np.isfinite(omega):
                        raise ValueError(f'invalid frequency in {path}')
                    if iw in frequencies and frequencies[iw] != omega:
                        raise ValueError(f'inconsistent frequency in {path}')
                    frequencies[iw] = omega
            if frequencies:
                if len(frequencies) != nf:
                    raise ValueError(f'missing frequency nodes: {path}')
                break  # Empty BLACS rank files are allowed.
        node = min(frequencies, key=lambda k: abs(frequencies[k]))
        omega = frequencies[node]
        if selected is not None and abs(selected-omega) > 1e-12:
            raise ValueError('different lowest frequencies across q points')
        selected = omega
        matrix = np.zeros((naux, naux), dtype=complex)
        seen = np.zeros((naux, naux), dtype=bool)
        for path, _ in paths:
            with path.open() as stream:
                def rows():
                    for line in stream:
                        if line.strip() and not line.startswith('#') and int(line.split(None, 1)[0]) == node:
                            yield line
                data = np.loadtxt(rows(), ndmin=2)
            if data.size == 0:
                continue
            if data.shape[1] != 6 or not np.all(np.isfinite(data)):
                raise ValueError(f'invalid W entries: {path}')
            indices = data[:, 2:4].astype(int)
            if np.any(indices != data[:, 2:4]) or np.min(indices) < 1 or np.max(indices) > naux:
                raise ValueError(f'invalid auxiliary indices: {path}')
            mu, nu = (indices-1).T
            if np.any(np.abs(data[:, 1]-omega) > 1e-12) or np.any(seen[mu, nu]) or len(np.unique(mu*naux+nu)) != len(mu):
                raise ValueError(f'duplicate entries or inconsistent frequency: {path}')
            matrix[mu, nu] = data[:, 4]+1j*data[:, 5]; seen[mu, nu] = True
        if not seen.all():
            raise ValueError(f'missing W rank block at q={index}')
        matrices[index] = matrix
        print(f'Read {quantity} at q={index}, omega={omega:.17e} Ha', flush=True)
    return complete_inverse(matrices, mesh), selected


def complete_inverse(matrices, mesh):
    """Only time reversal can be restored without auxiliary rotation metadata."""
    for index in itertools.product(*(range(m) for m in mesh)):
        if index not in matrices:
            partner = tuple((-i) % m for i, m in zip(index, mesh))
            if partner not in matrices:
                raise ValueError('missing general q star: export aims with symmetry none or inverse')
            matrices[index] = matrices[partner].conj()
    return np.array([matrices[index] for index in itertools.product(*(range(m) for m in mesh))]).reshape((*mesh, *next(iter(matrices.values())).shape))


def read_coulomb(directory, mesh, qs, sizes, prefix="coulomb_cut"):
    """Read FHI-aims binary-v1 upper atom-pair blocks, completing Hermitian V."""
    naux = sum(sizes); offsets = np.cumsum([0]+sizes)
    pairs = list(itertools.combinations_with_replacement(range(len(sizes)), 2))
    matrices = {}
    for path in sorted(directory.glob(f'{prefix}_*.dat')):
        with path.open('rb') as stream:
            marker, iq, n, flag, na, nb = struct.unpack('=6i', stream.read(24))
            if marker != -20129433 or n != naux or na != len(sizes) or flag not in (0, 1):
                raise ValueError(f'expected aims binary-v1 cut Coulomb: {path}')
            if tuple(sizes) != struct.unpack(f'={na}i', stream.read(4*na)):
                raise ValueError(f'auxiliary atom layout mismatch: {path}')
            records = [struct.unpack('=iq', stream.read(12)) for _ in range(nb)]
            matrix = np.zeros((naux, naux), dtype=complex); seen = set()
            for pair, offset in records:
                if pair not in range(len(pairs)) or pair in seen:
                    raise ValueError(f'invalid/duplicate atom-pair index: {path}')
                seen.add(pair); i, j = pairs[pair]; stream.seek(offset)
                block = np.fromfile(stream, dtype='=c16' if flag else '=f8', count=sizes[i]*sizes[j]).reshape(sizes[i], sizes[j])
                si, sj = slice(offsets[i], offsets[i+1]), slice(offsets[j], offsets[j+1])
                matrix[si, sj] = block
                if i != j:
                    matrix[sj, si] = block.conj().T
            if len(seen) != len(pairs):
                raise ValueError(f'missing Coulomb atom-pair blocks: {path}')
            index = grid_index(qs[iq], mesh)
            if index in matrices:
                raise ValueError(f'duplicate Coulomb q point: {path}')
            matrices[index] = matrix
    return complete_inverse(matrices, mesh)


def to_real_space(matrices):
    # NumPy fftn uses the negative phase. The explicit 1/Nq matches LibRPA;
    # fftshift gives R=-N//2,...,(N-1)//2 with z varying fastest.
    return np.fft.fftshift(np.fft.fftn(matrices, axes=(0, 1, 2)), axes=(0, 1, 2))/np.prod(matrices.shape[:3])


def write_blocks(directory, matrices, sizes, omega, description="W_aims - v_cut"):
    directory.mkdir(parents=True, exist_ok=True)
    mesh = matrices.shape[:3]; offsets = np.cumsum([0]+sizes)
    cells = itertools.product(*(range(-(n//2), (n-1)//2+1) for n in mesh))
    for ir, (cell, matrix) in enumerate(zip(cells, matrices.reshape((-1, sum(sizes), sum(sizes))))):
        for i, ni in enumerate(sizes):
            for j, nj in enumerate(sizes):
                block = matrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
                mu, nu = np.indices((ni, nj))
                data = np.column_stack((mu.ravel()+1, nu.ravel()+1, block.real.ravel(), block.imag.ravel()))
                path = directory/f'Wc_Mu_{i}_Nu_{j}_iR_{ir}_ifreq_0.mtx'
                with path.open('w') as stream:
                    # ABACUS's legacy reader expects the R vector on line 3.
                    stream.write('%%MatrixMarket matrix coordinate complex general\n%\n')
                    stream.write(f'% Wc = {description}, Hartree; R = ({cell[0]} {cell[1]} {cell[2]}); omega_Ha = {omega:.17e}\n')
                    stream.write(f'{ni} {nj} {ni*nj}\n')
                    np.savetxt(stream, data, fmt=['%d', '%d', '%.17e', '%.17e'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); options = read_options(args.input.resolve())
    sizes, mesh, qs = read_layout(options['input_dir'])
    w, omega = read_w(options['screened_dir'], mesh, sum(sizes))
    v = read_coulomb(options['input_dir'], mesh, qs, sizes)
    # LibRPA FT_Vq returns the real part, whereas FT_Wc_q2R retains complex W.
    wc = to_real_space(w)-to_real_space(v).real
    write_blocks(args.output, wc, sizes, omega)
    print(f'Wc(R) written: coarse mesh={mesh}, omega={omega:.17e} Ha')


if __name__ == '__main__':
    main()
