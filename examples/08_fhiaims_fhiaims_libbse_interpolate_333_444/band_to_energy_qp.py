#!/usr/bin/env python3
"""Convert complete 4x4x4 band QP data to energy_qp (Hartree).

FHI-aims GW_band energies use a common chemical-potential reference; the
reference cancels from BSE transition energies. For aims, e_gs is a placeholder
copy of e_qp, as in the existing benchmark; LibBSE reads KS data separately.
LibRPA provides KS_band_spin_1.dat, so its actual KS column is retained.
"""
import argparse
from pathlib import Path
import numpy as np


def convert(producer, source, output):
    if producer == 'aims':
        qp = np.concatenate([np.loadtxt(source / f'GW_band1{i:03d}.out', ndmin=2)
                             for i in range(1, 17)])
        ks = qp.copy()
        first_state, ha_to_ev = 11, 27.2113845
    else:
        qp = np.loadtxt(source / 'GW_band_spin_1.dat', ndmin=2)
        ks = np.loadtxt(source / 'KS_band_spin_1.dat', ndmin=2)
        first_state, ha_to_ev = 1, 27.211386245988
    if qp.shape != (64, 20) or ks.shape != qp.shape:
        raise ValueError(f'Expected 64 k points and 8 occupation/energy pairs: {qp.shape}, {ks.shape}')
    if not np.isfinite(qp).all() or not np.isfinite(ks).all():
        raise ValueError('Non-finite KS/QP data')
    if not np.allclose(qp[:, 1:4], ks[:, 1:4], atol=1e-7):
        raise ValueError('KS/QP k-point ordering differs')
    frac = np.mod(qp[:, 1:4], 1.0)
    indices = np.rint(frac * 4).astype(int) % 4
    if not np.allclose(frac * 4, np.rint(frac * 4), atol=1e-6):
        raise ValueError('QP coordinates do not lie on the unshifted 4x4x4 grid')
    keys = indices[:, 0]*16 + indices[:, 1]*4 + indices[:, 2]
    if len(set(keys)) != 64:
        raise ValueError('Incomplete or duplicate fine-grid k points')
    occ = qp[:, 4::2].copy()
    if producer == 'librpa' and np.isclose(occ.max(), 2/64):
        occ *= 64  # Compatibility with the old k-weighted band writer.
    expected = np.tile([2.0]*4 + [0.0]*4, (64, 1))
    if not np.allclose(occ, expected, atol=1e-3):
        raise ValueError('Expected four occupied and four empty bands at every k point')
    sep = '-'*100 + '\n'
    with output.open('w') as handle:
        handle.write(' state occ_num e_gs(Ha) e_qp(Ha)\n' + sep)
        for key, row in enumerate(np.argsort(keys), 1):
            xyz = indices[row]/4
            handle.write(f' K_point {key:4d} : {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}\n'+sep)
            for band in range(8):
                handle.write(f' {first_state+band:6d} {occ[row,band]:8.4f} '
                             f'{ks[row,5+2*band]/ha_to_ev:20.12E} '
                             f'{qp[row,5+2*band]/ha_to_ev:20.12E}\n')
            handle.write(sep+'\n')
    print(f'Validated and converted 64 k points, bands {first_state}--{first_state+7}: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('producer', choices=['aims', 'librpa'])
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    convert(args.producer, args.source, args.output)
