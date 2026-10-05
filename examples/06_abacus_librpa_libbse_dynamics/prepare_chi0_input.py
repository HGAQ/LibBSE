#!/usr/bin/env python3
"""Select ABACUS's consistent small auxiliary-basis export for chi0 screening.

With shrink_abfs_pca_thr enabled, ABACUS exports Cs_shrinked_data, full V and
coulomb_cut in its small basis; Cs_data and coulomb_unshrinked_cut instead use
the large basis. Bare chi0 must use one basis throughout. This view only selects
and aliases existing producer files, without transforming any matrix values.
LibRPA then runs with use_shrink_abfs=f in this selected producer basis.
"""
import argparse
import json
import os
from pathlib import Path


def prepare(source, destination):
    source = source.resolve()
    mapping = {}
    for name in ('band_out', 'stru_out', 'velocity_matrix'):
        p = source/name
        if not p.is_file() or not p.stat().st_size:
            raise ValueError(f'Missing producer input: {p}')
        mapping[name] = p
    for pattern in ('KS_eigenvector_*.dat', 'Cs_shrinked_data_*.txt',
                    'coulomb_mat_*.txt', 'coulomb_cut_*.txt'):
        files = sorted(source.glob(pattern))
        if not files:
            raise ValueError(f'Missing producer files: {pattern}')
        for p in files:
            name = p.name.replace('Cs_shrinked_data_', 'Cs_data_')
            mapping[name] = p
            if name.startswith('coulomb_cut_'):
                mapping[name.replace('coulomb_cut_', 'coulomb_unshrinked_cut_')] = p
    for name in ('bz_sampling_out',):
        if (source/name).is_file():
            mapping[name] = source/name
    # Empty MPI rank fragments contain only nq. Nonempty blocks must agree on
    # the total small-basis dimension for both full and cut Coulomb matrices.
    dimensions = set()
    for pattern in ('coulomb_mat_*.txt', 'coulomb_cut_*.txt'):
        found = False
        for p in source.glob(pattern):
            with p.open() as handle:
                next(handle)
                header = next((line.split() for line in handle if line.strip()), [])
                if header:
                    if len(header) != 5:
                        raise ValueError(f'Unexpected legacy Coulomb header: {p}')
                    dimensions.add(int(header[0]))
                    found = True
        if not found:
            raise ValueError(f'No nonempty Coulomb block: {pattern}')
    if len(dimensions) != 1:
        raise ValueError(f'Full/cut auxiliary dimensions differ: {dimensions}')
    destination.mkdir(exist_ok=False)
    for name, p in mapping.items():
        (destination/name).symlink_to(os.path.relpath(p, destination.resolve()))
    manifest = {'auxiliary_basis': 'ABACUS producer small basis',
                'naux': dimensions.pop(), 'matrix_values_changed': False,
                'files': {name: os.path.relpath(p, destination.resolve()) for name, p in mapping.items()}}
    (destination/'input_view.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Prepared matching RI/full-V/cut-V input view with {manifest["naux"]} auxiliary functions: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
