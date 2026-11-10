"""Explicitly package a completed aims Si 222 all-symmetry export; never run in CI."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace an existing fixture archive')
    source_root = args.source.resolve()
    if 'Have a nice day' not in (source_root / 'aims.out').read_text():
        raise ValueError('Producer did not finish successfully')
    source_manifest = source_root / 'aims_symmetry_all.dat'
    symmetry_lines = source_manifest.read_text().splitlines()
    if symmetry_lines[1].split() != ['218', '8']:
        raise ValueError('Expected Si 222 with 218 auxiliary functions')
    representative_ids = sorted({int(line.split()[1]) for line in symmetry_lines[3:] if line.strip()})
    source_paths = [source_manifest, source_root / 'control.in', source_root / 'geometry.in',
                    source_root / 'aims.out', source_root / 'producer.sha256']
    with tempfile.TemporaryDirectory(prefix='si222-symmetry-pack-') as temporary_directory:
        archive_root = Path(temporary_directory)
        screening_directory = archive_root / 'screening_all'
        screening_directory.mkdir()
        (screening_directory / source_manifest.name).write_bytes(source_manifest.read_bytes())
        for representative_id in representative_ids:
            representative_parts = []
            representative_header = None
            for source_path in sorted(source_root.glob(f'periodic_gw_w_q_{representative_id}_rank_*.dat')):
                source_paths.append(source_path)
                current_header = [line for line in source_path.read_text().splitlines() if line.startswith('#')]
                if representative_header is not None and current_header != representative_header:
                    raise ValueError('Producer rank headers differ')
                representative_header = current_header
                representative_parts.append(np.loadtxt(source_path, ndmin=2))
            representative_entries = np.concatenate(representative_parts)
            if representative_entries.shape != (218 * 218, 6):
                raise ValueError('Incomplete or non-single-frequency representative W')
            np.savez_compressed(screening_directory / f'periodic_gw_w_q_{representative_id}_rank_0.npz',
                                entries=representative_entries, header=np.asarray(representative_header))
        for symmetry_line in symmetry_lines[3:]:
            if not symmetry_line.strip():
                continue
            delta_filename = symmetry_line.split()[-1]
            source_path = source_root / delta_filename
            source_paths.append(source_path)
            delta_header = source_path.read_text().splitlines()[:2]
            delta_entries = np.loadtxt(source_path, skiprows=2)
            if delta_entries.shape != (218 * 218, 4):
                raise ValueError('Incomplete Delta')
            delta_indices = delta_entries[:, :2].astype(int) - 1
            if (np.any(delta_indices < 0) or np.any(delta_indices >= 218)
                    or len(np.unique(delta_indices[:, 0] * 218 + delta_indices[:, 1])) != 218 * 218):
                raise ValueError('Invalid Delta indices')
            symmetry_delta = np.empty((218, 218), dtype=complex)
            symmetry_delta[delta_indices[:, 0], delta_indices[:, 1]] = delta_entries[:, 2] + 1j * delta_entries[:, 3]
            np.savez_compressed(screening_directory / Path(delta_filename).with_suffix('.npz'),
                                symmetry_delta=symmetry_delta, header=np.asarray(delta_header))
        provenance_directory = archive_root / 'provenance/gw_all'
        provenance_directory.mkdir(parents=True)
        for filename in ('control.in', 'geometry.in', 'aims.out', 'producer.sha256'):
            (provenance_directory / filename).write_bytes((source_root / filename).read_bytes())
        source_checksums = []
        for source_path in source_paths:
            with source_path.open('rb') as source_handle:
                digest = hashlib.file_digest(source_handle, 'sha256').hexdigest()
            source_checksums.append(f'{digest}  {source_path.name}\n')
        (provenance_directory / 'source_files.sha256').write_text(''.join(source_checksums))
        with tarfile.open(args.output, 'w:xz') as archive:
            for archive_path in sorted(archive_root.rglob('*')):
                if archive_path.is_file():
                    archive.add(archive_path, arcname=str(archive_path.relative_to(archive_root)))


if __name__ == '__main__':
    main()
