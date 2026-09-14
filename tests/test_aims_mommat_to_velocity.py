#!/usr/bin/env python3
"""Regression test for the FHI-aims Fortran HDF5 grid ordering."""

from pathlib import Path
import importlib.util
import struct
import tempfile

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "aims_mommat_to_velocity.py"
SPEC = importlib.util.spec_from_file_location("aims_mommat_to_velocity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def read_block(path: Path, ik: int, nk: int, nbands: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = struct.unpack("<7i", stream.read(28))
        assert header == (-12345680, 29, nk, 1, nbands, nbands, 3)
        table = [struct.unpack("<iq", stream.read(12)) for _ in range(nk)]
        assert [item[0] for item in table] == list(range(1, nk + 1))
        stream.seek(table[ik][1])
        return np.fromfile(stream, dtype="<c16", count=3 * nbands * nbands).reshape(
            1, 3, nbands, nbands
        ) / CONVERTER.ATOMIC_TO_EV_ANGSTROM


def main() -> None:
    # h5py sees the Fortran-written grid as (kz, ky, kx) = (2, 1, 3).
    h5_shape = (2, 1, 3)
    canonical_shape = h5_shape[::-1]
    nk = int(np.prod(h5_shape))
    nbands = 3
    npairs = nbands * (nbands + 1) // 2
    with tempfile.TemporaryDirectory(prefix="libbse_mommat_test_") as temporary:
        directory = Path(temporary)
        (directory / "band_out").write_text(f"{nk}\n1\n{nbands}\n{nbands}\n")
        k_points = np.zeros((*h5_shape, 4))
        momentum = np.zeros((*h5_shape, npairs, 6))
        for kx, ky, kz in np.ndindex(canonical_shape):
            ik = (kx * canonical_shape[1] + ky) * canonical_shape[2] + kz
            source_cell = (kz, ky, kx)
            k_points[source_cell] = (ik, kx, ky, kz)
            for pair in range(npairs):
                for alpha in range(3):
                    base = 1000 * ik + 100 * alpha + pair + 1
                    momentum[source_cell + (pair, 2 * alpha)] = base
                    momentum[source_cell + (pair, 2 * alpha + 1)] = -base / 10
        with h5py.File(directory / "mommat.h5", "w") as handle:
            handle["Energy_window"] = np.array([[1.0], [3.0]])
            handle["k_points"] = k_points
            handle["Momentummatrix"] = momentum

        output = directory / "velocity_matrix"
        CONVERTER.convert(directory, output)

        # ik=4 lives at source HDF5 cell (kz,ky,kx)=(0,0,2). Pair 4 is
        # (state 2,state 3) in the row-major packed upper triangle.
        block = read_block(output, ik=4, nk=nk, nbands=nbands)
        gradient = complex(4000 + 4 + 1, -(4000 + 4 + 1) / 10)
        expected = -1j * gradient
        assert np.allclose(block[0, 0, 2, 1], expected)
        assert np.allclose(block[0, 0, 1, 2], np.conjugate(expected))


if __name__ == "__main__":
    main()
