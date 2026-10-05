#!/usr/bin/env python3
"""Convert FHI-aims mommat.h5 to LibRPA velocity_matrix binary v1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import sys
import tempfile

try:
    import h5py
    import numpy as np
except ImportError as error:  # pragma: no cover - depends on the installation
    raise SystemExit(
        "aims_mommat_to_velocity.py requires the h5py and numpy packages"
    ) from error


VELOCITY_V1_MARKER = -12345680
VELOCITY_V1_COMPLEX_DOUBLE = 29
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210544
ATOMIC_TO_EV_ANGSTROM = HARTREE_TO_EV * BOHR_TO_ANGSTROM


def _exact_integer(value: float, description: str) -> int:
    nearest = round(float(value))
    if not np.isfinite(value) or abs(float(value) - nearest) > 1.0e-8:
        raise ValueError(f"invalid integer in {description}: {value}")
    return nearest


def _read_band_dimensions(input_dir: Path) -> tuple[int, int, int, int]:
    try:
        words = (input_dir / "band_out").read_text(encoding="utf-8").split()
        dimensions = tuple(int(word) for word in words[:4])
    except (OSError, ValueError) as error:
        raise ValueError("cannot read dimensions from FHI-aims band_out") from error
    if len(dimensions) != 4 or any(value <= 0 for value in dimensions):
        raise ValueError("invalid dimensions in FHI-aims band_out")
    if dimensions[1] != 1:
        raise ValueError("only spin-unpolarized FHI-aims momentum data are supported")
    return dimensions


def _read_momentum(
    input_dir: Path, dimensions: tuple[int, int, int, int]
) -> tuple[np.ndarray, int, int]:
    nk, _nspin, nbands, _nao = dimensions
    hdf5_file = input_dir / "mommat.h5"
    with h5py.File(hdf5_file, "r") as handle:
        window = np.asarray(handle["Energy_window"]).reshape(-1)
        if window.size != 2:
            raise ValueError("unsupported FHI-aims Energy_window layout")
        state_min = _exact_integer(window[0], "Energy_window")
        state_max = _exact_integer(window[1], "Energy_window")
        if state_min < 1 or state_max < state_min or state_max > nbands:
            raise ValueError("mommat.h5 band window is outside band_out")

        k_points = np.asarray(handle["k_points"], dtype=np.float64)
        momentum = np.asarray(handle["Momentummatrix"], dtype=np.float64)

    nwindow = state_max - state_min + 1
    npairs = nwindow * (nwindow + 1) // 2
    if k_points.ndim != 4 or k_points.shape[-1] != 4:
        raise ValueError("unsupported FHI-aims k_points layout")
    if momentum.shape != (*k_points.shape[:3], npairs, 6):
        raise ValueError("unsupported FHI-aims Momentummatrix layout")
    if int(np.prod(k_points.shape[:3])) != nk:
        raise ValueError("mommat.h5 k grid is inconsistent with band_out")

    # FHI-aims writes the grid with the Fortran HDF5 interface.  h5py exposes
    # those axes as (kz, ky, kx); reverse them before C-order flattening so the
    # resulting order is (kx, ky, kz), with kz varying fastest.  This is the
    # same reordering used by the validated h5_ref_.py analysis script.
    k_points = k_points.transpose(2, 1, 0, 3)
    momentum = momentum.transpose(2, 1, 0, 3, 4)

    raw_indices = np.rint(k_points[..., 0]).astype(np.int64).reshape(-1)
    if not np.allclose(k_points[..., 0].reshape(-1), raw_indices, atol=1.0e-8):
        raise ValueError("mommat.h5 contains a non-integral k-point index")
    if np.array_equal(np.sort(raw_indices), np.arange(nk)):
        indices = raw_indices
    elif np.array_equal(np.sort(raw_indices), np.arange(1, nk + 1)):
        indices = raw_indices - 1
    else:
        raise ValueError("mommat.h5 contains an invalid k-point index range")

    ordered = np.empty((nk, npairs, 6), dtype=np.float64)
    for cell, ik in enumerate(indices):
        ordered[int(ik)] = momentum.reshape(nk, npairs, 6)[cell]
    return ordered, state_min, state_max


def _velocity_blocks(
    packed_momentum: np.ndarray,
    dimensions: tuple[int, int, int, int],
    state_min: int,
    state_max: int,
) -> np.ndarray:
    nk, nspin, nbands, _nao = dimensions
    blocks = np.zeros((nk, nspin, 3, nbands, nbands), dtype="<c16")
    rows, columns = np.triu_indices(state_max - state_min + 1)
    rows = rows + state_min - 1
    columns = columns + state_min - 1

    # The packed band triangle itself follows the FHI-aims n_state outer,
    # m_state inner loops (the row-major upper-triangle formula in h5_ref_.py).
    # The Fortran-order correction above applies to the three k-grid axes.
    for ik in range(nk):
        gradients = (
            packed_momentum[ik, :, 0::2]
            + 1j * packed_momentum[ik, :, 1::2]
        )
        velocity = -1j * gradients
        for alpha in range(3):
            values = velocity[:, alpha]
            blocks[ik, 0, alpha, columns, rows] = values
            blocks[ik, 0, alpha, rows, columns] = np.conjugate(values)
            diagonal = rows == columns
            blocks[ik, 0, alpha, rows[diagonal], columns[diagonal]] = (
                values[diagonal].real
            )
    blocks *= ATOMIC_TO_EV_ANGSTROM
    return blocks


def convert(input_dir: Path, output_file: Path) -> None:
    input_dir = input_dir.resolve()
    output_file = output_file.absolute()
    if output_file.is_symlink():
        raise ValueError(f"refusing to replace symbolic link: {output_file}")

    dimensions = _read_band_dimensions(input_dir)
    packed, state_min, state_max = _read_momentum(input_dir, dimensions)
    blocks = _velocity_blocks(packed, dimensions, state_min, state_max)
    nk, nspin, nbands, nao = dimensions

    header_bytes = 7 * 4
    table_bytes = nk * (4 + 8)
    block_bytes = nspin * 3 * nbands * nbands * 16
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=output_file.name + ".", suffix=".tmp",
            dir=output_file.parent, delete=False
        ) as output:
            temporary_name = output.name
            output.write(struct.pack(
                "<7i", VELOCITY_V1_MARKER, VELOCITY_V1_COMPLEX_DOUBLE,
                nk, nspin, nbands, nao, 3
            ))
            payload_start = header_bytes + table_bytes
            for ik in range(nk):
                output.write(struct.pack("<iq", ik + 1, payload_start + ik * block_bytes))
            for block in blocks:
                output.write(np.ascontiguousarray(block, dtype="<c16").tobytes(order="C"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, output_file)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="FHI-aims export directory")
    parser.add_argument(
        "output_file", type=Path, nargs="?",
        help="output path (default: INPUT_DIR/velocity_matrix)"
    )
    arguments = parser.parse_args(argv)
    output = arguments.output_file or arguments.input_dir / "velocity_matrix"
    try:
        convert(arguments.input_dir, output)
    except (OSError, KeyError, ValueError) as error:
        print(f"aims_mommat_to_velocity.py: {error}", file=sys.stderr)
        return 1
    print(f"Converted {arguments.input_dir / 'mommat.h5'} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
