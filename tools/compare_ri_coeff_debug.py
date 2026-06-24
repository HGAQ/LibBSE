#!/usr/bin/env python3
"""Compare LibBSE RI coefficient debug output with FHI-aims ELSI CSC output.

Current target data:
  reference: ./3MGO_BSE/lvl_tricoeff_tmp_k_*.csc
  mine:      ./RI_coeff_k_debug/RI_coeff_k_real_rank_*_fhi_aims.txt

The FHI-aims debug matrix packs (basis1,basis2) into an ELSI matrix column:
  col = (basis2 - 1) * max_n_basis_sp + basis1
Here basis1 is atom-local in FHI-aims.  The script maps it to LibBSE's global
basis1 with the known 3MGO atom layout: Mg has 9 NAO/43 aux, O has 8 NAO/42 aux.
Only the real part is compared.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_DIR = REPO_ROOT / "3MGO_BSE"
DEFAULT_MINE_DIR = REPO_ROOT / "RI_coeff_k_debug"

CSC_K_RE = re.compile(r"lvl_tricoeff_tmp_k_(\d+)\.csc$")
TEXT_K_ROW_RE = re.compile(r"lvl_tricoeff_tmp_k_(\d+)(?:_row_(\d+))?\.txt$")
ELSI_HEADER_SIZE = 16
ELSI_CMPLX_DATA = 1


@dataclass(frozen=True)
class Entry:
    key: tuple[int, int, int, int]  # (k, aux, basis1, basis2)
    value: float
    source: str
    line: int


@dataclass
class LoadStats:
    parsed: int = 0
    kept: int = 0
    skipped_malformed: int = 0
    skipped_bad_key: int = 0
    skipped_bad_float: int = 0
    skipped_nonfinite: int = 0
    skipped_too_large: int = 0
    skipped_padding_basis: int = 0
    duplicate_same: int = 0
    duplicate_conflict_keys: int = 0
    duplicate_conflict_lines: int = 0


@dataclass
class CscMeta:
    n_by_k: dict[int, int]
    nnz_by_k: dict[int, int]
    max_n_basis_sp: int
    basis_per_atom: list[int]
    aux_per_atom: list[int]
    basis_offsets: list[int]
    aux_offsets: list[int]
    full_basis2: bool = False


@dataclass
class AuxPermutationResult:
    inferred: dict[int, dict[int, int]]
    residual_rows: dict[int, set[int]]
    pair_hits: Counter[tuple[int, int, int]]
    pair_diff_sum: defaultdict[tuple[int, int, int], float]
    sample_count: Counter[tuple[int, int]]
    total_samples: int
    matched_samples: int
    ambiguous_samples: int


@dataclass
class CheckResult:
    checked: int = 0
    correct: int = 0
    wrong: int = 0
    missing: int = 0

    @property
    def percent(self) -> float:
        return 100.0 * self.correct / self.checked if self.checked else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 3MGO_BSE ELSI CSC reference against LibBSE RI debug text.",
    )
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--mine-dir", type=Path, default=DEFAULT_MINE_DIR)
    parser.add_argument(
        "--reference-format",
        choices=("auto", "csc", "text"),
        default="auto",
        help="Reference format. auto uses complete text shards when present, otherwise CSC.",
    )
    parser.add_argument("--csc-glob", default="lvl_tricoeff_tmp_k_*.csc")
    parser.add_argument("--text-glob", default="lvl_tricoeff_tmp_k_*.txt")
    parser.add_argument("--mine-glob", default="RI_coeff_k_real_rank_*_fhi_aims.txt")
    parser.add_argument(
        "--max-n-basis-sp",
        type=int,
        default=None,
        help="max_n_basis_sp used in FHI column packing. Defaults to max basis1 in mine.",
    )
    parser.add_argument(
        "--basis-per-atom",
        default="9,8",
        help="Comma-separated NAO basis count per atom in geometry order.",
    )
    parser.add_argument(
        "--aux-per-atom",
        default="43,42",
        help="Comma-separated auxiliary basis count per atom in geometry order.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--max-reference-abs", type=float, default=2.0)
    parser.add_argument("--max-mine-abs", type=float, default=2.0)
    parser.add_argument("--max-report", type=int, default=20)
    parser.add_argument(
        "--infer-aux-permutation",
        action="store_true",
        help="Diagnose FHI aux-row to LibBSE aux-row ordering by value matching.",
    )
    parser.add_argument(
        "--perm-kpoint",
        type=int,
        default=1,
        help="K point used for aux permutation inference. Use 0 to use all k points.",
    )
    parser.add_argument(
        "--perm-min-abs",
        type=float,
        default=1e-8,
        help="Ignore reference samples smaller than this during permutation inference.",
    )
    parser.add_argument(
        "--infer-o-basis-pair-relocation",
        action="store_true",
        help="Infer O atom (basis1,basis2) relocation after aux permutation and show self-check impact.",
    )
    parser.add_argument(
        "--basis-pair-atom",
        type=int,
        default=2,
        help="1-based atom index whose basis-pair relocation is inferred. Defaults to atom 2 (O in 3MGO).",
    )
    parser.add_argument(
        "--basis-pair-min-abs",
        type=float,
        default=1e-4,
        help="Only use large wrong reference values above this threshold for basis-pair relocation.",
    )
    parser.add_argument(
        "--basis-pair-min-hits",
        type=int,
        default=3,
        help="Minimum hits required to apply an inferred basis-pair relocation.",
    )
    return parser.parse_args()


def read_max_n_basis_sp(case_dir: Path) -> int | None:
    check_path = case_dir / "check.txt"
    if not check_path.exists():
        return None

    pattern = re.compile(r"reso:\s*max_n_basis_sp,n_cells,n_full_freq\s+(\d+)")
    with check_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None


def is_close(a: float, b: float, atol: float, rtol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def count_skip(stats: LoadStats, reason: str) -> None:
    if reason == "malformed":
        stats.skipped_malformed += 1
    elif reason == "bad_key":
        stats.skipped_bad_key += 1
    elif reason == "bad_float":
        stats.skipped_bad_float += 1
    elif reason == "nonfinite":
        stats.skipped_nonfinite += 1
    elif reason == "too_large":
        stats.skipped_too_large += 1
    elif reason == "padding_basis":
        stats.skipped_padding_basis += 1


def parse_int_list(text: str, label: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid {label}: {text}") from exc
    if not values or any(value <= 0 for value in values):
        raise SystemExit(f"invalid {label}: {text}")
    return values


def offsets(counts: list[int]) -> list[int]:
    result = [0]
    for count in counts[:-1]:
        result.append(result[-1] + count)
    return result


def atom_from_aux(row: int, aux_offsets: list[int], aux_per_atom: list[int]) -> int | None:
    for atom, (offset, count) in enumerate(zip(aux_offsets, aux_per_atom)):
        if offset < row <= offset + count:
            return atom
    return None


def parse_real_token(token: str, max_abs: float) -> tuple[float | None, str | None]:
    try:
        value = float(token.replace("D", "E").replace("d", "E"))
    except ValueError:
        return None, "bad_float"
    if not math.isfinite(value):
        return None, "nonfinite"
    if abs(value) > max_abs:
        return None, "too_large"
    return value, None


def read_mine_text(mine_dir: Path, pattern: str, max_abs: float) -> tuple[list[Entry], list[Path], LoadStats]:
    stats = LoadStats()
    entries: list[Entry] = []
    files = sorted(mine_dir.glob(pattern))

    for file in files:
        with file.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                parts = line.split()
                if len(parts) != 5:
                    count_skip(stats, "malformed")
                    continue
                try:
                    key = tuple(int(parts[i]) for i in range(4))
                except ValueError:
                    count_skip(stats, "malformed")
                    continue
                if any(index <= 0 for index in key):
                    count_skip(stats, "bad_key")
                    continue
                value, reason = parse_real_token(parts[4], max_abs)
                if reason is not None:
                    count_skip(stats, reason)
                    continue
                stats.parsed += 1
                entries.append(Entry(key, value, str(file), line_no))

    return entries, files, stats


def parse_csc_k(path: Path) -> int:
    match = CSC_K_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse k point from CSC filename: {path}")
    return int(match.group(1))


def parse_text_k(path: Path) -> int:
    match = TEXT_K_ROW_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse k point from text filename: {path}")
    return int(match.group(1))


def read_one_elsi_csc(path: Path) -> tuple[int, int, list[tuple[int, int, float]]]:
    """Read an ELSI complex CSC file as (n_basis, nnz, [(row,col,real), ...])."""
    data = path.read_bytes()
    header_bytes = ELSI_HEADER_SIZE * 8
    if len(data) < header_bytes:
        raise ValueError(f"ELSI CSC file is too small: {path}")

    header = struct.unpack_from(f"<{ELSI_HEADER_SIZE}q", data, 0)
    data_type = int(header[2])
    n_basis = int(header[3])
    nnz = int(header[5])
    if data_type != ELSI_CMPLX_DATA:
        raise ValueError(f"Expected complex ELSI CSC data in {path}, got type {data_type}")
    if n_basis <= 0 or nnz < 0:
        raise ValueError(f"Invalid ELSI CSC dimensions in {path}: n={n_basis}, nnz={nnz}")

    col_ptr_offset = header_bytes
    row_ind_offset = col_ptr_offset + n_basis * 8
    value_offset = row_ind_offset + nnz * 4
    expected_size = value_offset + nnz * 16
    if len(data) < expected_size:
        raise ValueError(f"Truncated ELSI CSC file {path}: got {len(data)}, need {expected_size}")

    # ELSI stores n_basis column pointers.  The final end pointer is implicit:
    # nnz + 1.  Column pointers are 1-based.
    col_ptr = struct.unpack_from(f"<{n_basis}q", data, col_ptr_offset)
    row_ind = struct.unpack_from(f"<{nnz}i", data, row_ind_offset)

    records: list[tuple[int, int, float]] = []
    for col0 in range(n_basis):
        start = int(col_ptr[col0]) - 1
        end = int(col_ptr[col0 + 1]) - 1 if col0 + 1 < n_basis else nnz
        if start < 0 or end < start or end > nnz:
            raise ValueError(f"Invalid CSC col_ptr in {path}: col={col0 + 1}, start={start}, end={end}")
        for pos in range(start, end):
            real, _imag = struct.unpack_from("<dd", data, value_offset + 16 * pos)
            records.append((int(row_ind[pos]), col0 + 1, real))

    return n_basis, nnz, records


def read_reference_csc(
    case_dir: Path,
    pattern: str,
    max_n_basis_sp: int,
    basis_per_atom: list[int],
    aux_per_atom: list[int],
    max_abs: float,
) -> tuple[list[Entry], list[Path], LoadStats, CscMeta]:
    stats = LoadStats()
    entries: list[Entry] = []
    files = sorted(case_dir.glob(pattern), key=parse_csc_k)
    n_by_k: dict[int, int] = {}
    nnz_by_k: dict[int, int] = {}
    basis_offsets = offsets(basis_per_atom)
    aux_offsets = offsets(aux_per_atom)

    for file in files:
        k_point = parse_csc_k(file)
        n_basis, nnz, records = read_one_elsi_csc(file)
        n_by_k[k_point] = n_basis
        nnz_by_k[k_point] = nnz

        for row, col, value in records:
            if row <= 0 or col <= 0:
                count_skip(stats, "bad_key")
                continue
            if not math.isfinite(value):
                count_skip(stats, "nonfinite")
                continue
            if abs(value) > max_abs:
                count_skip(stats, "too_large")
                continue
            local_basis1 = (col - 1) % max_n_basis_sp + 1
            basis2 = (col - 1) // max_n_basis_sp + 1
            atom = atom_from_aux(row, aux_offsets, aux_per_atom)
            if atom is None:
                count_skip(stats, "bad_key")
                continue
            if local_basis1 > basis_per_atom[atom]:
                count_skip(stats, "padding_basis")
                continue
            basis1 = basis_offsets[atom] + local_basis1
            stats.parsed += 1
            entries.append(Entry((k_point, row, basis1, basis2), value, str(file), col))

    return entries, files, stats, CscMeta(
        n_by_k,
        nnz_by_k,
        max_n_basis_sp,
        basis_per_atom,
        aux_per_atom,
        basis_offsets,
        aux_offsets,
        False,
    )


def read_reference_text(
    case_dir: Path,
    pattern: str,
    max_n_basis_sp: int,
    basis_per_atom: list[int],
    aux_per_atom: list[int],
    max_abs: float,
) -> tuple[list[Entry], list[Path], LoadStats, CscMeta]:
    stats = LoadStats()
    entries: list[Entry] = []
    files = sorted(case_dir.glob(pattern), key=lambda path: (parse_text_k(path), path.name))
    n_by_k: dict[int, int] = {}
    nnz_by_k: dict[int, int] = Counter()
    basis_offsets = offsets(basis_per_atom)
    aux_offsets = offsets(aux_per_atom)
    n_basbas = sum(aux_per_atom)

    for file in files:
        with file.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                parts = line.split()
                if len(parts) < 5:
                    count_skip(stats, "malformed")
                    continue
                try:
                    k_point = int(parts[0])
                    row = int(parts[1])
                    local_basis1 = int(parts[2])
                    basis2 = int(parts[3])
                except ValueError:
                    count_skip(stats, "malformed")
                    continue
                if k_point <= 0 or row <= 0 or local_basis1 <= 0 or basis2 <= 0:
                    count_skip(stats, "bad_key")
                    continue

                value, reason = parse_real_token(parts[4], max_abs)
                if reason is not None:
                    count_skip(stats, reason)
                    continue

                atom = atom_from_aux(row, aux_offsets, aux_per_atom)
                if atom is None:
                    count_skip(stats, "bad_key")
                    continue
                if local_basis1 > basis_per_atom[atom]:
                    count_skip(stats, "padding_basis")
                    continue
                if basis2 > sum(basis_per_atom):
                    count_skip(stats, "bad_key")
                    continue

                basis1 = basis_offsets[atom] + local_basis1
                stats.parsed += 1
                n_by_k[k_point] = n_basbas
                nnz_by_k[k_point] += 1
                entries.append(Entry((k_point, row, basis1, basis2), value, str(file), line_no))

    return entries, files, stats, CscMeta(
        n_by_k,
        dict(nnz_by_k),
        max_n_basis_sp,
        basis_per_atom,
        aux_per_atom,
        basis_offsets,
        aux_offsets,
        True,
    )


def unique_map(entries: list[Entry], stats: LoadStats, atol: float, rtol: float) -> dict[tuple[int, int, int, int], Entry]:
    kept: dict[tuple[int, int, int, int], Entry] = {}
    conflicted: dict[tuple[int, int, int, int], int] = {}

    for entry in entries:
        if entry.key in conflicted:
            conflicted[entry.key] += 1
            continue

        old = kept.get(entry.key)
        if old is None:
            kept[entry.key] = entry
            continue

        if is_close(entry.value, old.value, atol, rtol):
            stats.duplicate_same += 1
            continue

        del kept[entry.key]
        conflicted[entry.key] = 2

    stats.duplicate_conflict_keys = len(conflicted)
    stats.duplicate_conflict_lines = sum(conflicted.values())
    stats.kept = len(kept)
    return kept


def csc_col(key: tuple[int, int, int, int], max_n_basis_sp: int) -> int:
    _k, _row, basis1, basis2 = key
    return (basis2 - 1) * max_n_basis_sp + basis1


def csc_col_from_global_key(key: tuple[int, int, int, int], meta: CscMeta) -> int | None:
    _k, row, basis1, basis2 = key
    atom = atom_from_aux(row, meta.aux_offsets, meta.aux_per_atom)
    if atom is None:
        return None
    local_basis1 = basis1 - meta.basis_offsets[atom]
    if local_basis1 <= 0 or local_basis1 > meta.basis_per_atom[atom]:
        return None
    return (basis2 - 1) * meta.max_n_basis_sp + local_basis1


def key_is_representable(key: tuple[int, int, int, int], meta: CscMeta) -> bool:
    k_point, row, _basis1, basis2 = key
    n_basis = meta.n_by_k.get(k_point)
    if n_basis is None:
        return False
    col = csc_col_from_global_key(key, meta)
    if col is None or not (1 <= row <= n_basis):
        return False
    if meta.full_basis2:
        return 1 <= basis2 <= sum(meta.basis_per_atom)
    return 1 <= col <= n_basis


def hist(keys: list[tuple[int, int, int, int]], index: int) -> list[tuple[int, int]]:
    counts = Counter(key[index] for key in keys)
    return sorted(counts.items())


def print_stats(label: str, stats: LoadStats) -> None:
    print(f"  {label} parsed real records:      {stats.parsed}")
    print(f"  {label} kept unique records:      {stats.kept}")
    print(f"  {label} skipped malformed:        {stats.skipped_malformed}")
    print(f"  {label} skipped bad key:          {stats.skipped_bad_key}")
    print(f"  {label} skipped bad float:        {stats.skipped_bad_float}")
    print(f"  {label} skipped nonfinite:        {stats.skipped_nonfinite}")
    print(f"  {label} skipped too large:        {stats.skipped_too_large}")
    print(f"  {label} skipped padding basis:    {stats.skipped_padding_basis}")
    print(f"  {label} duplicate same:           {stats.duplicate_same}")
    print(f"  {label} duplicate conflict keys:  {stats.duplicate_conflict_keys}")
    print(f"  {label} duplicate conflict lines: {stats.duplicate_conflict_lines}")


def print_key_examples(title: str, keys: list[tuple[int, int, int, int]], entries: dict[tuple[int, int, int, int], Entry], max_report: int) -> None:
    if not keys:
        return
    print("")
    print(f"{title}:")
    for key in keys[:max_report]:
        entry = entries[key]
        print(f"  key={key} value={entry.value:.12e} source={entry.source}:{entry.line}")


def atom_aux_range(atom: int, meta: CscMeta) -> range:
    start = meta.aux_offsets[atom] + 1
    stop = start + meta.aux_per_atom[atom]
    return range(start, stop)


def atom_basis_range(atom: int, meta: CscMeta) -> range:
    start = meta.basis_offsets[atom] + 1
    stop = start + meta.basis_per_atom[atom]
    return range(start, stop)


def atom_from_basis(basis: int, basis_offsets: list[int], basis_per_atom: list[int]) -> int | None:
    for atom, (offset, count) in enumerate(zip(basis_offsets, basis_per_atom)):
        if offset < basis <= offset + count:
            return atom
    return None


def build_aux_permutation(
    ref_map: dict[tuple[int, int, int, int], Entry],
    mine_map: dict[tuple[int, int, int, int], Entry],
    meta: CscMeta,
    k_filter: int,
    min_abs: float,
    atol: float,
    rtol: float,
) -> None:
    # For a fixed (k,basis1,basis2), only aux ordering should differ.  Matching
    # values inside the same atom gives a direct FHI-row -> LibBSE-row signal.
    mine_by_column: dict[tuple[int, int, int], list[Entry]] = defaultdict(list)
    for entry in mine_map.values():
        k_point, _aux, basis1, basis2 = entry.key
        if k_filter and k_point != k_filter:
            continue
        mine_by_column[(k_point, basis1, basis2)].append(entry)

    pair_hits: Counter[tuple[int, int, int]] = Counter()
    pair_diff_sum: defaultdict[tuple[int, int, int], float] = defaultdict(float)
    sample_count: Counter[tuple[int, int]] = Counter()
    total_samples = 0
    matched_samples = 0
    ambiguous_samples = 0

    for ref in ref_map.values():
        k_point, ref_aux, basis1, basis2 = ref.key
        if k_filter and k_point != k_filter:
            continue
        if abs(ref.value) < min_abs:
            continue
        atom = atom_from_aux(ref_aux, meta.aux_offsets, meta.aux_per_atom)
        if atom is None:
            continue

        total_samples += 1
        sample_count[(atom, ref_aux)] += 1
        candidates: list[tuple[float, Entry]] = []
        allowed_aux = set(atom_aux_range(atom, meta))
        for mine in mine_by_column.get((k_point, basis1, basis2), []):
            mine_aux = mine.key[1]
            if mine_aux not in allowed_aux:
                continue
            diff = abs(mine.value - ref.value)
            if is_close(mine.value, ref.value, atol, rtol):
                candidates.append((diff, mine))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0])
        best_diff, best = candidates[0]
        if len(candidates) > 1:
            ambiguous_samples += 1
        matched_samples += 1
        pair_key = (atom, ref_aux, best.key[1])
        pair_hits[pair_key] += 1
        pair_diff_sum[pair_key] += best_diff

    inferred: dict[int, dict[int, int]] = {atom: {} for atom in range(len(meta.aux_per_atom))}
    residual_rows: dict[int, set[int]] = {atom: set() for atom in range(len(meta.aux_per_atom))}
    for atom in range(len(meta.aux_per_atom)):
        pairs = [
            (hits, pair_diff_sum[(pair_atom, ref_aux, mine_aux)] / hits, ref_aux, mine_aux)
            for (pair_atom, ref_aux, mine_aux), hits in pair_hits.items()
            if pair_atom == atom and hits > 0
        ]
        pairs.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

        used_ref: set[int] = set()
        used_mine: set[int] = set()
        for _hits, _mean_diff, ref_aux, mine_aux in pairs:
            if ref_aux in used_ref or mine_aux in used_mine:
                continue
            inferred[atom][ref_aux] = mine_aux
            used_ref.add(ref_aux)
            used_mine.add(mine_aux)

        # If value matching leaves the same number of unmatched source and target
        # rows, keep the permutation one-to-one by filling those rows in order.
        # These residual rows are diagnostic only and should be verified later.
        aux_rows = list(atom_aux_range(atom, meta))
        missing_ref = [aux for aux in aux_rows if aux not in used_ref]
        unused_mine = [aux for aux in aux_rows if aux not in used_mine]
        if len(missing_ref) == len(unused_mine):
            for ref_aux, mine_aux in zip(missing_ref, unused_mine):
                inferred[atom][ref_aux] = mine_aux
                residual_rows[atom].add(ref_aux)

    return AuxPermutationResult(
        inferred,
        residual_rows,
        pair_hits,
        pair_diff_sum,
        sample_count,
        total_samples,
        matched_samples,
        ambiguous_samples,
    )


def infer_aux_permutation(
    ref_map: dict[tuple[int, int, int, int], Entry],
    mine_map: dict[tuple[int, int, int, int], Entry],
    meta: CscMeta,
    k_filter: int,
    min_abs: float,
    atol: float,
    rtol: float,
) -> None:
    result = build_aux_permutation(ref_map, mine_map, meta, k_filter, min_abs, atol, rtol)

    print("")
    print("Aux permutation inference")
    print("  method: same (k,basis1,basis2), same atom, real-value match")
    print(f"  k point: {k_filter if k_filter else 'all'}")
    print(f"  min abs used for matching: {min_abs:g}")
    print(f"  samples considered:       {result.total_samples}")
    print(f"  samples matched:          {result.matched_samples}")
    print(f"  ambiguous samples:        {result.ambiguous_samples}")
    print("")

    for atom in range(len(meta.aux_per_atom)):
        aux_rows = list(atom_aux_range(atom, meta))
        missing = [aux for aux in aux_rows if aux not in result.inferred[atom]]
        print(f"Atom {atom + 1} aux rows {aux_rows[0]}..{aux_rows[-1]}")
        print("  FHI_aux -> LibBSE_aux    hits/usable    coverage    mean_abs_diff    status")

        for ref_aux in aux_rows:
            mine_aux = result.inferred[atom].get(ref_aux)
            usable = result.sample_count[(atom, ref_aux)]
            if mine_aux is None:
                print(f"  {ref_aux:7d} -> {'?':>10}    {0:4d}/{usable:<6d}    {0.0:7.2%}    {'-':>13}    none")
                continue

            pair_key = (atom, ref_aux, mine_aux)
            hits = result.pair_hits[pair_key]
            coverage = hits / usable if usable else 0.0
            mean_diff = result.pair_diff_sum[pair_key] / hits if hits else 0.0
            if ref_aux in result.residual_rows[atom]:
                status = "residual"
            else:
                status = "strong" if hits >= 3 and coverage >= 0.50 else "weak"
            print(
                f"  {ref_aux:7d} -> {mine_aux:10d}    "
                f"{hits:4d}/{usable:<6d}    {coverage:7.2%}    {mean_diff:13.6e}    {status}"
            )

        target_counts = Counter(result.inferred[atom].values())
        duplicate_targets = sorted(aux for aux, count in target_counts.items() if count > 1)
        if missing:
            print(f"  missing FHI rows: {', '.join(str(aux) for aux in missing)}")
        if duplicate_targets:
            print(f"  duplicate LibBSE targets: {', '.join(str(aux) for aux in duplicate_targets)}")
        permutation = [str(result.inferred[atom].get(aux, "?")) for aux in aux_rows]
        print("  suggested FHI->LibBSE rows:")
        print(f"  {','.join(permutation)}")
        print("")

    validate_aux_permutation(ref_map, mine_map, result.inferred, meta, k_filter, atol, rtol)


def validate_aux_permutation(
    ref_map: dict[tuple[int, int, int, int], Entry],
    mine_map: dict[tuple[int, int, int, int], Entry],
    inferred: dict[int, dict[int, int]],
    meta: CscMeta,
    k_filter: int,
    atol: float,
    rtol: float,
) -> None:
    checked = 0
    correct = 0
    wrong = 0
    missing = 0

    for ref in ref_map.values():
        k_point, ref_aux, basis1, basis2 = ref.key
        if k_filter and k_point != k_filter:
            continue
        atom = atom_from_aux(ref_aux, meta.aux_offsets, meta.aux_per_atom)
        if atom is None:
            continue
        mine_aux = inferred.get(atom, {}).get(ref_aux)
        if mine_aux is None:
            continue

        checked += 1
        mine = mine_map.get((k_point, mine_aux, basis1, basis2))
        if mine is None:
            missing += 1
            continue
        if is_close(mine.value, ref.value, atol, rtol):
            correct += 1
        else:
            wrong += 1

    percent = 100.0 * correct / checked if checked else 0.0
    print("")
    print("Permutation self-check")
    print(f"  checked reference records: {checked}")
    print(f"  correct after mapping:     {correct} ({percent:.6f}%)")
    print(f"  wrong after mapping:       {wrong}")
    print(f"  missing after mapping:     {missing}")


def check_reference_against_mine(
    ref_map: dict[tuple[int, int, int, int], Entry],
    mine_map: dict[tuple[int, int, int, int], Entry],
    aux_map: dict[int, dict[int, int]],
    meta: CscMeta,
    k_filter: int,
    atol: float,
    rtol: float,
    target_atom: int | None = None,
    pair_map: dict[tuple[int, int], tuple[int, int]] | None = None,
    aux_pair_map: dict[tuple[int, int, int], tuple[int, int]] | None = None,
) -> CheckResult:
    result = CheckResult()

    for ref in ref_map.values():
        k_point, ref_aux, basis1, basis2 = ref.key
        if k_filter and k_point != k_filter:
            continue
        aux_atom = atom_from_aux(ref_aux, meta.aux_offsets, meta.aux_per_atom)
        if aux_atom is None:
            continue
        mine_aux = aux_map.get(aux_atom, {}).get(ref_aux)
        if mine_aux is None:
            continue

        mine_basis1 = basis1
        mine_basis2 = basis2
        if target_atom is not None and aux_atom == target_atom:
            mapped_pair = None
            if aux_pair_map is not None:
                mapped_pair = aux_pair_map.get((ref_aux, basis1, basis2))
            if mapped_pair is None and pair_map is not None:
                mapped_pair = pair_map.get((basis1, basis2))
            if mapped_pair is not None:
                mine_basis1, mine_basis2 = mapped_pair

        result.checked += 1
        mine = mine_map.get((k_point, mine_aux, mine_basis1, mine_basis2))
        if mine is None:
            result.missing += 1
            continue
        if is_close(mine.value, ref.value, atol, rtol):
            result.correct += 1
        else:
            result.wrong += 1

    return result


def print_check_result(label: str, result: CheckResult) -> None:
    print(f"  {label} checked: {result.checked}")
    print(f"  {label} correct: {result.correct} ({result.percent:.6f}%)")
    print(f"  {label} wrong:   {result.wrong}")
    print(f"  {label} missing: {result.missing}")


def collect_wrong_after_mapping(
    ref_map: dict[tuple[int, int, int, int], Entry],
    mine_map: dict[tuple[int, int, int, int], Entry],
    aux_map: dict[int, dict[int, int]],
    meta: CscMeta,
    k_filter: int,
    atol: float,
    rtol: float,
    target_atom: int,
    pair_map: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[float, Entry, Entry | None, tuple[int, int]]]:
    wrong: list[tuple[float, Entry, Entry | None, tuple[int, int]]] = []

    for ref in ref_map.values():
        k_point, ref_aux, basis1, basis2 = ref.key
        if k_filter and k_point != k_filter:
            continue
        aux_atom = atom_from_aux(ref_aux, meta.aux_offsets, meta.aux_per_atom)
        if aux_atom is None:
            continue
        mine_aux = aux_map.get(aux_atom, {}).get(ref_aux)
        if mine_aux is None:
            continue
        mapped_pair = pair_map.get((basis1, basis2), (basis1, basis2)) if aux_atom == target_atom else (basis1, basis2)
        mine = mine_map.get((k_point, mine_aux, mapped_pair[0], mapped_pair[1]))
        if mine is None:
            wrong.append((math.inf, ref, None, mapped_pair))
            continue
        diff = abs(mine.value - ref.value)
        if not is_close(mine.value, ref.value, atol, rtol):
            wrong.append((diff, ref, mine, mapped_pair))

    return wrong


def infer_o_basis_pair_relocation(
    ref_map: dict[tuple[int, int, int, int], Entry],
    mine_map: dict[tuple[int, int, int, int], Entry],
    meta: CscMeta,
    k_filter: int,
    aux_min_abs: float,
    pair_min_abs: float,
    min_hits: int,
    target_atom_1based: int,
    max_report: int,
    atol: float,
    rtol: float,
) -> None:
    target_atom = target_atom_1based - 1
    if target_atom < 0 or target_atom >= len(meta.aux_per_atom):
        raise SystemExit(f"--basis-pair-atom is outside atom range: {target_atom_1based}")

    aux_result = build_aux_permutation(ref_map, mine_map, meta, k_filter, aux_min_abs, atol, rtol)
    aux_map = aux_result.inferred

    target_basis = set(atom_basis_range(target_atom, meta))
    target_aux = set(atom_aux_range(target_atom, meta))

    mine_by_k_aux: dict[tuple[int, int], list[Entry]] = defaultdict(list)
    for mine in mine_map.values():
        k_point, mine_aux, basis1, _basis2 = mine.key
        if k_filter and k_point != k_filter:
            continue
        if mine_aux not in target_aux or basis1 not in target_basis:
            continue
        if abs(mine.value) < pair_min_abs:
            continue
        mine_by_k_aux[(k_point, mine_aux)].append(mine)

    pair_hits: Counter[tuple[tuple[int, int], tuple[int, int]]] = Counter()
    aux_pair_hits: Counter[tuple[int, tuple[int, int], tuple[int, int]]] = Counter()
    direct_pair_stat: Counter[tuple[tuple[int, int], str]] = Counter()
    direct_aux_pair_stat: Counter[tuple[tuple[int, int, int], str]] = Counter()
    large_wrong = 0
    large_wrong_with_candidate = 0
    ambiguous_large_wrong = 0

    for ref in ref_map.values():
        k_point, ref_aux, basis1, basis2 = ref.key
        if k_filter and k_point != k_filter:
            continue
        aux_atom = atom_from_aux(ref_aux, meta.aux_offsets, meta.aux_per_atom)
        if aux_atom != target_atom:
            continue
        if basis1 not in target_basis:
            continue
        mine_aux = aux_map.get(aux_atom, {}).get(ref_aux)
        if mine_aux is None:
            continue

        direct = mine_map.get((k_point, mine_aux, basis1, basis2))
        direct_ok = direct is not None and is_close(direct.value, ref.value, atol, rtol)
        ref_pair = (basis1, basis2)
        direct_pair_stat[(ref_pair, "correct" if direct_ok else "wrong")] += 1
        direct_aux_pair_stat[((ref_aux, basis1, basis2), "correct" if direct_ok else "wrong")] += 1

        if direct_ok or abs(ref.value) < pair_min_abs:
            continue

        large_wrong += 1
        basis2_atom = atom_from_basis(basis2, meta.basis_offsets, meta.basis_per_atom)
        candidates: list[Entry] = []
        for mine in mine_by_k_aux.get((k_point, mine_aux), []):
            _mk, _ma, mine_basis1, mine_basis2 = mine.key
            if atom_from_basis(mine_basis2, meta.basis_offsets, meta.basis_per_atom) != basis2_atom:
                continue
            if is_close(mine.value, ref.value, atol, rtol):
                candidates.append(mine)

        if not candidates:
            continue

        candidates.sort(key=lambda entry: (entry.key[2], entry.key[3]))
        if len(candidates) > 1:
            ambiguous_large_wrong += 1
        large_wrong_with_candidate += 1
        for mine in candidates:
            mine_pair = (mine.key[2], mine.key[3])
            pair_hits[(ref_pair, mine_pair)] += 1
            aux_pair_hits[(ref_aux, ref_pair, mine_pair)] += 1

    pair_map: dict[tuple[int, int], tuple[int, int]] = {}
    for ref_pair in sorted({key[0] for key in pair_hits}):
        choices = [
            (hits, mine_pair)
            for (candidate_ref_pair, mine_pair), hits in pair_hits.items()
            if candidate_ref_pair == ref_pair
        ]
        choices.sort(key=lambda item: (-item[0], item[1]))
        if choices and choices[0][0] >= min_hits:
            pair_map[ref_pair] = choices[0][1]

    aux_pair_map: dict[tuple[int, int, int], tuple[int, int]] = {}
    for aux_ref_pair in sorted({(ref_aux, ref_pair) for ref_aux, ref_pair, _mine_pair in aux_pair_hits}):
        ref_aux, ref_pair = aux_ref_pair
        choices = [
            (hits, mine_pair)
            for (candidate_ref_aux, candidate_ref_pair, mine_pair), hits in aux_pair_hits.items()
            if candidate_ref_aux == ref_aux and candidate_ref_pair == ref_pair
        ]
        choices.sort(key=lambda item: (-item[0], item[1]))
        if choices and choices[0][0] >= min_hits:
            aux_pair_map[(ref_aux, ref_pair[0], ref_pair[1])] = choices[0][1]

    aux_only = check_reference_against_mine(
        ref_map, mine_map, aux_map, meta, k_filter, atol, rtol,
    )
    pair_only = check_reference_against_mine(
        ref_map, mine_map, aux_map, meta, k_filter, atol, rtol,
        target_atom=target_atom, pair_map=pair_map,
    )
    aux_pair = check_reference_against_mine(
        ref_map, mine_map, aux_map, meta, k_filter, atol, rtol,
        target_atom=target_atom, aux_pair_map=aux_pair_map,
    )

    print("")
    print("O basis-pair relocation inference")
    print(f"  target atom:              {target_atom_1based}")
    print(f"  k point:                  {k_filter if k_filter else 'all'}")
    print(f"  aux inference min abs:    {aux_min_abs:g}")
    print(f"  basis-pair min abs:       {pair_min_abs:g}")
    print(f"  basis-pair min hits:      {min_hits}")
    print(f"  large direct wrong:       {large_wrong}")
    print(f"  large wrong relocated:    {large_wrong_with_candidate}")
    print(f"  ambiguous relocated:      {ambiguous_large_wrong}")
    print(f"  pair-only map entries:    {len(pair_map)}")
    print(f"  aux+pair map entries:     {len(aux_pair_map)}")
    print("")
    print("Self-check against explicit reference records")
    print_check_result("aux-only", aux_only)
    print_check_result("pair-only", pair_only)
    print_check_result("aux+pair", aux_pair)

    if pair_hits:
        print("")
        print(f"Top {min(max_report, len(pair_hits))} pair-only relocation candidates:")
        for (ref_pair, mine_pair), hits in pair_hits.most_common(max_report):
            direct_ok = direct_pair_stat[(ref_pair, "correct")]
            direct_wrong = direct_pair_stat[(ref_pair, "wrong")]
            chosen = "chosen" if pair_map.get(ref_pair) == mine_pair else ""
            print(
                f"  {ref_pair} -> {mine_pair} hits={hits} "
                f"direct_ok={direct_ok} direct_wrong={direct_wrong} {chosen}"
            )

    if aux_pair_hits:
        print("")
        print(f"Top {min(max_report, len(aux_pair_hits))} aux+pair relocation candidates:")
        for (ref_aux, ref_pair, mine_pair), hits in aux_pair_hits.most_common(max_report):
            direct_ok = direct_aux_pair_stat[((ref_aux, ref_pair[0], ref_pair[1]), "correct")]
            direct_wrong = direct_aux_pair_stat[((ref_aux, ref_pair[0], ref_pair[1]), "wrong")]
            chosen = "chosen" if aux_pair_map.get((ref_aux, ref_pair[0], ref_pair[1])) == mine_pair else ""
            print(
                f"  aux={ref_aux} {ref_pair} -> {mine_pair} hits={hits} "
                f"direct_ok={direct_ok} direct_wrong={direct_wrong} {chosen}"
            )

    residual_wrong = collect_wrong_after_mapping(
        ref_map, mine_map, aux_map, meta, k_filter, atol, rtol, target_atom, pair_map,
    )
    if residual_wrong:
        print("")
        print(f"Top {min(max_report, len(residual_wrong))} residual wrong after pair-only mapping:")
        for diff, ref, mine, mapped_pair in sorted(residual_wrong, key=lambda item: item[0], reverse=True)[:max_report]:
            mine_text = "missing" if mine is None else f"{mine.key} {mine.value:.12e}"
            diff_text = "missing" if math.isinf(diff) else f"{diff:.12e}"
            print(
                f"  diff={diff_text} ref={ref.key} {ref.value:.12e} "
                f"mapped_pair={mapped_pair} mine={mine_text}"
            )

        print("")
        print("Residual wrong histogram after pair-only mapping:")
        aux_hist = Counter(ref.key[1] for _diff, ref, _mine, _mapped_pair in residual_wrong)
        pair_hist = Counter((ref.key[2], ref.key[3]) for _diff, ref, _mine, _mapped_pair in residual_wrong)
        print("  by aux:  " + " ".join(f"{aux}:{count}" for aux, count in aux_hist.most_common(max_report)))
        print("  by pair: " + " ".join(f"{pair}:{count}" for pair, count in pair_hist.most_common(max_report)))


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    mine_dir = args.mine_dir.resolve()
    if not case_dir.is_dir():
        raise SystemExit(f"case dir not found: {case_dir}")
    if not mine_dir.is_dir():
        raise SystemExit(f"mine dir not found: {mine_dir}")

    mine_entries, mine_files, mine_stats = read_mine_text(mine_dir, args.mine_glob, args.max_mine_abs)
    if not mine_files:
        raise SystemExit(f"no mine files found: {mine_dir}/{args.mine_glob}")
    if not mine_entries:
        raise SystemExit("mine files exist but no usable records were parsed")

    max_n_basis_sp = args.max_n_basis_sp or read_max_n_basis_sp(case_dir)
    if max_n_basis_sp is None:
        raise SystemExit(
            "--max-n-basis-sp was not given and max_n_basis_sp could not be read from check.txt"
        )
    basis_per_atom = parse_int_list(args.basis_per_atom, "--basis-per-atom")
    aux_per_atom = parse_int_list(args.aux_per_atom, "--aux-per-atom")
    if len(basis_per_atom) != len(aux_per_atom):
        raise SystemExit("--basis-per-atom and --aux-per-atom must have the same length")

    text_files_probe = sorted(case_dir.glob(args.text_glob))
    reference_format = args.reference_format
    if reference_format == "auto":
        reference_format = "text" if text_files_probe else "csc"

    if reference_format == "text":
        ref_entries, ref_files, ref_stats, csc_meta = read_reference_text(
            case_dir,
            args.text_glob,
            max_n_basis_sp,
            basis_per_atom,
            aux_per_atom,
            args.max_reference_abs,
        )
    else:
        ref_entries, ref_files, ref_stats, csc_meta = read_reference_csc(
            case_dir,
            args.csc_glob,
            max_n_basis_sp,
            basis_per_atom,
            aux_per_atom,
            args.max_reference_abs,
        )
    if not ref_files:
        pattern = args.text_glob if reference_format == "text" else args.csc_glob
        raise SystemExit(f"no {reference_format} reference files found: {case_dir}/{pattern}")

    mine_map = unique_map(mine_entries, mine_stats, args.atol, args.rtol)
    ref_map = unique_map(ref_entries, ref_stats, args.atol, args.rtol)

    if args.infer_aux_permutation:
        infer_aux_permutation(
            ref_map,
            mine_map,
            csc_meta,
            args.perm_kpoint,
            args.perm_min_abs,
            args.atol,
            args.rtol,
        )
        return 0

    if args.infer_o_basis_pair_relocation:
        infer_o_basis_pair_relocation(
            ref_map,
            mine_map,
            csc_meta,
            args.perm_kpoint,
            args.perm_min_abs,
            args.basis_pair_min_abs,
            args.basis_pair_min_hits,
            args.basis_pair_atom,
            args.max_report,
            args.atol,
            args.rtol,
        )
        return 0

    mine_keys = set(mine_map)
    ref_keys = set(ref_map)
    comparable_keys = sorted(key for key in mine_keys if key_is_representable(key, csc_meta))
    mine_only_keys = sorted(key for key in mine_keys if not key_is_representable(key, csc_meta))
    ref_only_keys = sorted(ref_keys - mine_keys)

    correct: list[tuple[float, Entry, Entry]] = []
    wrong: list[tuple[float, Entry, Entry]] = []
    wrong_against_implicit_zero = 0

    for key in comparable_keys:
        mine = mine_map[key]
        ref = ref_map.get(key)
        if ref is None:
            ref = Entry(key, 0.0, "<implicit CSC zero>", 0)
            if not is_close(mine.value, 0.0, args.atol, args.rtol):
                wrong_against_implicit_zero += 1

        diff = abs(mine.value - ref.value)
        if is_close(mine.value, ref.value, args.atol, args.rtol):
            correct.append((diff, mine, ref))
        else:
            wrong.append((diff, mine, ref))

    common_count = len(comparable_keys)
    correct_percent = 100.0 * len(correct) / common_count if common_count else 0.0
    wrong_percent = 100.0 * len(wrong) / common_count if common_count else 0.0

    csc_dims = sorted(set(csc_meta.n_by_k.values()))
    represented_basis2 = sum(basis_per_atom) if csc_meta.full_basis2 else min(csc_dims) // max_n_basis_sp if csc_dims else 0
    max_mine_basis2 = max(entry.key[3] for entry in mine_entries)
    total_ref_nnz = sum(csc_meta.nnz_by_k.values())

    print("3MGO_BSE RI coefficient real-part compare")
    print(f"  reference format:       {reference_format}")
    print(f"  reference dir:          {case_dir}")
    print(f"  reference files:        {len(ref_files)}")
    print(f"  reference matrix dims:  {csc_dims}")
    print(f"  reference records:      {total_ref_nnz}")
    print(f"  mine debug dir:         {mine_dir}")
    print(f"  mine debug files:       {len(mine_files)}")
    print(f"  max_n_basis_sp:         {max_n_basis_sp}")
    print(f"  basis per atom:         {basis_per_atom}")
    print(f"  aux per atom:           {aux_per_atom}")
    print(f"  represented basis2:     1..{represented_basis2}")
    print(f"  mine max basis2:        {max_mine_basis2}")
    print(f"  tolerance:              atol={args.atol:g}, rtol={args.rtol:g}")
    print("")
    print("Reference filtering")
    print_stats("reference", ref_stats)
    print("")
    print("Mine filtering")
    print_stats("mine", mine_stats)
    print("")
    print("Summary")
    print(f"  comparable mine keys:             {common_count}")
    print(f"  mine correct in comparable part:  {len(correct)} ({correct_percent:.6f}%)")
    print(f"  mine wrong in comparable part:    {len(wrong)} ({wrong_percent:.6f}%)")
    print(f"  wrong vs implicit CSC zero:       {wrong_against_implicit_zero}")
    print(f"  mine has, reference cannot hold:  {len(mine_only_keys)}")
    print(f"  reference explicit, mine missing: {len(ref_only_keys)}")

    if mine_only_keys:
        print("")
        print("Mine-only histogram by basis2:")
        print("  " + ", ".join(f"basis_2 = {basis2}:{count}\n" for basis2, count in hist(mine_only_keys, 3)))
    if wrong:
        wrong_keys = [mine.key for _diff, mine, _ref in wrong]
        print("")
        print("Wrong histogram by basis2:")
        print("  " + " ".join(f"basis_2 = {basis2}:{count}\n" for basis2, count in hist(wrong_keys, 3)))
    if wrong:
        wrong_keys = [mine.key for _diff, mine, _ref in wrong]
        print("")
        print("Wrong histogram by kpoint:")
        print("  " + " ".join(f"kpoint = {kpoint}:{count}\n" for kpoint, count in hist(wrong_keys, 0)))
    if wrong:
        wrong_keys = [mine.key for _diff, mine, _ref in wrong]
        print("")
        print("Wrong histogram by basis_1:")
        print("  " + " ".join(f"basis_1 = {kpoint}:{count}\n" for kpoint, count in hist(wrong_keys, 2)))
    
    if wrong:
        wrong_keys = [mine.key for _diff, mine, _ref in wrong]
        print("")
        print("Wrong histogram by aux_basis_1:")
        print("  " + " ".join(f"aux_basis_1 = {kpoint}:{count}\n" for kpoint, count in hist(wrong_keys, 1)))
    if wrong:
        print("")
        print(f"Top {min(args.max_report, len(wrong))} wrong entries by abs diff:")
        for diff, mine, ref in sorted(wrong, key=lambda item: item[0], reverse=True)[: args.max_report]:
            print(
                f"  key={mine.key} diff={diff:.12e} "
                f"mine={mine.value:.12e} ref={ref.value:.12e} "
                f"mine={mine.source}:{mine.line} ref={ref.source}:{ref.line}"
            )

    print_key_examples(
        f"First {min(args.max_report, len(mine_only_keys))} mine keys reference cannot hold",
        mine_only_keys,
        mine_map,
        args.max_report,
    )
    print_key_examples(
        f"First {min(args.max_report, len(ref_only_keys))} explicit reference keys missing in mine",
        ref_only_keys,
        ref_map,
        args.max_report,
    )

    return 1 if wrong or mine_only_keys or ref_only_keys else 0


if __name__ == "__main__":
    raise SystemExit(main())
