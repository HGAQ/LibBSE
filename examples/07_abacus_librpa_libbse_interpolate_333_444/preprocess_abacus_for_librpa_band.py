#!/usr/bin/env python3
"""Preprocess ABACUS output to generate database for LibRPA GW/EXX band calculation

Caveats:
    - can handle spin=1 and 2 and SOC
"""

import pathlib
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np

HA2EV = 27.211386245988


def get_kpoints(kpoint_file):
    kpoints = np.loadtxt(kpoint_file, skiprows=3, usecols=[1, 2, 3])
    assert (len(kpoints.shape) == 2 and kpoints.shape[1] == 3)
    return kpoints


def cleanup(outdir):
    for p in outdir.glob("band_*_k_*.txt"):
        print("Removing {}".format(p))
        p.unlink()

    fn = outdir / "band_kpath_info"
    if fn.exists():
        print("Removing {}".format(fn))
        fn.unlink()


def process_vxc(indir, nkpts, outdir):
    fn = indir / "vxc_out.dat"
    with open(fn, 'r') as h:
        lines_all = h.readlines()
    assert (nkpts == int(lines_all[0].strip()))
    nspins = int(lines_all[1].strip())
    nbands = int(lines_all[2].strip())
    lines_all = lines_all[3:]
    assert (len(lines_all) == nspins * nbands * nkpts)

    for ik in range(nkpts):
        lines_k = lines_all[nspins * nbands * ik:nspins * nbands * (ik + 1)]
        vxc = []
        for l in lines_k:
            x = l.split()[0].split()
            vxc.append(float(x[0]))

        vxc = np.array(vxc).reshape(nspins, nbands)

        with open(outdir / "band_vxc_k_{:05d}.txt".format(ik + 1), 'w') as h:
            for ispin in range(nspins):
                for ib in range(nbands):
                    print("{:8d} {:7d} {:27.16E}"
                          .format(ispin + 1, ib + 1, vxc[ispin, ib]), file=h)
    return nspins


def process_wfc(indir, nkpts, nspin, outdir, use_soc = False):
    nbands = None
    nbasis = None
    for isp in range(nspin):
        for ik in range(nkpts):
            #isk = isp * nkpts + ik
            if use_soc:
                fn = indir / "wfs12k{:d}_nao.txt".format(ik + 1)
            elif(nspin==2):
                fn = indir / "wfs{:d}k{:d}_nao.txt".format(isp + 1, ik + 1)
            elif(nspin==1):
                fn = indir / "wfk{:d}_nao.txt".format(ik + 1)
            else:
                raise ValueError(f"Unexpected value for nspin: {nspin}. Expected 1 or 2 or `use_soc`.")
            with open(fn, 'r') as h:
                lines = h.readlines()
            if nbands is None:
                nbands = int(lines[2].split()[0])
            if nbasis is None:
                nbasis = int(lines[3].split()[0])
            lines = lines[4:]
            indices_band = [i for i, l in enumerate(lines) if l.endswith("(band)\n")]
            assert (len(indices_band) == nbands)
            
            eigs = []
            occs = []
            vecs = []

            for i, index_band in enumerate(indices_band):
                if i != nbands - 1:
                    lines_band = lines[index_band:indices_band[i + 1]]
                else:
                    lines_band = lines[index_band:]
                eigs.append(float(lines_band[1].split()[0]))
                occs.append(float(lines_band[2].split()[0]))
                vecs.extend(map(float, " ".join(lines_band[3:]).replace("\n", " ").split()))

            # convert to numpy array
            eigs = np.array(eigs) * 0.5
            occs = np.array(occs) * nkpts
            vecs = np.array(vecs)
            assert (len(eigs) == nbands)
            assert (len(occs) == nbands)
            assert (len(vecs) == nbands * nbasis * 2)

            mode = 'a'
            if isp == 0:
                mode = 'w'

            with open(outdir / "band_KS_eigenvalue_k_{:05d}.txt".format(ik + 1), mode) as h:
                for ib in range(nbands):
                    print("{:8d} {:7d} {:27.16E} {:27.16E} {:27.16E}"
                          .format(isp + 1, ib + 1, occs[ib], eigs[ib], eigs[ib] * HA2EV), file=h)

            with open(outdir / "band_KS_eigenvector_k_{:05d}.txt".format(ik + 1), mode + 'b') as h:
                vecs.tofile(h, sep='')

    return nbasis, nbands


def write_kpath_info(nspins, nbands, nbasis, kpoints, outdir):
    nkpts = len(kpoints)
    with open(outdir / "band_kpath_info", "w") as h:
        print("{:6d} {:5d} {:5d} {:5d}".format(nbasis, nbands, nspins, nkpts), file=h)

        for kpt in kpoints:
            print("{:18.12f} {:17.12f} {:17.12f}".format(*kpt), file=h)


def _parser():
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument("-i", dest="indir", default="OUT.ABACUS", type=str)
    p.add_argument("-o", dest="outdir", default=".", type=str,
                   help="Output directory for generated files (default: current directory)")
    p.add_argument("--clean", dest="cleanup", action="store_true",
                   help="Clean up generated database")
    return p


def main():
    use_soc = False
    args = _parser().parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.cleanup:
        cleanup(outdir)
        return

    indir = pathlib.Path(args.indir)

    kpoints = get_kpoints(indir / "KPT.info")
    nkpts = len(kpoints)

    nspin = process_vxc(indir, nkpts, outdir)
    nbasis, nbands = process_wfc(indir, nkpts, nspin, outdir, use_soc)

    write_kpath_info(nspin, nbands, nbasis, kpoints, outdir)


if __name__ == '__main__':
    main()
