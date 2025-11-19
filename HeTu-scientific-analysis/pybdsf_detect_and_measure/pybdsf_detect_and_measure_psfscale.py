#!/usr/bin/env python3
"""
pybdsf_detect_and_measure_psfscale.py

自动用 PyBDSF 在一张 FITS 大图上检测源并测量参数，
rms_box 和 grid 步长随 PSF 的 minor axis (BMIN) 缩放：
 box = 15 × θ_minor
 step = 3 × θ_minor
并启用 atrous (à trous) wavelet 模式。
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits

try:
    import bdsf
except Exception as e:
    print("ERROR: cannot import pybdsf (bdsf). Please install with: pip install pybdsf", file=sys.stderr)
    raise


def parse_args():
    p = argparse.ArgumentParser(description='Run PyBDSF with PSF-scaled rms_box and à trous wavelet')
    p.add_argument('fits', help='Input FITS image')
    p.add_argument('--outdir', default=None, help='Output directory (default: same folder as input)')
    p.add_argument('--thresh_pix', type=float, default=5.0, help='Pixel threshold (sigma) for detection (default 5.0)')
    p.add_argument('--thresh_isl', type=float, default=3.0, help='Island threshold (sigma) (default 3.0)')
    p.add_argument('--adaptive_rms', action='store_true', help='Use adaptive rms box')
    p.add_argument('--group_by_isl', action='store_true', help='Group Gaussians into sources (SRL catalog)')
    p.add_argument('--export_model', action='store_true', help='Export gaussian model and residual images as FITS')
    p.add_argument('--cores', type=int, default=1, help='Number of cores for parallel sections (if splitting)')
    p.add_argument('--verb', action='store_true', help='Verbose output')
    return p.parse_args()


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)


def read_beam_and_cdelt(fitsfile):
    """读取 PSF BMIN 和像素大小 CDELT1"""
    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        bmin_deg = hdr.get('BMIN')    # BMIN 通常是度
        cdelt1_deg = abs(hdr.get('CDELT1'))  # 每像素的度
        if bmin_deg is None or cdelt1_deg is None:
            raise ValueError("FITS header缺少 BMIN 或 CDELT1 关键词")
        return bmin_deg, cdelt1_deg, hdr


def main():
    args = parse_args()
    fitsfile = Path(args.fits)
    if not fitsfile.exists():
        print(f"ERROR: input FITS not found: {fitsfile}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else fitsfile.parent
    outdir = ensure_outdir(outdir)

    base = fitsfile.stem
    if args.verb:
        print(f"Input: {fitsfile}\nOutput dir: {outdir}\nbase: {base}")

    # --- 1. 读取 BMIN 和像素尺度 ---
    bmin_deg, cdelt1_deg, hdr = read_beam_and_cdelt(fitsfile)
    if args.verb:
        print(f"Beam minor axis BMIN = {bmin_deg} deg, pixel scale = {cdelt1_deg} deg/pix")

    # --- 2. 转换为像素并按 (15,3) 比例 ---
    theta_minor_pix = bmin_deg / cdelt1_deg
    box_pix = int(round(15 * theta_minor_pix))
    step_pix = int(round(3 * theta_minor_pix))
    if args.verb:
        print(f"rms_box (box, step) = ({box_pix}, {step_pix}) pixels")

    print("Running PyBDSF with PSF-scaled rms_box and atrous wavelet...")

    img = bdsf.process_image(
        str(fitsfile),
        thresh_pix=args.thresh_pix,
        thresh_isl=args.thresh_isl,
        rms_box=(box_pix, step_pix),
        adaptive_rms_box=args.adaptive_rms,
        atrous_do=True,        # 启用 à trous
        atrous_jmax=3,         # 三个 wavelet 尺度
        group_by_isl=args.group_by_isl,
        verbose=args.verb
    )

    # 输出目录与文件同原版
    gaul_csv = outdir / f"{base}_gaul.csv"
    srl_csv  = outdir / f"{base}_srl.csv"
    img.write_catalog(format='csv', catalog_type='gaul', outfile=str(gaul_csv))
    img.write_catalog(format='csv', catalog_type='srl', outfile=str(srl_csv))

    if args.export_model:
        img.export_image(img_type='gaus_model', outfile=str(outdir / f"{base}_gaus_model.fits"))
        img.export_image(img_type='resid', outfile=str(outdir / f"{base}_resid.fits"))

    print("Done.")
    print(f"  Number of Gaussians fitted: {len(img.gaussians)}")
    if hasattr(img, 'srclist') and img.srclist is not None:
        print(f"  Number of sources (SRL): {len(img.srclist)}")


if __name__ == '__main__':
    main()
