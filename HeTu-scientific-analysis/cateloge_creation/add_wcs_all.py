#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import argparse
import re
import glob
import sys
def log(message, level="INFO"):
    """Logging function with level indicator"""
    print(f"[{level}] {message}", file=sys.stderr, flush=True)

def parse_bbox(bbox_str):
    """Parse bounding box string into a list of floats"""
    bbox_str = bbox_str.strip('[]')
    return [float(s) for s in bbox_str.split(',')]

def validate_bbox(xmin, ymin, xmax, ymax):
    """Validate bounding box coordinates"""
    if xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin:
        return False
    return True

def normalize_ra(ra_deg):
    """Normalize Right Ascension to 0-360 degrees"""
    return ra_deg % 360

def is_ra_within(ra, ra_min, ra_max):
    """Check if RA is within specified range (handles 0-360 boundary)"""
    ra_min_norm = normalize_ra(ra_min)
    ra_max_norm = normalize_ra(ra_max)
    ra_norm = normalize_ra(ra)
    
    if ra_min_norm <= ra_max_norm:
        return ra_min_norm <= ra_norm <= ra_max_norm
    else:
        return (ra_min_norm <= ra_norm <= 360) or (0 <= ra_norm <= ra_max_norm)

def calculate_ra_distance(ra1, ra2):
    """Calculate minimum angular distance between two RAs (0-180 degrees)"""
    ra1_norm = normalize_ra(ra1)
    ra2_norm = normalize_ra(ra2)
    delta = abs(ra1_norm - ra2_norm)
    return min(delta, 360 - delta)

def find_matching_fits(csv_path, fits_parent_dir):
    """Find matching FITS folder and files for a CSV file"""
    csv_basename = os.path.basename(csv_path)
    match = re.match(r'(\d+)\.csv$', csv_basename)
    if not match:
        return {}
    csv_prefix = match.group(1)
    fits_folder = os.path.join(fits_parent_dir, csv_prefix)
    
    if not os.path.isdir(fits_folder):
        return {}
    
    fits_files = glob.glob(os.path.join(fits_folder, '*.fits'))
    fits_map = {}
    for fits_path in fits_files:
        core_id = os.path.splitext(os.path.basename(fits_path))[0]
        fits_map[core_id] = (fits_folder, fits_path)
    return fits_map

def process_csv_file(csv_file, fits_parent_dir, output_dir):
    """Process a single CSV file and convert pixel coordinates to celestial coordinates"""
    csv_data = pd.read_csv(csv_file)
    fits_map = find_matching_fits(csv_file, fits_parent_dir)
    if not fits_map:
        print(f"No matching FITS folder found for: {os.path.basename(csv_file)}")
        return False

    results = []
    matched_records = 0
    missing_fits = 0
    
    for _, row in csv_data.iterrows():
        try:
            component_id = row['component_id']
            core_id = os.path.splitext(component_id)[0]
            if core_id not in fits_map:
                missing_fits += 1
                continue

            fits_folder, fits_path = fits_map[core_id]
            bbox = parse_bbox(str(row['bbox']))
            if len(bbox) < 4:
                continue
            xmin, ymin, xmax, ymax = bbox
            
            if not validate_bbox(xmin, ymin, xmax, ymax):
                continue

            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                if 'CRVAL1' not in header or 'CRVAL2' not in header:
                    continue
                    
                wcs = WCS(header)
                 # 定义内部函数进行坐标转换
                def convert_pixel_to_world(x, y):
                    """Convert pixel coordinates (x, y) to RA/DEC using WCS"""
                    # 对于4D FITS，固定频率和Stokes参数为参考值
                    pix_coords = np.array([[x, y, header['CRPIX3'], header['CRPIX4']]])
                    world_coords = wcs.all_pix2world(pix_coords, 0)
                    return world_coords[0][0], world_coords[0][1]  # 返回RA/DEC

                                # 转换边界框的四个角点
                ra_min, dec_min = convert_pixel_to_world(xmin, ymin)
                ra_max, dec_max = convert_pixel_to_world(xmax, ymax)
                ra_center_bbox, dec_center_bbox = convert_pixel_to_world((xmin+xmax)/2, (ymin+ymax)/2)
                
                # 获取FITS图像的参考中心坐标
                ra_center = header['CRVAL1']
                dec_center = header['CRVAL2']
                
                # 处理RA环绕问题
                if ra_min > ra_max:
                    ra_min, ra_max = ra_max, ra_min
                    
                # 计算与参考中心的位置关系
                is_inside = (ra_min <= ra_center <= ra_max) and (dec_min <= dec_center <= dec_max)
                delta_ra = (ra_center - ra_center_bbox) * np.cos(np.radians(dec_center))
                angular_distance = np.hypot(delta_ra, dec_center - dec_center_bbox)
                
                # 获取频率和Stokes参数
                #freq_val = header['CRVAL3']
                #stokes_val = header['CRVAL4']
                result_row = {
                    **row.to_dict(),
                    'fits_id': core_id,
                    'fits_center_ra': ra_center,
                    'fits_center_dec': dec_center,
                    'bbox_center_ra': ra_center_bbox,
                    'bbox_center_dec': dec_center_bbox,
                    'bbox_ra_min': ra_min,
                    'bbox_ra_max': ra_max,
                    'bbox_dec_min': dec_min,
                    'bbox_dec_max': dec_max,
                    'is_inside_bbox': is_inside,
                    'angular_distance': angular_distance
                }
                results.append(result_row)
                matched_records += 1
                
        except Exception as e:
            print(f"Error processing {component_id}: {str(e)}")
            continue

    if results:
        output_file = os.path.join(output_dir, f"wcs_{os.path.basename(csv_file)}")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Successfully processed {matched_records} records, {missing_fits} missing FITS files")
        return True
    return False

def main():
    """Main function: Batch process CSV files and convert coordinates"""
    parser = argparse.ArgumentParser(description='Batch process CSV files and convert pixel coordinates to celestial coordinates')
    parser.add_argument('-c', '--csv_dir', required=True, help='Directory containing CSV files')
    parser.add_argument('-f', '--fits_parent_dir', required=True, help='Base directory containing FITS folders')
    parser.add_argument('-o', '--output_dir', default='wcs_results', help='Output directory for results')
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(args.csv_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in directory {args.csv_dir}")
    
    success_count = 0
    for csv_file in csv_files:
        print(f"\nProcessing CSV file: {os.path.basename(csv_file)}")
        if process_csv_file(csv_file, args.fits_parent_dir, args.output_dir):
            success_count += 1
    
    print(f"\nBatch processing completed: Successfully processed {success_count}/{len(csv_files)} CSV files")
    print(f"Results saved in directory: {args.output_dir}")

if __name__ == "__main__":
    main()
