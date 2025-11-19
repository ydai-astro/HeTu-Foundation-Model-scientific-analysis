#!/usr/bin/env python
import os
import re
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# Set directories (modify as needed)
dirA = '/groups/hetu_ai/home/share/racs-mid-csv/'   # Directory for A catalog CSV files
dirB = '/groups/hetu_ai/home/share/HeTu/xzj_code/rst/output_internimage_0722/csv/'   # Directory for B catalog CSV files
output_dir1 = 'output_internimage_0722/csv'
output_dir2 = 'output_internimage_0722/txt'

os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

# Regular expression to extract the SB number (supports "SB33098" or "SB_33098")
pattern = re.compile(r"(SB[_]?(\d+))", re.IGNORECASE)

# Loop over CSV files in directory A
for fileA in os.listdir(dirA):
    if not fileA.lower().endswith('.csv'):
        continue
    matchA = pattern.search(fileA)
    if not matchA:
        continue
    sb_num = matchA.group(2)  # Extract numeric part, e.g., "33098"
    
    # In directory B, find a file whose name contains the same sb_num
    candidates = [f for f in os.listdir(dirB) 
                  if f.lower().endswith('.csv') and sb_num in f]
    if not candidates:
        print(f"No matching B file found for {fileA}.")
        continue
    # Use the first matching candidate (adjust strategy if needed)
    fileB = candidates[0]
    
    # Construct full paths for A and B files
    pathA = os.path.join(dirA, fileA)
    pathB = os.path.join(dirB, fileB)
    
    print(f"Processing: A file '{fileA}' and B file '{fileB}'")
    
    # Read CSV files
    dfA = pd.read_csv(pathA)
    dfB = pd.read_csv(pathB)
    
    # For B catalog, filter to keep only records with labels == 1
    dfB0 = dfB[dfB['labels'] == 1].reset_index(drop=True)
    
    # Merge based on A's col_component_id and B's component_id
    merged = pd.merge(dfA, dfB0, left_on='col_component_id', right_on='component_id', 
                      how='inner', suffixes=('_A', '_B'))
    
    # Create SkyCoord objects and compute angular separation between A and B coordinates
    catA = SkyCoord(ra=merged['col_ra_deg_cont'].values * u.deg,
                    dec=merged['col_dec_deg_cont'].values * u.deg)
    catB = SkyCoord(ra=merged['RA'].values * u.deg,
                    dec=merged['Dec'].values * u.deg)
    d2d = catA.separation(catB)
    
    # Set match threshold to 40 arcsec and filter matching records
    match_threshold = 40 * u.arcsec
    mask = d2d < match_threshold
    matched = merged[mask].copy()
    matched['Separation_arcsec'] = d2d[mask].arcsec
    
    # Save matched results to CSV file; use naming convention "matched_catalog_SB_<sb_num>.csv"
    out_matched = os.path.join(output_dir1, f"matched_catalog_SB_{sb_num}.csv")
    matched.to_csv(out_matched, index=False)
    print(f"Matched results saved to '{out_matched}'.")
    
    # Calculate the fraction of matched A sources over all A sources
    total_A = len(dfA)
    matched_count = len(matched)
    match_fraction = matched_count / total_A if total_A > 0 else 0
    
    out_ratio = os.path.join(output_dir2, f"match_ratio_SB_{sb_num}.txt")
    with open(out_ratio, 'w') as f:
        f.write("Matched A sources / Total A sources: {} / {} = {:.2%}\n"
                .format(matched_count, total_A, match_fraction))
    print(f"Match ratio saved to '{out_ratio}'.\n")
