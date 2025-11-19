import pandas as pd
import argparse
import os
import time
from rtree import index
import glob

def is_overlapping_or_containing(box1, box2):
    """Determine if two bounding boxes overlap"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

def process_data(input_df):
    """Process bounding boxes with spatial indexing to remove overlaps"""
    if input_df.empty:
        return input_df
    
    # Sort by score in descending order to prioritize higher scores
    sorted_df = input_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Create R-tree spatial index
    p = index.Property()
    idx = index.Index(properties=p)
    
    # Store indices of final retained boxes
    final_indices = []
    
    # Iterate through each bounding box
    for i, row in sorted_df.iterrows():
        bbox = (row['bbox_ra_min'], row['bbox_dec_min'], 
                row['bbox_ra_max'], row['bbox_dec_max'])
        
        # Query all boxes overlapping with current box
        overlapping_indices = list(idx.intersection(bbox))
        
        # If no overlaps, retain current box
        if not overlapping_indices:
            final_indices.append(i)
            # Add current box to index: (left, bottom, right, top, id)
            idx.insert(i, bbox)
    
    return sorted_df.iloc[final_indices].reset_index(drop=True)

def process_single_csv(input_file, output_dir):
    """Process a single CSV file and save the result"""
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Warning: Skipping non-existent file: {input_file}")
        return False
    
    try:
        start_time = time.time()
        df = pd.read_csv(input_file)
        print(f"Processing {input_file}: Read {len(df)} records")
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return False
    
    # Check required columns
    required_columns = ['bbox_ra_min', 'bbox_ra_max', 'bbox_dec_min', 'bbox_dec_max', 'score']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error in {input_file}: Missing required columns: {', '.join(missing_columns)}")
        return False
    
    # Process data
    try:
        process_start = time.time()
        processed_df = process_data(df)
        process_time = time.time() - process_start
        print(f"  Processed {input_file}: Original={len(df)}, Remaining={len(processed_df)}")
        print(f"  Removed {len(df) - len(processed_df)} overlapping records in {process_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False
    
    # Generate output file path
    file_basename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"processed_{file_basename}")
    
    # Save results
    try:
        os.makedirs(output_dir, exist_ok=True)
        save_start = time.time()
        processed_df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file} in {time.time()-save_start:.2f} seconds")
        print(f"  Total time for {input_file}: {time.time()-start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch process CSV files to remove overlapping bounding boxes')
    parser.add_argument('--input_dir', required=True, help='Input directory containing CSV files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed CSV files')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing (requires joblib)')
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all CSV files in input directory
    csv_files = glob.glob(os.path.join(args.input_dir, '*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in '{args.input_dir}'")
        return
    
    print(f"Found {len(csv_files)} CSV files to process in '{args.input_dir}'")
    
    # Process files (sequential or parallel)
    if args.parallel:
        try:
            from joblib import Parallel, delayed
            print("Using parallel processing...")
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(process_single_csv)(f, args.output_dir) for f in csv_files
            )
            success_count = sum(results)
        except ImportError:
            print("Parallel processing enabled but joblib not installed, falling back to sequential processing")
            success_count = 0
            for f in csv_files:
                if process_single_csv(f, args.output_dir):
                    success_count += 1
    else:
        print("Using sequential processing...")
        success_count = 0
        for i, f in enumerate(csv_files):
            print(f"\nProcessing file {i+1}/{len(csv_files)}:")
            if process_single_csv(f, args.output_dir):
                success_count += 1
    
    print(f"\nBatch processing completed: Successfully processed {success_count}/{len(csv_files)} CSV files")

if __name__ == "__main__":
    main()