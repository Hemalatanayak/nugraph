import os, sys
sys.path.append('/home/hnayak/.conda/envs/NugraphBase/lib/python3.10/site-packages')
import os
import sys
import argparse
import h5py
import numpy as np

# ==========================================
# 1. Helper Functions for Peak Extraction
# ==========================================
def load_and_aggregate(file_path):
    print(f"Loading {file_path} for aggregate statistics...")
    with h5py.File(file_path, 'r') as file:
        dataset_group = file['dataset']
        
        combined_time = []
        combined_rms = []
        planes = ['u', 'v', 'y']
        
        for key in dataset_group.keys():
            data = dataset_group[key][()]
            for plane in planes:
                # Apply semantic mask
                valid_mask = data[f'{plane}/y_semantic'] > -1
                combined_time.append(data[f'{plane}/pos'][valid_mask, 1])
                combined_rms.append(data[f'{plane}/x'][valid_mask, 1])
                
    return np.concatenate(combined_time), np.concatenate(combined_rms)


def get_peak_coordinates(time_data, rms_data, y_bins, x_bins, x_centers):
    peak_t = []
    peak_r = []
    
    for i in range(len(y_bins) - 1):
        t_start, t_end = y_bins[i], y_bins[i+1]
        mask = (time_data >= t_start) & (time_data < t_end)
        
        if np.sum(mask) > 50: 
            slice_rms = rms_data[mask]
            hist, _ = np.histogram(slice_rms, bins=x_bins)
            max_bin_index = np.argmax(hist)
            peak_r.append(x_centers[max_bin_index])
            peak_t.append((t_start + t_end) / 2.0)
            
    return np.array(peak_t), np.array(peak_r)


# ==========================================
# 2. Main Rescaling & Rewriting Function
# ==========================================
def rescale_and_save_hdf5(reference_file, target_file, output_file):
    # Define bins
    y_bins = np.linspace(0, 400, 15)
    x_bins = np.linspace(0, 10, 60)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    planes = ['u', 'v', 'y']

    # --- Step A: Calculate Global Linear Fits ---
    print("--- Step 1: Calculating Linear Fits ---")
    time_1, rms_1 = load_and_aggregate(reference_file)
    time_2, rms_2 = load_and_aggregate(target_file)
    
    t1_peaks, r1_peaks = get_peak_coordinates(time_1, rms_1, y_bins, x_bins, x_centers)
    t2_peaks, r2_peaks = get_peak_coordinates(time_2, rms_2, y_bins, x_bins, x_centers)
    
    m1, b1 = np.polyfit(t1_peaks, r1_peaks, 1)
    m2, b2 = np.polyfit(t2_peaks, r2_peaks, 1)
    
    print(f"Reference Line (Dataset 1): RMS = {m1:.5f}*t + {b1:.5f}")
    print(f"Target Line (Dataset 2):    RMS = {m2:.5f}*t + {b2:.5f}")

    # --- Step B: Rewrite Target File with Rescaled RMS ---
    print(f"\n--- Step 2: Rescaling '{target_file}' and writing to '{output_file}' ---")
    
    with h5py.File(target_file, 'r') as src, h5py.File(output_file, 'w') as dst:
        
        # Copy all top-level objects except 'dataset'
        for key in src.keys():
            if key != 'dataset':
                src.copy(key, dst)

        src_dataset = src['dataset']
        dst_dataset = dst.create_group('dataset')
        
        total_events = len(src_dataset.keys())
        
        for idx, event_key in enumerate(src_dataset.keys()):
            src_event = src_dataset[event_key]
            data = src_event[()]  # Load structured array
            
            for plane in planes:
                # Time is index 1 of 'pos', RMS is index 1 of 'x'
                node_times = data[f'{plane}/pos'][:, 1]
                node_rms = data[f'{plane}/x'][:, 1]
                
                # Calculate target and current peaks for these specific node times
                target_peak_at_t = (m1 * node_times) + b1
                current_peak_at_t = (m2 * node_times) + b2
                
                # Safeguard against division by zero or near-zero
                # If current_peak_at_t is too close to zero, cap it at a small epsilon
                current_peak_at_t = np.where(np.abs(current_peak_at_t) < 1e-5, 1e-5, current_peak_at_t)
                
                scale_factors = target_peak_at_t / current_peak_at_t
                
                # Apply the scaling to the RMS column
                data[f'{plane}/x'][:, 1] = node_rms * scale_factors

            # Save modified event dataset to the new file
            dst_dataset.create_dataset(
                event_key,
                data=data,
                dtype=src_event.dtype,
                compression=src_event.compression,
                shuffle=src_event.shuffle,
                fletcher32=src_event.fletcher32
            )
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_events} events...")
                
    print("\nDone! Rescaled dataset saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rescale HDF5 RMS values based on a reference dataset's linear fit.")
    
    # Add -r, -t, and -o as short flags alongside the long flags
    parser.add_argument("-r", "--reference_file", help="Path to the reference HDF5 file (Dataset 1)")
    parser.add_argument("-t", "--target_file", help="Path to the target HDF5 file to be modified (Dataset 2)")
    parser.add_argument("-o", "--output_file", help="Path to write the new, rescaled HDF5 file")

    args = parser.parse_args()

    rescale_and_save_hdf5(args.reference_file, args.target_file, args.output_file)



# python /home/hnayak/nugraph_project/nugraph/scripts/rescale_rms.py -r /scratch/7DayLifetime/cerati/numiallwr2.half.gnn.h5 -t /scratch/7DayLifetime/cerati/concat-final-makeup.tiny.gnn.h5 -o
    # /scratch/7DayLifetime/hnayak/nugraph_data/noisy_data/rms_scaled_noise/concat-final-makeup.tiny.gnn.rescaled.h5