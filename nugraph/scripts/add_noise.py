import os, sys
sys.path.append('/home/hnayak/.conda/envs/NugraphBase/lib/python3.10/site-packages')
import h5py
import numpy as np

def modify_hdf5(input_file, output_file, noise_type):
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dst:

        # Copy all top‑level objects except 'dataset'
        for key in src.keys():
            if key != 'dataset':
                src.copy(key, dst)

        src_dataset = src['dataset']
        dst_dataset = dst.create_group('dataset')
        
        noise_percentage = 0.4  # 40% noise (tried 0.1, 0.2, 0.4)
        planes = ['u', 'v', 'y']
        
        # ==========================================
        # PASS 1: Calculate Global Robust Spread 
        # ==========================================
        global_sigmas = {plane: [0.0, 0.0] for plane in planes}
        
        if noise_type == 'gaussian':
            print("Pass 1: Calculating global 68% spread across all events...")
            
            # Dictionary to accumulate data: { plane: [ [integral_arrays], [rms_arrays] ] }
            all_data = {plane: [[], []] for plane in planes}
            
            for event_key in src_dataset.keys():
                data = src_dataset[event_key][()]
                for plane in planes:
                    all_data[plane][0].append(data[f'{plane}/x'][:, 0])
                    all_data[plane][1].append(data[f'{plane}/x'][:, 1])

            # Calculate robust sigma for each feature (Integral=0, RMS=1)
            for plane in planes:
                for i in (0, 1):
                    combined = np.concatenate(all_data[plane][i])
                    # Robust Sigma: Half of the 16% to 84% range
                    sigma = (np.quantile(combined, 0.84) - np.quantile(combined, 0.16)) / 2.0
                    global_sigmas[plane][i] = sigma
            
            print("Global Sigmas Calculated (Plane: [Integral, RMS]):", global_sigmas)

        # ==========================================
        # PASS 2: Apply Noise and Write Data
        # ==========================================
        print(f"Pass 2: Applying '{noise_type}' noise and writing to output file...")
        
        # Multipliers for 'fixed' noise type (Plane -> [Integral_mult, RMS_mult])
        fixed_mults = {'u': [0.8, 1.5], 'v': [1.2, 0.9], 'y': [1.1, 1.2]}
        
        for event_key in src_dataset.keys():
            src_event = src_dataset[event_key]
            data = src_event[()]  # Load structured array
            
            for plane in planes:
                feature_data = data[f'{plane}/x']
                num_nodes = feature_data.shape[0]
                
                # Apply Gaussian Noise
                if noise_type == 'gaussian':
                    for i in (0, 1):
                        noise = np.random.normal(0, global_sigmas[plane][i], size=num_nodes) * noise_percentage
                        feature_data[:, i] += noise
                        
                # Apply Fixed Multipliers
                elif noise_type == 'fixed':
                    for i in (0, 1):
                        feature_data[:, i] *= fixed_mults[plane][i]

            # Save modified dataset to the new file
            dst_dataset.create_dataset(
                event_key,
                data=data,
                dtype=src_event.dtype,
                compression=src_event.compression,
                shuffle=src_event.shuffle,
                fletcher32=src_event.fletcher32
            )
            
        print("Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modify HDF5 dataset values.")
    parser.add_argument("input_file", help="Path to the input HDF5 file")
    parser.add_argument("output_file", help="Path to write modified HDF5 file")
    parser.add_argument("--noise", choices=['gaussian', 'fixed'], default='gaussian', help="Type of noise to apply")

    args = parser.parse_args()

    modify_hdf5(args.input_file, args.output_file, args.noise)


## python /home/hnayak/nugraph_project/nugraph/scripts/add_noise.py /scratch/7DayLifetime/cerati/concat-final-makeup.tiny.gnn.h5 /home/hnayak/nugraph_project/noisy_data/40_percent_noise/concat-final-makeup.tiny.gnn.noisy_gaussian.h5 --noise gaussian