import os
import numpy as np
import glob

# Your input folder based on your VS Code explorer
input_dir = "dataset_v3_50k/test"
# We will create a new folder for the CSVs to keep things clean
output_dir = "dataset_v3_50k_csv/test"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all .npy files in the target folder
npy_files = glob.glob(os.path.join(input_dir, "*.npy"))

print(f"Found {len(npy_files)} files. Starting conversion...")

for filepath in npy_files:
    # Extract the filename without the extension (e.g., "heatmap_00000")
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_path = os.path.join(output_dir, f"{base_name}.csv")
    
    try:
        # Load the numpy array
        data = np.load(filepath)
        
        # Save as CSV. fmt='%g' keeps the numbers readable.
        np.savetxt(output_path, data, delimiter=",", fmt="%g")
        print(f"Converted: {base_name}.npy -> {base_name}.csv")
        
    except Exception as e:
        print(f"Error converting {base_name}.npy: {e}")

print(f"Done! All CSVs are saved in: {output_dir}")