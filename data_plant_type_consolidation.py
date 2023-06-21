# data_plant_type_consolidation.py
import os
import shutil
from config import *

# List of plants
plants = PLANT_CLASSES

# Base directory
base_dir = DATA_DIR_OLD

# New directory to store organized data
new_dir = DATA_DIR

# Create directories for each plant
for plant in plants:
    dir_path = os.path.join(new_dir, plant)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")

# Print the contents of the base directory
print(f"Contents of base directory {base_dir}:")
print(os.listdir(base_dir))

# Move files to the corresponding directory
for dir_name in os.listdir(base_dir):
    print(f'Working on {dir_name}..')
    for plant in plants:
        # If the plant name is in the directory name
        if plant in dir_name:
            print(f'Noticed it is a {plant} dir..')
            # Full path to the directory
            dir_path = os.path.join(base_dir, dir_name)
            # If it is a directory
            if os.path.isdir(dir_path):
                # Move all files in the directory to the corresponding plant directory
                for file_name in os.listdir(dir_path):
                    old_file_path = os.path.join(dir_path, file_name)
                    new_file_path = os.path.join(new_dir, plant, file_name)
                    shutil.move(old_file_path, new_file_path)
                    print(f"Moved file from {old_file_path} to {new_file_path}")
                # Remove the directory
                os.rmdir(dir_path)
                print(f"Deleted directory: {dir_path}")
                break  # no need to check the other plants
