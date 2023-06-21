import os
import shutil

data_dir = "/content/drive/MyDrive/Plant_Classification/assets/images/PlantVillage-Dataset/raw/color"

# get list of all directories in data_dir
directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

print(f"Directories: {directories}")

for directory in directories:
    # get base plant type (everything before the ___)
    base_plant_type = directory.split("___")[0]

    # create consolidated directory if it doesn't exist
    consolidated_dir = os.path.join(data_dir, f"{base_plant_type}_Consolidated")
    if not os.path.exists(consolidated_dir):
        print(f"Creating directory: {consolidated_dir}")
        os.mkdir(consolidated_dir)

    # move all images in directory to consolidated directory
    for filename in os.listdir(os.path.join(data_dir, directory)):
        if filename.endswith(".jpg"):  # adjust this if your images are not .jpg
            source = os.path.join(data_dir, directory, filename)
            destination = os.path.join(consolidated_dir, filename)
            print(f"Moving file from {source} to {destination}")
            shutil.move(source, destination)

    # if the directory ends with "_Consolidated" (and is not the main consolidated directory), delete it
    if directory.endswith("_Consolidated") and directory != f"{base_plant_type}_Consolidated":
        shutil.rmtree(os.path.join(data_dir, directory))
