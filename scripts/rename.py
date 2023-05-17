import os

directory = "/Users/ziadh/Desktop/college/gp/src/images/snow-man"  # Specify the directory containing the images
extension = ".jpg"  # Specify the file extension of the images

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory) if f.endswith(extension)]

# Sort the image files alphabetically
image_files.sort()

# Rename the image files
for i, filename in enumerate(image_files, start=1):
    new_filename = os.path.join(directory, f"{i}{extension}")
    os.rename(os.path.join(directory, filename), new_filename)
