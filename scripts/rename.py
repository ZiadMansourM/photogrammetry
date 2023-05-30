import os
import glob

def rename_images(directory: str) -> None:
    """Renames all .JPG images in a directory to a range of numbers and changes their extension to .jpg.
    
    Args:
    - directory: The path to the directory containing the images.
    """
    # Get a list of all .JPG files in the directory
    image_files = glob.glob(os.path.join(directory, "*.jpg.mask.png"))
    # sort images by _number.jpg
    image_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[1]))
    print(f"Found {len(image_files)} images in {directory}")

    # Iterate over the image files and rename them
    for idx, image_file in enumerate(image_files, start=1):
        # Create the new file name with the updated extension
        new_file_name = f"{idx}.jpg"
        new_file_path = os.path.join(directory, new_file_name)

        # # Rename the file
        os.rename(image_file, new_file_path)

        print(f"Renamed {image_file} to {new_file_path}")

# Replace 'path_to_directory' with the actual path to the directory containing the images
path_to_directory = "src/data/cottage/masks"
rename_images(path_to_directory)