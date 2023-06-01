import os
import glob
from PIL import Image


def rename_images(directory: str) -> None:
    """Renames all .JPG images in a directory to a range of numbers and changes their extension to .jpg.
    
    Args:
    - directory: The path to the directory containing the images.
    """
    # Get a list of all .JPG files in the directory
    image_files = glob.glob(os.path.join(directory, "*.png"))
    # sort images by _number.jpg
    # image_files.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0].split("_")[1]))
    image_files.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))
    print(f"Found {len(image_files)} images in {directory}")

    # Iterate over the image files and rename them
    for idx, image_file in enumerate(image_files, start=1):
        im = Image.open(image_file)

        rgb_im = im.convert('RGB')
        rgb_im.save(f"{directory}/{idx}.jpg")

        print(f"Renamed {image_file} to {directory}/{idx}.jpg")

# Replace 'path_to_directory' with the actual path to the directory containing the images
path_to_directory = "src/data/fountain/images"
rename_images(path_to_directory)