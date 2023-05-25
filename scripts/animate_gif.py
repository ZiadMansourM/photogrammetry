import os
import subprocess

image_directory = r"E:\Project\photogrammetry\src\data\cottage\output\temp\testing_feature_match"
output_path = r"E:\Project\photogrammetry\src\data\cottage\output\temp\testing_feature_match\output.mp4"  # Update with the desired output path
framerate = 120

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(".jpg")]
image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# Create a temporary file with a list of input images
with open("input_files.txt", "w") as f:
    for image_file in image_files:
        f.write(f"file '{os.path.join(image_directory, image_file)}'\n")

# Run FFmpeg command to create a video
command = f'ffmpeg -y -f concat -safe 0 -i "input_files.txt" -framerate {framerate} -c:v libx264 -pix_fmt yuv420p "{output_path}"'
subprocess.run(command, shell=True, check=True)

# Remove temporary file
os.remove("input_files.txt")
