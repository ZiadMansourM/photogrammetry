import imageio
import glob
import cv2
import os

image_paths = glob.glob(os.path.join(r"C:\Users\yousf\OneDrive\Desktop\University\Graduation Project\Codes\photogrammetry\src\data\hammer\testing_feature_match", "*.jpg"))
image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
output_path = r'C:\Users\yousf\OneDrive\Desktop\University\Graduation Project\Codes\photogrammetry\src\data\hammer\testing_feature_match\output.gif'
new_size = (800, 800)
# print("\n".join([path for path in image_paths]))
with imageio.get_writer(output_path, mode='I', duration=10) as writer:
    for image_path in image_paths:
        image = imageio.imread(image_path)
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        writer.append_data(resized_image)
        print("Done appending image: ", image_path)
print("Done creating gif")
