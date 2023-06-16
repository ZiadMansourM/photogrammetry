import cv2 as OpenCV
import os

from .image import Image


class FeatureMatches:
    def __init__(self, image_one: Image, image_two: Image, matches: list[OpenCV.DMatch]):
        self.image_one: Image = image_one
        self.image_two: Image = image_two
        self.matches: list[OpenCV.DMatch] = matches

    def draw_matches(self, output_filename: str) -> None:
        combined_image = OpenCV.hconcat([
            self.image_one.rgb_image,
            self.image_two.rgb_image
        ])
        for match in self.matches:
            x1, y1 = self.image_one.keypoints[match.queryIdx].pt
            x2, y2 = self.image_two.keypoints[match.trainIdx].pt
            # Draw a line connecting the matched keypoints
            OpenCV.line(
                combined_image, 
                (int(x1), int(y1)), 
                (int(x2) + self.image_one.rgb_image.shape[1], int(y2)), 
                (0, 255, 0), 
                1
            )
        OpenCV.imwrite(output_filename, combined_image)
        
    def animate_matches(self, output_filename: str) -> None:
        import subprocess
        for match in self.matches:
            combined_image = OpenCV.hconcat([
                self.image_one.rgb_image,
                self.image_two.rgb_image
            ])
            x1, y1 = self.image_one.keypoints[match.queryIdx].pt
            x2, y2 = self.image_two.keypoints[match.trainIdx].pt
            # Write match.queryIdx at the top left corner
            OpenCV.putText(
                combined_image,
                f"{match.queryIdx}",
                (50, 150),  # position: 10 pixels from left, 20 pixels from top
                OpenCV.FONT_HERSHEY_SIMPLEX,  # font
                5,  # font scale
                (0, 255, 0),  # font color (green)
                5,  # thickness
                OpenCV.LINE_AA  # line type
            )
            # Write match.trainIdx at the top right corner
            image_two_width = self.image_one.rgb_image.shape[1]
            OpenCV.putText(
                combined_image,
                f"{match.trainIdx}",
                (image_two_width + 50, 150),  # position: 10 pixels from right, 20 pixels from top
                OpenCV.FONT_HERSHEY_SIMPLEX,  # font
                5,  # font scale
                (0, 255, 0),  # font color (green)
                5,  # thickness
                OpenCV.LINE_AA  # line type
            )
            # Draw a line connecting the matched keypoints
            OpenCV.line(
                combined_image, 
                (int(x1), int(y1)), 
                (int(x2) + self.image_one.rgb_image.shape[1], int(y2)), 
                (0, 255, 0), 
                1
            )
            OpenCV.imwrite(
                f"{output_filename}/{match.queryIdx}_{match.trainIdx}.jpg",
                combined_image,
            )
        framerate = 120
        # Get a list of image files in the directory
        image_files = [f for f in os.listdir(output_filename) if f.endswith(".jpg")]
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        # Create a temporary file with a list of input images
        with open("input_files.txt", "w") as f:
            for image_file in image_files:
                f.write(f"file '{os.path.join(output_filename, image_file)}'\n")
        # Run FFmpeg command to create a video
        command = f'ffmpeg -y -f concat -safe 0 -i "input_files.txt" -framerate {framerate} -c:v libx264 -pix_fmt yuv420p "{output_filename}/output.mp4"'
        subprocess.run(command, shell=True, check=True)
        # Remove temporary file
        os.remove("input_files.txt")

    def __repr__(self):
        return f"FeatureMatches({self.image_one}, {self.image_two} ---> {len(self.matches)})"

    def __getstate__(self):
        state = self.__dict__.copy()
        state['matches'] = [
            {'queryIdx': m.queryIdx, 'trainIdx': m.trainIdx, 'distance': m.distance} for m in self.matches
        ]
        return state
    
    def __setstate__(self, state):
        state['matches'] = [
            OpenCV.DMatch(match['queryIdx'], match['trainIdx'], match['distance']) for match in state['matches']
        ]
        self.__dict__ = state