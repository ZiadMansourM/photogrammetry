import os
import time
import requests

# Function to upload images and wait for the object file to be generated
def upload_images_and_wait(image_name_set):
    API_ENDPOINT = "https://photogrammetry.sreboy.com"

    # Define the image directory and set name
    IMAGE_DIR = "images/{image_name_set}"

    # Define the upload URL
    UPLOAD_URL = f"{API_ENDPOINT}/images/"

    # Define the object retrieval URL
    OBJECT_URL = f"{API_ENDPOINT}/object/"

    # Define the polling interval in seconds
    POLL_INTERVAL = 20 * 60  # 20 minutes

    # Function to check if the object file exists
    def object_file_exists():
        response = requests.get(OBJECT_URL)
        return response.status_code == 200

    # Create a list of files to upload
    files = []
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(".jpg"):
            filepath = os.path.join(IMAGE_DIR, filename)
            files.append(("images", (filename, open(filepath, "rb"), "image/jpeg")))

    # Upload the images to the server
    response = requests.post(UPLOAD_URL, files=files, data={"name": image_name_set})

    # Check if the upload was successful
    if response.status_code != 200:
        print(f"Error uploading images: {response.text}")
        return

    # Wait for the object file to be generated
    while not object_file_exists():
        print("Object file not found yet. Sleeping...")
        time.sleep(POLL_INTERVAL)

    # Retrieve the object file
    response = requests.get(OBJECT_URL)

    # Save the object file to disk
    obj_path = f"output/{image_name_set}/test-api-object.obj"
    with open(obj_path, "wb") as f:
        f.write(response.content)

    print(f"Object file saved to {obj_path}")

if __name__ == "__main__":
    # Call the upload_images_and_wait function
    upload_images_and_wait("snow-man")