from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from typing import List
import concurrent.futures

import davinci

app = FastAPI()

def run_davinci(name: str):
    davinci.davinci_run(name)

@app.post("/images/")
async def upload_images(name: str, images: List[UploadFile] = File(...)):
    """
    Upload a set of images and save them under a directory with the given name in the /images directory.

    :param name: The name of the directory to create and save the images to
    :param images: A list of uploaded images to save
    :return: A success message if the images were uploaded successfully
    """
    # Check that the images directory exists
    if not os.path.exists("images/"):
        os.makedirs("images/")

    # Check if a directory with the same name already exists
    dir_path = f"images/{name}"
    if os.path.exists(dir_path):
        raise HTTPException(status_code=409, detail=f"Directory {dir_path} already exists")

    # Create a directory with the specified name inside the images directory
    os.makedirs(dir_path)

    # Save the uploaded images to the specified directory
    for img in images:
        with open(f"{dir_path}/{img.filename}", "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)

    # Run the Davinci algorithm on the uploaded images in a background thread
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(run_davinci, name)

    return {"message": f"Images uploaded successfully under {dir_path}"}

@app.get("/object/")
async def get_object(name: str):
    """
    Retrieve the .obj file for a set of images with the given name from the /output directory.

    :param name: The name of the directory containing the images to retrieve the .obj file for
    :return: The .obj file for the specified set of images
    """
    # Check that the specified directory exists
    dir_path = f"output/{name}"
    if not os.path.exists(dir_path):
        raise HTTPException(status_code=404, detail=f"Directory {dir_path} not found")

    # Check that the .obj file exists in the specified directory
    obj_path = f"{dir_path}/object.obj"
    if not os.path.isfile(obj_path):
        raise HTTPException(status_code=404, detail=f"Object file not found in {dir_path}")

    return FileResponse(obj_path, media_type="application/octet-stream")