from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from typing import List
import scanmate 
import threading

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.post("/{dataset}/images")
async def upload_images(dataset: str, images: List[UploadFile] = File(...)):
    if os.path.exists(f"data/{dataset}"):
        raise HTTPException(status_code=409, detail=f"Error: {dataset} already exists.")

    os.makedirs(f"data/{dataset}")
    os.makedirs(f"data/{dataset}/images")
    os.makedirs(f"data/{dataset}/masks")
    os.makedirs(f"data/{dataset}/bak")
    os.makedirs(f"data/{dataset}/logs")
    os.makedirs(f"data/{dataset}/output")
    os.makedirs(f"data/{dataset}/output/feature-match")
    os.makedirs(f"data/{dataset}/output/image-match")
    os.makedirs(f"data/{dataset}/output/triangulate")
    os.makedirs(f"data/{dataset}/output/sift")
    # Save the uploaded images to the specified directory
    for img in images:
        with open(f"data/{dataset}/images/{img.filename}", "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
    # Start a new thread to process the images without blocking the request
    processing_thread = threading.Thread(target=scanmate.run, args=(dataset,))
    processing_thread.start()
    return {"message": f"Images uploaded successfully under {dataset}"}

@app.delete("/{dataset}")
async def delete_images(dataset: str):
    directory = f"data/{dataset}"

    if not os.path.exists(directory):
        raise HTTPException(status_code=404, detail=f"Error: {dataset} not found")

    # Delete the dataset directory
    shutil.rmtree(directory)

    return {"message": f"{dataset} deleted successfully"}

@app.get("/{dataset}/object")
async def get_object(dataset: str):
    # Check that the specified directory exists
    output_files: list[str] = [
        f"data/{dataset}/output/triangulate/points_cloud.stl",
        f"data/{dataset}/output/triangulate/core_points.stl",
        f"data/{dataset}/output/triangulate/camera_proj.stl",
        f"data/{dataset}/output/triangulate/mesh.stl"
    ]
    if not os.path.exists(f"data/{dataset}"):
        raise HTTPException(status_code=404, detail=f"Error: {dataset} not found")

    # Check that the .obj file exists in the specified directory
    if any(
        not os.path.exists(file_path)
        for file_path in output_files
    ):
        raise HTTPException(
            status_code=202,
            detail=f"Processing not done yet for {dataset}.",
            headers={"Retry-After": "300"}
        )

    return FileResponse(
        output_files[3],
        media_type="application/octet-stream",
        status_code=200
    )