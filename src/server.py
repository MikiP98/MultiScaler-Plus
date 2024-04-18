# coding=utf-8
# import json

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from scaler import scale_image_batch
from utils import image_to_byte_array, string_to_algorithm

app = FastAPI()
# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post(
    "/scale",
    responses={
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response
)
def main(content: UploadFile, algorithm: str = 'bicubic', factor: float = 2):
    print(content)
    img = Image.open(content.file)

    try:
        scaling_algorithm = string_to_algorithm(algorithm.lower())
        if not scaling_algorithm:
            raise ValueError(f"\"{algorithm}\" does not exist!")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    config_plus = {

    }
    scaled_image = scale_image_batch(scaling_algorithm, [[img]], [factor], config_plus=config_plus).pop().pop()
    if not scaled_image:
        scaled_image = Image.open("./web_temp_output/image.png")

    return Response(content=image_to_byte_array(scaled_image), media_type="image/png")
