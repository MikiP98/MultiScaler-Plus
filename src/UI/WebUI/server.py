# coding=utf-8
import os
# import saving.encoder

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from scaling.scaler_manager import scale_image_batch
from utils import image_to_byte_array, pngify  # , string_to_algorithm


string_to_algorithm = {}  # temporal error shutupper
cli_algorithms = set()  # temporal error shutupper


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
    # print(content)
    img = Image.open(content.file)

    try:
        scaling_algorithm = string_to_algorithm(algorithm.lower())
        if not scaling_algorithm:
            raise ValueError(f"\"{algorithm}\" does not exist!")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # print(f"Algorithm: {algorithm}")
    if scaling_algorithm in cli_algorithms:
        config_plus = {
            'sharpness': 0.5,
            'relative_input_path_of_images': ['./web_temp_input/image.png'],
            'relative_output_path_of_images': ['./web_temp_output/image.png']
        }
        image_bytes = image_to_byte_array(img)

        # check if 'web_temp_input' and 'web_temp_output' directories exist, if not create them
        if not os.path.exists('./web_temp_input'):
            os.makedirs('./web_temp_input')
        if not os.path.exists('./web_temp_output'):
            os.makedirs('./web_temp_output')

        with open('./web_temp_input/image.png', 'wb') as f:
            f.write(image_bytes)
    else:
        config_plus = None

    scaled_image_list = scale_image_batch(scaling_algorithm, [pngify(img)], [factor], config_plus=config_plus)

    if not scaled_image_list or len(scaled_image_list) == 0:
        scaled_image = Image.open("./web_temp_output/image.png")
    else:
        scaled_image = scaled_image_list.pop()['images'][0][0]

    return Response(content=image_to_byte_array(scaled_image), media_type="image/png")
