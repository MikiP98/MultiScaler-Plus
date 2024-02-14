from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image 
from Utils import image_to_byte_array, string_to_scaling_algorithm
from scaler import scale_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get(
    "/scale",
    responses = {
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response
)
async def main(content: UploadFile, algorithm: str = 'bicubic', factor: float = 2):
    img = Image.open(content.file)
    try:
        scaling_algorithm = string_to_scaling_algorithm(algorithm.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    scaled_image = scale_image(scaling_algorithm, img, factor)
    return Response(content=image_to_byte_array(scaled_image), media_type="image/png")