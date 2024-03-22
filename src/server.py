from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image 
from utils import image_to_byte_array, string_to_scaling_algorithm
from scaler import scale_image_batch
from fastapi.middleware.cors import CORSMiddleware

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
    responses = {
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response
)
def main(content: UploadFile, algorithm: str = 'bicubic', factor: float = 2):
    img = Image.open(content.file)
    try:
        scaling_algorithm = string_to_scaling_algorithm(algorithm.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    scaled_image = scale_image_batch(scaling_algorithm, img, [factor])
    return Response(content=image_to_byte_array(scaled_image), media_type="image/png")