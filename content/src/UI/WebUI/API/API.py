# coding=utf-8
from DTOs import *

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import Response
# from PIL import Image
# from scaling.scaler_manager import scale_image_batch


app = FastAPI()
# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


type JSON = dict[str, None | bool | int | float | str | list[JSON] | dict[str, JSON]]


@app.get("/test")
def read_root() -> JSON:
    return {"Hello": "World"}


@app.get("/handshake")
def read_root() -> ServerConfigDTO:
    return {
        "api_version": "1.0.0",
        "endpoints": [
            {
                "endpoint": "process/scale",
                "display_name": "scaling",
                "processes": [
                    {
                        "name": "PIL_LANCHOS",
                        "display_name": "Lanchos *(PIL)*",
                        "description": "High quality classic scaling algorithm. "
                                       "In theory best for both down and up scaling among classic algorithms.",
                        "tags": ["Classic"],
                        "non_cost_per_output_pixel": 0,
                        "premium_cost_per_output_pixel": 0,
                        "premium_only": False
                    }
                ],
                "premium_only": False
            }
        ],
        "make_premium_a_tag": True,
        "require_account_for_non_zero_costs_processing": False
    }
