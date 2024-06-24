from flask import Flask, request
import io
from PIL import Image


app = Flask(__name__)


@app.route('/', methods=['POST'])
def receive_image():
    file = request.files['file']  # get the file
    image = Image.open(io.BytesIO(file.read()))  # convert bytes to image
    # process your image here
    return 'Image received and processed', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
