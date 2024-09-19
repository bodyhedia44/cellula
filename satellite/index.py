from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import io
from PIL import Image
import rasterio

app = Flask(__name__)

model = load_model('my_model.h5')

def preprocess_image(image_path):
    with rasterio.open(image_path) as dataset:
        img = dataset.read()
        img = np.moveaxis(img, 0, -1) 
    img = np.expand_dims(img, axis=0)
    
    return img

def postprocess_image(output):

    output = output[0] 

    output = (output > 0.5).astype(np.uint8) * 255 

    print(f"Processed output shape: {output.shape}") 

    np_img = np.squeeze(output, axis=2)

    output_image = Image.fromarray(np_img)

    return output_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            file_path = 'temp.tif'
            file.save(file_path)

            preprocessed_img = preprocess_image(file_path)

            output = model.predict(preprocessed_img)

            segmented_img = postprocess_image(output)

            img_io = io.BytesIO()
            segmented_img.save(img_io, 'PNG')
            img_io.seek(0)

            return send_file(img_io, mimetype='image/png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
