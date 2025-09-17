from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('brain_tumour_mri.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['file']
        if img_file:
            # Ensure folder exists just before saving (extra safety)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(path)

            img = image.load_img(path, target_size=(300, 300))
            x = image.img_to_array(img)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)

            prediction = model.predict(x)
            probability = prediction[0][0] * 100  # percentage
            result = "Tumor Detected" if probability > 50 else "No Tumor Detected"
            prob_text = f"{probability:.2f}% confidence"

            return render_template('index.html', result=result, prob_text=prob_text, image_path=path)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
