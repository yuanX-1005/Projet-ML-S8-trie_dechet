from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('model_vgg16_v2.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def preprocess_image(image):
    try:
        image = image.resize((224, 224))
        image = np.array(image)
        if image.shape == (224, 224, 3):  # Ensure image has 3 channels
            image = image / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        return image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            try:
                image = Image.open(file).convert('RGB')  # Ensure image is in RGB format
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_label = label_encoder.inverse_transform(predicted_class)[0]
                
                # Save the uploaded image to display it on the result page
                file_path = os.path.join('static/uploads', file.filename)
                image.save(file_path)
                
                return render_template('index.html', prediction=predicted_label, image_url=file_path)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the upload directory exists
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
