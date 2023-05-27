import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

import cv2
import numpy as np
from matplotlib import pyplot as plt
import radialProfile
import os
import base64
from flask import Flask, request, render_template, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_deepfake', methods=['POST'])
def detect_deepfake():
    epsilon = 1e-8
    file = request.files['file']

    # Save the uploaded file
    filename = 'uploaded_image.jpg'
    file.save(filename)

    img = cv2.imread(filename, 0)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

    # Calculate the 2D power spectrum
    psd2D = np.log(np.abs(fshift) ** 2)

    # Visualization
    fig = plt.figure(figsize=(20, 5))  # Increase width of the figure
    ax = fig.add_subplot(131, aspect='auto')  # Increase size of each subplot
    ax2 = fig.add_subplot(132, aspect='auto')  # Increase size of each subplot
    ax3 = fig.add_subplot(133, aspect='auto')  # Increase size of each subplot

    ax.set_title('Input Image', size=15)
    ax2.set_title('1D Power Spectrum', size=15)
    ax3.set_title('2D Power Spectrum', size=15)

    ax.axis('off')

    plt.xlabel('Spatial Frequency', fontsize=12)
    plt.ylabel('Power Spectrum', fontsize=12)

    ax.imshow(img_color, cmap='gray')
    ax2.plot(psd1D)
    ax3.imshow(psd2D, cmap='jet')

    # Save the visualization
    output_filename = 'output.png'
    fig.savefig(output_filename)
    plt.close(fig)

    # Prepare the response
    response = {
        'output_image': get_base64_encoded_image(output_filename)
    }

    # Remove the temporary files
    os.remove(filename)
    os.remove(output_filename)

    return jsonify(response)

def get_base64_encoded_image(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
    return encoded_image

if __name__ == '__main__':
    app.run()