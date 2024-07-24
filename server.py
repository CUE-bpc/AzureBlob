import json
import os
from flask import Flask, jsonify, request
import urllib.parse
from flask_cors import CORS
import base64
from azure.storage.blob import BlobServiceClient  # Python v12 SDK    --      pip install azure-storage-blob
import requests
import urllib.request
# Hier gibt's ein import problem, Azure checkt das Ding nicht bei startup
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray

app = Flask(__name__, static_folder='', static_url_path='')
cors = CORS(app)
# kleiner test

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/upload-ocr', methods=['POST'])
def upload_ocr():
    return 'To be implemented.'

# Endpoint to upload image to Azure Blob Storage
# Takes base64 string, converts to PNG, uploads to Azure Blob Storage and returns URL
# Call with POST request to /azure-upload with base64 string in the request body
@app.route('/azure-upload', methods=['POST'])
def azure_upload():
    # Configuration details (ideally these should be stored in environment variables or a config file)
    storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    connection_string = "DefaultEndpointsProtocol=https" + \
                        ";AccountName=" + storage_account_name + \
                        ";AccountKey=" + storage_account_key

    # Get base64 string from the request body (JSON format)
    data = request.get_json()
    if not data or 'base64_string' not in data:
        return jsonify({"error": "No base64_string field in request body"}), 400

    base_64_string_data = data['base64_string']

    if not base_64_string_data:
        return jsonify({"error": "Your base64_string is empty"}), 400

    # Remove the data URL prefix if it exists
    if base_64_string_data.startswith("data:image/png;base64,"):
        base_64_string_data = base_64_string_data[len("data:image/png;base64,"):]

    container_name = "images"
    filename = "zaehlerstand.png" # Name of the file to be uploaded

    # Debug: print the received base64 string
    print("Received base64 string:", base_64_string_data)

    # Azure Storage Blob endpoint
    azure_storage_endpoint = "https://" + storage_account_name + ".blob.core.windows.net/"

    # Convert Base64 string to bytes object (PNG format)
    png = base64.b64decode(base_64_string_data)

    # Create a new instance of BlobServiceClient to interact with the blob service
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

    # Upload the image to the specified container in Azure Blob Storage
    blob_client.upload_blob(png, blob_type="BlockBlob", overwrite=True)

    # Get the URL of the uploaded blob
    blob_url = azure_storage_endpoint + container_name + "/" + blob_client.get_blob_properties().name

    return blob_url


# TODO: Error detection (kein Zählerstand erkannt), Bild trotzdem hochladen etc.
# TODO: azure_upload und ocr_stand kombinieren in einer Funktion, s.o. upload_ocr
# Endpoint to mark Zählerstand in Zaehlerstand.png image and to receive recognized Zählerstand
# See also /azure-upload
@app.route('/ocr-stand', methods=['GET'])
def ocr_stand():

    custom_vision_imgurl = os.getenv('CUSTOM_VISION_IMGURL')
    custom_vision_prediction_key = os.getenv('CUSTOM_VISION_PREDICTION_KEY')

    # Define the URL of the image, see above
    image_url = 'https://voicebotimages.blob.core.windows.net/images/zaehlerstand.png'

    # Download the image from the URL
    with urllib.request.urlopen(image_url) as response:
        data = response.read()

    # decode the image file as a cv2 image, useful for later to display results
    img = cv2.imdecode(np.array(bytearray(data), dtype='uint8'), cv2.IMREAD_COLOR)

    custom_vision_headers = {
        'Content-Type': 'application/octet-stream',
        'Prediction-Key': custom_vision_prediction_key
    }

    custom_vision_resp = requests.post(url=custom_vision_imgurl,
                                       data=data,
                                       headers=custom_vision_headers).json()

    # inspect the top result, based on probability
    hit = pd.DataFrame(custom_vision_resp['predictions']).sort_values(by='probability', ascending=False).head(
        1).to_dict()

    # extract the bounding box for the detected number plate
    boundingbox = list(hit['boundingBox'].values())[0]
    l, t, w, h = (boundingbox['left'],
                  boundingbox['top'],
                  boundingbox['width'],
                  boundingbox['height'])

    # extract bounding box coordinates and dimensions are scaled using image dimensions
    polylines1 = np.multiply([[l, t], [l + w, t], [l + w, t + h], [l, t + h]],
                             [img.shape[1], img.shape[0]])

    # draw polylines based on bounding box results
    temp_img = cv2.polylines(img, np.int32([polylines1]),
                             isClosed=True, color=(255, 255, 0), thickness=5)

    # display the original image with the plate region
    plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))

    # crop the image to the bounding box of the plate region
    crop_x = polylines1[:, 0].astype('uint16')
    crop_y = polylines1[:, 1].astype('uint16')

    img_crop = img[np.min(crop_y):np.max(crop_y),
               np.min(crop_x):np.max(crop_x)]

    # display the detected plate region
    #plt.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

    img_crop_height = img_crop.shape[0]
    if img_crop_height < 50:
        pil_image = Image.fromarray(img_crop)
        img_crop_width = img_crop.shape[1]
        difference = 50 / img_crop_height
        resized_dimensions = (int(img_crop_width * difference), int(img_crop_height * difference))
        pil_image_resized = pil_image.resize(resized_dimensions)
        img_crop_resized = asarray(pil_image_resized)

        plt.imshow(cv2.cvtColor(img_crop_resized, cv2.COLOR_BGR2RGB))
    else:
        img_crop_resized = img_crop

    # display the original image with the plate region
    #plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    # display the plot
    #plt.show()

    computer_vision_imgurl = 'https://germanywestcentral.api.cognitive.microsoft.com/computervision/imageanalysis:analyze?api-version=2023-02-01-preview&features=read'

    crop_bytes = bytes(cv2.imencode('.png', img_crop_resized)[1])

    # make a call to the computer_vision_imgurl
    computer_vision_resp = requests.post(
        url=computer_vision_imgurl,
        data=crop_bytes,
        headers={
            'Ocp-Apim-Subscription-Key': custom_vision_prediction_key,
            'Content-Type': 'application/octet-stream'}).json()

    #print('Der Zählerstand ist {}'.format(computer_vision_resp['readResult']['content']))

    # --- Upload new image to Blob Storage ---

    # Configuration details (ideally these should be stored in environment variables or a config file)
    storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    connection_string = "DefaultEndpointsProtocol=https" + \
                        ";AccountName=" + storage_account_name + \
                        ";AccountKey=" + storage_account_key

    container_name = "images"
    filename = "zaehlerstand.png"  # Name of the file to be uploaded

    # Azure Storage Blob endpoint
    azure_storage_endpoint = "https://" + storage_account_name + ".blob.core.windows.net/"

    # Convert temp_img to png byte stream
    _, temp_img_png = cv2.imencode('.png', temp_img)
    png = temp_img_png.tobytes()

    # Create a new instance of BlobServiceClient to interact with the blob service
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)

    # Upload the image to the specified container in Azure Blob Storage
    blob_client.upload_blob(png, blob_type="BlockBlob", overwrite=True)

    # Get the URL of the uploaded blob
    blob_url = azure_storage_endpoint + container_name + "/" + blob_client.get_blob_properties().name

    # --- END ---

    # Return Zählerstand and its image

    result = {
        "stand": computer_vision_resp['readResult']['content'],
        "image_url": image_url
    }

    return json.dumps(result)

if __name__ == '__main__':
    app.run()
