import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import json
from io import BytesIO
import base64
import scripts.detect_all_type_files as detector
import webbrowser

app = Flask(__name__)
CORS(app)
json_file_path = ''
html_opened = False
# Function to process the uploaded image
def process_image(file_bytes,filename):
    global json_file_path
    #image = cv2.imread(file_path)
    if file_bytes is not None:
        detect = detector.Detection()
        # Replace this with your desired image processing logic
        res = detect.get_detection(filename,file_bytes)
        # print(res)
        json_fields, proccessed = res
        #img = field_detector.processed_image()
        #json_fields = field_detector.get_annotations()
        file_path = os.path.join("annotations", filename[:-4]+'.json')
        response_data = {"filename": os.path.basename(file_path), "data": json.loads(json_fields)}

        # Write the JSON data to the file
        json_file_path = file_path
        with open(file_path, "w") as json_file:
            json.dump(response_data, json_file, indent=4)

        imgs_data=[]
        for image in proccessed:            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            imgs_data.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        return {"status": "success", "message": "Image processed successfully", "processed_json": response_data, "processed_image": imgs_data}
    else:
        return {"status": "error", "message": "Invalid image file"}

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})

    if image_file:
        filename = image_file.filename
        #file_path = os.path.join("uploads", filename)
        image_file_str = image_file.read()
        file_bytes = np.frombuffer(image_file_str, np.uint8)
        # convert numpy array to image
        #img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        #print(type(img))
        #image_file.save(file_path)
        #cv2.imwrite(file_path, img)
        result = process_image(file_bytes,image_file.filename)
        #os.remove(file_path)  # Remove the uploaded file after processing
        return jsonify(result)
    
@app.route('/get_json_file', methods=['GET'])
def get_json_file():
    global json_file_path
    #print(json_file_path)
    if os.path.exists(json_file_path):
        # Read the JSON data from the file
        with open(json_file_path, 'r') as file:
            json_data = file.read()

        # Set the appropriate headers for the response
        response = make_response(json_data)
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = f'attachment; filename={os.path.basename(json_file_path)}'
        #print(os.path.basename(json_file_path))
        return response
    else:
        return jsonify({"status": "error", "message": "JSON file not found"})




if __name__ == '__main__':
    #os.makedirs("uploads", exist_ok=True)
    os.makedirs("annotations", exist_ok=True)
    if not html_opened:
        htmlfilename = 'file:///'+os.getcwd()+'/' + 'template/index.html'
        webbrowser.open_new_tab(htmlfilename)
        html_opened = True
    app.run(debug=False)
    
