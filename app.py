from flask import Flask, render_template, request, redirect, url_for
import cv2 as cv
import os
import geocoder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['DETECTED_FOLDER'] = 'static/detected/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)

# Load YOLO model
class_name = []
with open(r'utils/obj.names', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

net1 = cv.dnn.readNet(r'utils/yolov4_tiny.weights', r'utils/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Route for homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            # Process the image with YOLO
            detected_path = detect_potholes(image_path, file.filename)

            # Render results
            return render_template('index.html', uploaded_image=file.filename, detected_image=detected_path)
    return render_template('index.html')



# Function to perform pothole detection
def detect_potholes(image_path, filename):
    # Load image
    image = cv.imread(image_path)
    height, width, _ = image.shape

    # Detection parameters
    Conf_threshold = 0.5
    NMS_threshold = 0.4

    # Perform detection
    classes, scores, boxes = model1.detect(image, Conf_threshold, NMS_threshold)

    # Draw detections on the image
    g = geocoder.ip('me')
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        rec_area = w * h
        frame_area = width * height

        # Adjust severity calculation
        severity = "Low"
        if rec_area / frame_area > 0.1:
            severity = "High"
        elif rec_area / frame_area > 0.02:
            severity = "Medium"

        if score >= 0.2:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(image, f"pothole: {severity}", (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    # Save the detected image
    detected_path = os.path.join(app.config['DETECTED_FOLDER'], filename)
    cv.imwrite(detected_path, image)
    return filename

if __name__ == '__main__':
    app.run(debug=True)
