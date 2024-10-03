from flask import Flask, render_template, jsonify, Response, request
import time
import io
import numpy as np
import cv2
import psutil
from ultralytics import YOLO
import sys

model = YOLO('yolov8m.pt')

app = Flask(__name__)

detected = []

def generate_frames():
    """Generate frames for the video feed."""
    global detected
    while True:
        source = "rtsp://raspi.local:8554/cam"

        results = model(source, classes=[2], stream=True)
        # results = model(source="0", classes=[2], stream=True)
        # results = model(source="0", stream=True)
        for result in results:
            annotated_frame = result.plot()
            detected = result.boxes.cls.numpy()
        
            if annotated_frame is not None:
                annotated_frame = cv2.resize(annotated_frame, (640, 480))
                _, jpeg = cv2.imencode('.jpg', annotated_frame)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Provide an amount of car detected stats."""
    stats = {
        'cars': len(detected)
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)