from flask import Flask, render_template, Response, jsonify, request, send_file
from PIL import Image, ImageOps
import virtualMousePainting as virtualMouse
import time
import cv2
import numpy as np

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

canvas = np.zeros((480, 640, 3), np.uint8)

app = Flask(__name__)

#virtual mouse
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/virtual_mouse')
def virtual_mouse():
    return render_template('virtual_mouse.html')



#painting의 gen()  -> canvas , video 분리 x
@app.route('/video_feed')
def video_feed():
    return Response(virtualMouse.gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/index')
def index():
    return render_template('index.html')



#두번째꺼 -> 완성 / canvas, video 분리 o
@app.route('/canvas')
def canvas():
    return render_template('canvas.html')

@app.route('/webcam_feed')
def webcam_feed_route():
    return Response(virtualMouse.webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/canvas_feed')
def canvas_feed_route():
    return Response(virtualMouse.canvas_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_canvas', methods=['POST'])
def save_canvas_route():
    filename = virtualMouse.save_canvas()
    return jsonify({"message": "Canvas saved", "filename": filename})


if __name__ == '__main__':
    app.run(debug=True)