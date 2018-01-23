import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from models.yolo_detection import yolo_detect
from models.mask_rcnn_detection import mask_rcnn_detect
import cv2
import numpy as np
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif','jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

############### 初めのページ#############################
@app.route('/')
def index():
    return render_template('index.html')

############### argmentationのページ#############################
@app.route('/augmentation_initial')
def augmentation_initial():
    return render_template('augmentation.html')

@app.route('/augmentation_send', methods=['GET', 'POST'])
def augmentation_send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            img_name = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            return render_template('augmentation.html', img_name=img_name)
        else:
            return render_template('augmentation.html', warning="warning")
    else:
        return redirect(url_for('augmentation'))

@app.route('/augmentation_augmentation', methods = ['POST'])
def augmentation_augmentation():
    #print(url_for('static', filename= "uploads/" + request.form['image']))
    im = cv2.imread("." + url_for('static', filename= "uploads/" + request.form['image']))

    if request.form['button_name'] == "Gray":
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_gray." + request.form['image'].rsplit('.', 1)[1]
        cv2.imwrite(img_arg_url,im_gray)

    if request.form['button_name'] == "Sobel_y":
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_sobely." + request.form['image'].rsplit('.', 1)[1]
        im_sobely = cv2.Sobel(im_gray, cv2.CV_32F, 0, 1, ksize=3)
        cv2.imwrite(img_arg_url,im_sobely)

    if request.form['button_name'] == "Sobel_x":
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_sobelx." + request.form['image'].rsplit('.', 1)[1]
        im_sobelx = cv2.Sobel(im_gray, cv2.CV_32F, 1, 0, ksize=3)
        cv2.imwrite(img_arg_url,im_sobelx)

    if request.form['button_name'] == "Canny":
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_canny = cv2.Canny(im_gray,100,200)
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_canny." + request.form['image'].rsplit('.', 1)[1]
        cv2.imwrite(img_arg_url,im_canny)

    if request.form['button_name'] == "Rotation":
        angle = 20
        M = cv2.getRotationMatrix2D((int(im.shape[0]/2),int(im.shape[1]/2)),angle,1)
        im_rotation = cv2.warpAffine(im,M,(im.shape[1], im.shape[0]))
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_rotation." + request.form['image'].rsplit('.', 1)[1]
        cv2.imwrite(img_arg_url,im_rotation)

    if request.form['button_name'] == "Flip":
        im_flip = np.fliplr(im)
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_flip." + request.form['image'].rsplit('.', 1)[1]
        cv2.imwrite(img_arg_url,im_flip)

    if request.form['button_name'] == "Noise":
        s_vs_p = 0.5
        amount = 0.004
        im_noise = im.copy()
        num_pepper = np.ceil(amount* im_noise.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in im_noise.shape]
        im_noise[coords[:-1]] = (0,0,0)
        img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_noise." + request.form['image'].rsplit('.', 1)[1]
        cv2.imwrite(img_arg_url,im_noise)

    return render_template('augmentation.html', img_arg_url=img_arg_url, img_name = request.form['image'])


############### YOLOのページ#############################
@app.route('/yolo_initial')
def yolo_initial():
    return render_template('yolo.html')

@app.route('/yolo_send', methods=['GET', 'POST'])
def yolo_send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            img_name = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            return render_template('yolo.html', img_name=img_name)
        else:
            return render_template('yolo.html', warning="warning")
    else:
        return redirect(url_for('yolo'))

@app.route('/yolo_detection', methods = ['POST'])
def yolo_detectio():
    image_url = url_for('static', filename= "uploads/" + request.form['image'])
    yolo_url = yolo_detect(image_url)
    return render_template('yolo.html', yolo_url=yolo_url)

############### Mask_RCNNのページ#############################
@app.route('/mask_rcnn_initial')
def mask_rcnn_initial():
    return render_template('mask_rcnn.html')

@app.route('/mask_rcnn_send', methods=['GET', 'POST'])
def mask_rcnn__send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            img_name = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            return render_template('mask_rcnn.html', img_name=img_name)
        else:
            return render_template('mask_rcnn.html', warning="warning")
    else:
        return redirect(url_for('mask_rcnn'))

@app.route('/mask_rcnn_detection', methods = ['POST'])
def mask_rcnn_detectio():
    image_url = url_for('static', filename= "uploads/" + request.form['image'])
    mask_rcnn_url = mask_rcnn_detect(image_url)
    return render_template('mask_rcnn.html', mask_rcnn_url=mask_rcnn_url)


if __name__ == '__main__':
    app.debug = True
    app.run()
