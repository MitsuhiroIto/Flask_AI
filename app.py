import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import cv2
import numpy as np
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif','jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/augmentation_initial')
def augmentation_initial():
    return render_template('augmentation.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
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


@app.route('/augmentation', methods = ['POST'])
def augmentation():
    print(url_for('static', filename= "uploads/" + request.form['image']))
    im = cv2.imread("." + url_for('static', filename= "uploads/" + request.form['image']))
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_edge = cv2.Canny(im_gray,100,200)
    img_arg_url= "." + url_for('static', filename="uploads/" + request.form['image']).rsplit('.', 1)[0] + "_edge." + request.form['image'].rsplit('.', 1)[1]
    cv2.imwrite(img_arg_url,im_edge)
    return render_template('augmentation.html', img_arg_url=img_arg_url)


if __name__ == '__main__':
    app.debug = True
    app.run()
