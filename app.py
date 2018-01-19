import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2
import numpy as np
app = Flask(__name__)

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif','jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            img_name = secure_filename(img_file.filename)
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            #print(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
            return render_template('index.html', img_name=img_name)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
    else:
        return redirect(url_for('index'))

@app.route('/augumentation', methods = ['POST'])
def augumentation():
    print(url_for('static', filename=request.form['image']))
    im = cv2.imread("." + url_for('static', filename=request.form['image']))
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_edge = canny(im_gray)
    plt.imshow(im_edge, cmap="gray")
    plt.axis("off")
    img_arg_url= "." + url_for('static', filename=request.form['image']).rsplit('.', 1)[0] + "_edge." + request.form['image'].rsplit('.', 1)[1]
    print(img_arg_url)
    plt.savefig(img_arg_url)

    return render_template('augumentation.html', img_arg_url=img_arg_url)


if __name__ == '__main__':
    app.debug = True
    app.run()
