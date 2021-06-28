import os

from flask import Flask, json, request
from werkzeug.utils import secure_filename

from "./digits_recognizer" import digits_recognizer

ALLOWED_EXTENSIONS = ['png', 'jpeg']
UPLOAD_FOLDER = './saved_files'

api = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_digit_from_image():
    digits_recognizer.test()

@api.route('/recognize', methods=['POST'])
#Called when route requested
def recognize():
    # check if the post request has the file part
    if 'file' not in request.files:
        print('No file part')
        return "No file part"
    
    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        print("Empty file")
        return "Empty file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        predict_digit_from_image(file)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return "Saved"
    # return json.dumps(companies)

if __name__ == '__main__':
    api.run()