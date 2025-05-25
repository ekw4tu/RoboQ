from flask import Flask, request, redirect, url_for, render_template, jsonify
import os
import fitz  # PyMuPDF
from script import generate_questions
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
THUMBNAIL_FOLDER = 'static/thumbnails'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THUMBNAIL_FOLDER'] = THUMBNAIL_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert PDF to images using PyMuPDF
        try:
            pdf_document = fitz.open(file_path)
            thumbnail_paths = []
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                thumbnail_filename = f"{filename.rsplit('.', 1)[0]}_thumb_{page_num}.png"
                thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumbnail_filename)
                pix.save(thumbnail_path)
                thumbnail_paths.append(thumbnail_filename)

            return render_template('thumbnails.html', thumbnails=thumbnail_paths, filename=filename)
        except Exception as e:
            return str(e)
    else:
        return 'Invalid file type'

@app.route('/thumbnails/<filename>')
def view_thumbnails(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return 'File not found', 404

    try:
        pdf_document = fitz.open(file_path)
        thumbnail_paths = []
        for page_num in range(len(pdf_document)):
            thumbnail_filename = f"{filename.rsplit('.', 1)[0]}_thumb_{page_num}.png"
            thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumbnail_filename)
            if not os.path.exists(thumbnail_path):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                pix.save(thumbnail_path)
            thumbnail_paths.append(thumbnail_filename)

        return render_template('thumbnails.html', thumbnails=thumbnail_paths, filename=filename)
    except Exception as e:
        return str(e)

@app.route('/page/<int:page_number>/<filename>')
def get_page(page_number, filename):
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("file_path = ", file_path)
        questions = generate_questions(file_path, [page_number])
        print("questions = ", questions)
        return render_template('quiz.html', questions=questions, pagenum=page_number+1)

    return f'Missing the filename for {page_number}'

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
