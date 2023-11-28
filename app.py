from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
from model.melodygenerator import MelodyGenerator

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_EXTENSIONS = {'mid'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

mg = MelodyGenerator(model_path="model/model.h5", mapping_path="model/mapping.json")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        seed = mg.encode_audio(file_path)

        if not seed:
            return render_template('index.html', error='Invalid MIDI file')

        melody = mg.generate_melody(seed, 500, 64, 0.3)
        output_filename = f"{filename}_melody.mid"
        output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
        print(melody)
        mg.save_melody(melody, output_path)

        return send_file(output_path, as_attachment=True)
    else:
        return render_template('index.html', error='Invalid file format')

if __name__ == '__main__':
    app.run(debug=True)
