from create_model import predict 
#Import servert rest
from flask import Flask, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
FILE_EXTENSIONS = ["mp3","wav","ogg", "flac"]
UPLOAD_FOLDER = 'home'


# Inicializacion de  clases  de Neurak Network and MFCC (13 variables)


def file_extension(filename):
    return os.path.splitext(filename)[1][1:]


def allowed_file(filename):
    return '.' in filename and file_extension(filename).lower() in FILE_EXTENSIONS


@app.route("/predict", methods=['POST'])
def file():
    if "file" not in request.files:
        return "file is required", 400

    file = request.files["file"]

    if file.filename == "":
        return "file not selected", 400

    if file and allowed_file(file.filename):
        filename = "file1.{}".format(file_extension(file.filename))
        #os.stat(os.path.join(os.path.abspath(UPLOAD_FOLDER)))
        if not os.path.exists(os.path.join(os.path.abspath(UPLOAD_FOLDER))):
            os.mkdir(os.path.join(os.path.abspath(UPLOAD_FOLDER)))
        file.save(os.path.join(os.path.abspath(UPLOAD_FOLDER), filename))
        #RandoForest="RF" Redesneuronales="RN"
        predictions,probabilidad, evaluar= predict().predecir(os.path.join(UPLOAD_FOLDER, filename),"RN")
        print(evaluar)
        return predictions+";"+str(probabilidad)+";"+str(evaluar), 200
    else:
        return "file type is not allowed", 400


if __name__ == "__main__":
    app.run(port=5000)



