from flask import Flask, request, render_template, redirect, url_for
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from PIL import Image
import time

from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
import joblib
import WORC
import pandas as pd

# Load machine learning stuff
df = pd.read_hdf('estimator_all_0.hdf5')
feature_params = pd.read_hdf('features_predict_MRI_0_Lipo-115_0.hdf5')

# classification_model = df["Diagnosis"].iloc[0][-1].best_estimator_
classification_model = joblib.load('trained_worc_model.pkl')
extparams = feature_params.iloc[1]

# Flask Setup
app = Flask(__name__)

this_directory = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(this_directory, f'static/')
UPLOAD_FOLDER = os.path.join(this_directory, f'uploads')
SEGMENTATION_FOLDER = os.path.join(STATIC_FOLDER, f'segmentations/')
FEATURE_FOLDER = os.path.join(this_directory, f'Features/')

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["IMAGE_SERVER"] = 'http://localhost:8000/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ! Change to False on launch
session = True 

def process_dicom(filepath):
    """Load and process a DICOM file"""
    ds = pydicom.dcmread(filepath, force=True)
    image = ds.pixel_array
    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]
    return ds, image

def save_dicom_image(image):
    """Save processed DICOM image with bounding boxes"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    

    # Extracting features
    features = []
    extractor = featureextractor.RadiomicsFeatureExtractor(**extparams)
    segmentation_file = "segmentation.nii.gz"
    segmentation = os.path.join(STATIC_FOLDER, segmentation_file)

    if os.path.exists(image) and os.path.exists(segmentation):
        try:
            feature_vector = extractor.execute(image, segmentation)
            features.append(feature_vector)
        except Exception as e:
            print(f"Error with extracting features: {e}")
    else:
        print(f"Missing files for image or segmentation")

    # Classification
    X = pd.DataFrame(features)
    y_pred = classification_model.pred(X)
    
    save_path = os.path.join(STATIC_FOLDER, "processed_image.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def process_data(image_path, segmentation_path):
    extractor = featureextractor.RadiomicsFeatureExtractor(**extparams)

    features = []

    if os.path.exists(image_path) and os.path.exists(segmentation_path):
        try:
            feature_vector = extractor.execute(image_path, segmentation_path)
            # Remove non-feature metadata
            feature_vector = {k: v for k, v in feature_vector.items() if "diagnostics" not in k}
            features.append(feature_vector)
        except Exception as e:
            print(f"Error with feature extraction: {e}")
    else:
        print(f"Missing files for {image_path}")

    # feature_file = 'features_predict_MR_0_' + patient_name + '_0'
    # feature_path = os.path.join(app.config)
        
    X = pd.DataFrame(features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(features, X, segmentation_path, image_path, flush=True)

    return X_scaled

def predict_class(patient_id, image_path, model):
    segmentation_file = 'segmentation.nii.gz'
    segmentation_path = 'lipoma_data/' + patient_id + '/' + segmentation_file

    X = process_data(
        image_path=image_path,
        segmentation_path=segmentation_path,
    )

    y = model.predict(X)
    print(y)
    return segmentation_path, y



@app.route("/", methods=["GET", "POST"])
def home():
    global session
    global classification_model

    if not session:
        return redirect('/login')
    
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            timestamped_name = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], timestamped_name)
            

            fmt = filename.split('.')[1]
            print(fmt)
            if fmt.lower() == 'dcm':
                ds, image = process_dicom(filepath)
                save_dicom_image(image)
            else:
                file.save(filepath)

            patient_id=filename.split('.')[0]

            segmentation_path, classification = predict_class(
                patient_id=patient_id,
                image_path=filepath,
                model=classification_model,
            )

            if classification == 1:
                classification = "ALT/WDLPS"
            else:
                classification = "Lipoma"

            print(filepath)

            return render_template("index.html",
                                    image_server=app.config["IMAGE_SERVER"],
                                    image_file="uploads/" + timestamped_name,
                                    patient_id=patient_id,
                                    segmentation_file=segmentation_path,
                                    classification=classification,
                                   )
    
    # No results should be shown until a file is uploaded
    return render_template("index.html",
                           image_server=app.config["IMAGE_SERVER"],
                           patient_id='',
                           image_file='', 
                           segmentation_file='',
                           classification='',
                        )

@app.route("/login", methods=["GET", "POST"])
def login_page():
    global session
    if request.method == "POST":
        session = True
        return redirect(url_for('home'))
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)