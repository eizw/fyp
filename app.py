from flask import Flask, request, render_template, redirect, url_for
import os
import pydicom
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50

# Flask Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads/"
STATIC_FOLDER = "static/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load Models
deep_lesion_model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
deep_lesion_model.eval()

lesion_classifier = resnet50(weights="DEFAULT")
lesion_classifier.fc = torch.nn.Linear(2048, 4)  
lesion_classifier.eval()

transform_detect = transforms.Compose([transforms.ToPILImage(), transforms.Resize((512, 512)), transforms.ToTensor()])
transform_classify = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

LESION_CLASSES = ["Tumor", "Cyst", "Hemorrhage", "Inflammation"]

def process_dicom(filepath):
    """Load and process a DICOM file"""
    ds = pydicom.dcmread(filepath, force=True)
    image = ds.pixel_array
    if len(image.shape) == 3:
        image = image[image.shape[0] // 2]
    return ds, image

def detect_lesions(image):
    """Run lesion detection (Faster R-CNN)"""
    image_rgb = np.stack([image] * 3, axis=-1) if len(image.shape) == 2 else image
    image_tensor = transform_detect(image_rgb).unsqueeze(0)

    with torch.no_grad():
        detections = deep_lesion_model(image_tensor)

    boxes = []
    for box, score in zip(detections[0]['boxes'], detections[0]['scores']):
        if score > 0.75:
            x1, y1, x2, y2 = map(int, box.tolist())
            boxes.append((x1, y1, x2, y2, score.item()))

    return boxes

def classify_lesion(image, box):
    """Classify a detected lesion using ResNet50"""
    x1, y1, x2, y2 = box
    lesion_crop = image[y1:y2, x1:x2]

    if lesion_crop.size == 0:
        return "Unknown"

    lesion_crop = Image.fromarray(lesion_crop.astype(np.uint8))
    lesion_tensor = transform_classify(lesion_crop).unsqueeze(0)

    with torch.no_grad():
        output = lesion_classifier(lesion_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return LESION_CLASSES[predicted_class]

def save_dicom_image(image, boxes):
    """Save processed DICOM image with bounding boxes"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    ax.axis("off")

    for (x1, y1, x2, y2, conf) in boxes:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"))

    save_path = os.path.join(STATIC_FOLDER, "processed_image.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            ds, image = process_dicom(filepath)
            boxes = detect_lesions(image)
            processed_image_path = save_dicom_image(image, boxes)

            return render_template("index.html", patient=ds.PatientName, modality=ds.Modality, study_date=ds.StudyDate, 
                                   lesion_count=len(boxes), image_path=processed_image_path)
    
    # No results should be shown until a file is uploaded
    return render_template("index.html", lesion_count=None)

if __name__ == "__main__":
    app.run(debug=True)