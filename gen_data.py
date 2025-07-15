import os
import shutil
import glob
import math

# output_path = "nnUNet_raw/Dataset420_Lipoma/"
this_directory = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(this_directory, f'data/')
os.makedirs(output_path, exist_ok=True)

def copy_file(src, dst):
    shutil.copy(src, dst)

def prune_data():
    image_path = output_path + "imagesTr/"
    label_path = output_path + "labelsTr/"
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    for f in glob.glob(image_path + "*"):
        os.remove(f)
    for f in glob.glob(label_path + "*"):
        os.remove(f)


def process_datafile(path):
    # data_path = path + "1/NIFTI/"
    patient_id = path.split('/')[-2]

    data_path = path + ""
    image_file = data_path + "image.nii.gz"
    if patient_id == "Lipo-073":
      label_file = data_path + "segmentation_WDLPS.nii.gz"
    else:
      label_file = data_path + "segmentation.nii.gz"

    image_path = output_path + "imagesTr/"
    label_path = output_path + "labelsTr/"

    fileext = ".nii.gz"
    filename = "FYP" + path.split('/')[-2][5:8]


    #image
    # copy_file(image_file, image_path + filename + "_0000" + fileext)
    print(image_file, image_path)
    copy_file(image_file, image_path + patient_id + fileext)
    #label
    # try:
    #     copy_file(label_file, label_path + filename + fileext)
    # except FileNotFoundError:
    #     copy_file(data_path + "segmentation_Lipoma.nii.gz", label_path + filename + fileext)
    



def main():
    input_path = "lipoma_data/"

    n = len([entry for entry in os.listdir(input_path) \
        if os.path.isdir(os.path.join(input_path, entry))])

    train = math.floor(n * 0.8)
    test = math.ceil(n * 0.2)

    prune_data()
    
    for files in glob.glob(input_path + '*'):
        process_datafile(files + "/")

if __name__ == '__main__':
    main()