import os
import shutil
import glob
import math

def copy_file(src, dst):
    shutil.copy(src, dst)

def prune_data():
    output_path = "nnUNet_raw/Dataset420_Lipoma/"
    image_path = output_path + "imagesTr/"
    label_path = output_path + "labelsTr/"

    for f in glob.glob(image_path + "*"):
        os.remove(f)
    for f in glob.glob(label_path + "*"):
        os.remove(f)


def process_datafile(path):
    data_path = path + "1/NIFTI/"
    image_file = data_path + "image.nii.gz"
    label_file = data_path + "segmentation.nii.gz"

    output_path = "nnUNet_raw/Dataset420_Lipoma/"
    image_path = output_path + "imagesTr/"
    label_path = output_path + "labelsTr/"

    fileext = ".nii.gz"
    filename = "FYP" + path.split('/')[-2][5:8]

    #image
    copy_file(image_file, image_path + filename + "_0000" + fileext)
    #label
    try:
        copy_file(label_file, label_path + filename + fileext)
    except FileNotFoundError:
        copy_file(data_path + "segmentation_Lipoma.nii.gz", label_path + filename + fileext)
    



def main():
    input_path = "data/"

    n = len([entry for entry in os.listdir(input_path) \
        if os.path.isdir(os.path.join(input_path, entry))])

    train = math.floor(n * 0.8)
    test = math.ceil(n * 0.2)

    prune_data()
    
    for files in glob.glob(input_path + '*'):
        process_datafile(files + "/")

if __name__ == '__main__':
    main()