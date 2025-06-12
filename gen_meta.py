import json
import SimpleITK as sitk

image_path = "data/Lipo-001_MR/1/NIFTI/image.nii.gz"

# read image
itk_image = sitk.ReadImage(image_path) 

# get metadata dict
header = {k: itk_image.GetMetaData(k) for k in itk_image.GetMetaDataKeys()}

# save dict in 'header.json'
with open("nnUNet_raw/Dataset420_Lipoma/dataset.json", "w") as outfile:
    json.dump(header, outfile, indent=4)