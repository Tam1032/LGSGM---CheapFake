import joblib
import os
DATA_DIR = './Data'
subset = 'train'
size = 12000
images_data_train = joblib.load(f"{DATA_DIR}/{subset}_{size}/cheapfake_{subset}_lowered_images_data_{size}.joblib")
error_files = []
for key, value in images_data_train.items():
    file_joblib = key[:-4] + ".joblib"
    num_bbox = len(value['bbox'])
    path_joblib = os.path.join(f"{DATA_DIR}/{subset}_10000/PENET/VisualObjectFeatures_b5", file_joblib)
    check_bbox = len(joblib.load(path_joblib))
    if num_bbox != check_bbox:
        error_files.append(key)
print(error_files)