import numpy as np
import os
import argparse
import cv2
import json
import math
from tqdm import tqdm
import random
import multiprocessing
import collections
from scipy.spatial.transform import Rotation as R
import shutil

parser = argparse.ArgumentParser(
                    prog='Converts the data from nerf format to Colmap format',
                    description='Converts the dataset from nerf format to colmap format')

parser.add_argument('--dataset_dir', type=str, required=True,
                    help='Path to the dataset directory')

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

YCB_data = {
    1: "01_master_chef_can", 2: "02_cracker_box", 3: "03_sugar_box", 4: "04_tomato_soup_can", 5: "05_mustard_bottle",
    6: "06_tuna_fish_can", 7: "07_pudding_box", 8: "08_gelatin_box", 9: "09_potted_meat_can", 10: "10_banana",
    11: "11_pitcher_base", 12: "12_bleach_cleanser", 13: "13_bowl", 14: "14_mug", 15: "15_power_drill",
    16: "16_wood_block", 17: "17_scissors", 18: "18_large_marker", 19: "19_large_clamp", 20: "20_extra_large_clamp",
    21: "21_foam_brick"
}

# reverse the dictionary
YCB_data = {v: k for k, v in YCB_data.items()}
YCB_data = {k: v for k, v in sorted(YCB_data.items(), key=lambda item: item[1])}


def load_intrinsics(transoform_json):
    fl_x = transoform_json['fl_x']
    fl_y = transoform_json['fl_y']
    k1 = transoform_json['k1']
    k2 = transoform_json['k2']
    k3 = transoform_json['k3']
    p1 = transoform_json['p1']
    p2 = transoform_json['p2']
    cx = transoform_json['cx']
    cy = transoform_json['cy']
    w = transoform_json['w']
    h = transoform_json['h']

    K = np.array([[fl_x, 0, cx],
                    [0, fl_y, cy],
                    [0, 0, 1]])
    return K, w, h


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = "# Camera list with one line of data per camera:\n" + \
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
             "# Number of cameras: {}\n".format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_image_text(transforms, path):
    real_scale = transforms["real_world_scale"]
    images = {}
    for frame in transforms['frames']:
        file_path = frame['file_path']
        file_name = os.path.basename(file_path)
        img_id = file_name.split('.')[0]
        img_id = img_id.split('_')[1]
        img_id = int(img_id)

        transform_matrix = np.array(frame['transform_matrix'])

        # convert opengl to opencv
        transform_matrix = np.matmul(transform_matrix, flip_mat)

        # scale scene to real world scale
        transform_matrix[:3, 3] *= real_scale

        # convert scene to w2c
        transform_matrix = np.linalg.inv(transform_matrix)

        Rot = transform_matrix[:3, :3]
        T = transform_matrix[:3, 3] * 1000

        # to transform from opengl to opencv
        r = R.from_euler('zyx', [-90,0,-90], degrees=True)
        Rot = np.matmul(Rot, np.linalg.inv(r.as_matrix()))

        images[img_id] = BaseImage(id=img_id, camera_id=0, qvec=rotmat2qvec(Rot), tvec=T, name=file_name, xys=[0], point3D_ids=[])

    write_images_text(images, os.path.join(path, "images" + ".txt"))


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items())) / len(images)
    HEADER = "# Image list with two lines of data per image:\n" + \
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n" + \
             "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_cam_file(transforms, path):

    fl_x = transforms['fl_x']
    fl_y = transforms['fl_y']
    cx = transforms['cx']
    cy = transforms['cy']
    k1 = transforms['k1']
    k2 = transforms['k2']
    k3 = transforms['k3']
    p1 = transforms['p1']
    p2 = transforms['p2']
    height = transforms['h']
    width = transforms['w']


    cameras = {
        '0': Camera(id=0, model="OPENCV", height=height, width=width,
                    params=[fl_x, fl_y, cx, cy, k1, k2, p1, p2])}
    write_cameras_text(cameras, os.path.join(path, "cameras" + ".txt"))

def write_empty_points3d_file(path):
    path = os.path.join(path, "points3D.txt")
    with open(path, "w") as fid:
        fid.write("\n")

def main(dataset_dir):
    # object_name = os.path.basename(dataset_dir)
    # object_id = YCB_data[object_name]
    mask_dir = os.path.join(dataset_dir, 'mask_mesh')
    transform_json = os.path.join(dataset_dir, 'nerf', 'transforms_all.json')
    output_dir = os.path.join(dataset_dir, 'colmap_format')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    with open(transform_json, 'r') as f:
        transforms = json.load(f)

    write_cam_file(transforms, output_dir)
    write_image_text(transforms, output_dir)
    write_empty_points3d_file(output_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_base = args.dataset_dir
    # main(dataset_base)

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))

    for dataset in os.listdir(dataset_base):
        print(f"Processing {dataset}")
        dataset_dir = os.path.join(dataset_base, dataset)
        process.apply_async(main, args=(dataset_dir,))
    process.close()
    process.join()

    print("Done")
