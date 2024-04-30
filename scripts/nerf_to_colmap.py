import argparse
import collections
import json
import os
import multiprocessing

import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil

from scripts.common import (
    YCB_data,
    flip_mat,
    load_intrinsics,
)
from scripts.read_write_model import (
    BaseImage,
    Camera,
    rotmat2qvec,
    write_cameras_text,
    write_images_text
)


def gen_image_text(transforms, path):
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


def gen_cam_file(transforms, path):
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


def gen_empty_points3d_file(path):
    path = os.path.join(path, "points3D.txt")
    with open(path, "w") as fid:
        fid.write("\n")


def main(dataset_dir):
    mask_dir = os.path.join(dataset_dir, 'mask_mesh')
    transform_json = os.path.join(dataset_dir, 'nerf', 'transforms_all.json')
    output_dir = os.path.join(dataset_dir, 'colmap_format')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    with open(transform_json, 'r') as f:
        transforms = json.load(f)

    gen_cam_file(transforms, output_dir)
    gen_image_text(transforms, output_dir)
    gen_empty_points3d_file(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='Converts the data from nerf format to Colmap format',
                        description='Converts the dataset from nerf format to colmap format')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the dataset directory')
    args = parser.parse_args()
    dataset_base = args.dataset_dir

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))
    for dataset in os.listdir(dataset_base):
        print(f"Processing {dataset}")
        dataset_dir = os.path.join(dataset_base, dataset)
        process.apply_async(main, args=(dataset_dir,))
    process.close()
    process.join()

    print("Done")
