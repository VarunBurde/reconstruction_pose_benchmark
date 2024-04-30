import numpy as np
import os
import argparse
import cv2
import json
import math
from tqdm import tqdm
import random
import multiprocessing
from scipy.spatial.transform import Rotation as R
import shutil

parser = argparse.ArgumentParser(
                    prog='Converts the data from nerf format to bop format',
                    description='Converts the dataset from nerf format to bop format')

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


def main(dataset_dir):
    object_name = os.path.basename(dataset_dir)
    object_id = YCB_data[object_name]
    mask_dir = os.path.join(dataset_dir, 'mask_mesh')
    transform_json = os.path.join(dataset_dir, 'optimise_pose_transforms_all.json')
    output_dir = os.path.join(dataset_dir, 'bop_format')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}

    with open(transform_json, 'r') as f:
        transforms = json.load(f)

    K,w,h = load_intrinsics(transforms)
    real_scale = transforms["real_world_scale"]

    tqdm.write(f'Converting {object_name} to BOP format')
    progress = tqdm(total=len(transforms['frames']))

    for frame in transforms['frames']:
        file_path = frame['file_path']
        file_name = os.path.basename(file_path)
        scene_id = file_name.split('.')[0]
        scene_id = scene_id.split('_')[1]
        scene_id = int(scene_id)

        scene_camera[scene_id] = {}
        scene_gt[scene_id] = []
        scene_gt_info[scene_id] = []

        transform_matrix = np.array(frame['transform_matrix'])

        # convert opengl to opencv
        transform_matrix = np.matmul(transform_matrix, flip_mat)

        # to transform from opengl to opencv
        r = R.from_euler('zyx', [-90,0,-90], degrees=True)
        transform_matrix[:3, :3] = np.matmul(transform_matrix[:3, :3], np.linalg.inv(r.as_matrix()))

        # scale scene to real world scale
        transform_matrix[:3, 3] *= real_scale

        # convert scene to w2c
        transform_matrix = np.linalg.inv(transform_matrix)

        cam_R_w2c = transform_matrix[:3, :3]
        cam_t_w2c = transform_matrix[:3, 3] * 1000

        # scene camera json
        scene_camera[scene_id]["cam_K"] = K.tolist()
        scene_camera[scene_id]["depth_scale"] = 1.0
        scene_camera[scene_id]["cam_R_w2c"] = cam_R_w2c.tolist()
        scene_camera[scene_id]["cam_t_w2c"] = cam_t_w2c.tolist()

        # scene_gt_info json
        mask_location = os.path.join(mask_dir, file_name )
        mask = cv2.imread(mask_location, cv2.IMREAD_GRAYSCALE)

        # find bounding box
        mask = mask > 0
        y, x = np.where(mask)
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        bbox = [min_x, min_y, max_x, max_y]
        # convert to int
        bbox = [int(i) for i in bbox]


        scene_gt_info[scene_id].append({
            "bbox_obj": bbox,
            "bbox_visib": bbox,
            "px_count_all": int(mask.size),
            "px_count_valid": int(mask.size),
            "px_count_visib": int(mask.size),
            "visib_fract": 1.0
        })

        scene_gt[scene_id].append({
            "cam_R_m2c": cam_R_w2c.flatten().tolist(),
            "cam_t_m2c": cam_t_w2c.tolist(),
            "obj_id": object_id
        })

        progress.update(1)

    progress.close()

    # save scene camera json
    scene_camera_file = os.path.join(output_dir, 'scene_camera.json')
    with open(scene_camera_file, 'w') as f:
        json.dump(scene_camera, f, indent=4)

    # save scene_gt_info json
    scene_gt_info_file = os.path.join(output_dir, 'scene_gt_info.json')
    with open(scene_gt_info_file, 'w') as f:
        json.dump(scene_gt_info, f, indent=4)

    # save scene_gt json
    scene_gt_file = os.path.join(output_dir, 'scene_gt.json')
    with open(scene_gt_file, 'w') as f:
        json.dump(scene_gt, f, indent=4)

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_base = args.dataset_dir

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))

    for dataset in os.listdir(dataset_base):
        dataset_dir = os.path.join(dataset_base, dataset)
        process.apply_async(main, args=(dataset_dir,))
    process.close()
    process.join()

    print("Done")
