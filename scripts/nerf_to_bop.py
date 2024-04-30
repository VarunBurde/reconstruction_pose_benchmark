import argparse
import json
import multiprocessing
import os
import shutil
from tqdm import tqdm

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.common import (
    YCB_data,
    flip_mat,
    load_intrinsics,
)


def main(dataset_dir):
    object_name = os.path.basename(dataset_dir)
    object_id = YCB_data[object_name]
    mask_dir = os.path.join(dataset_dir, 'mask_mesh')
    transform_json = os.path.join(dataset_dir, 'nerf', 'transforms_all.json')
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
        transform_matrix = np.matmul(transform_matrix, flip_mat) # OK
        transform_matrix = np.linalg.inv(transform_matrix)

        # to transform from opengl to opencv
        r = R.from_euler('zyx', [-90,0,-90], degrees=True)
        transform_matrix[:3, :3] = np.matmul(transform_matrix[:3, :3], np.linalg.inv(r.as_matrix()))

        # scale scene to real world scale
        transform_matrix[:3, 3] *= real_scale # cTw

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
        box_tol = 5 # to account for mask inacuraccy
        min_x = max(0, np.min(x) - box_tol)
        max_x = min(mask.shape[1], np.max(x) + box_tol)
        min_y = max(0, np.min(y) - box_tol)
        max_y = min(mask.shape[0], np.max(y) + box_tol)
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
    parser = argparse.ArgumentParser(
                        prog='Converts the data from nerf format to bop format',
                        description='Converts the dataset from nerf format to bop format')
    
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the dataset directory')
    args = parser.parse_args()
    dataset_base = args.dataset_dir
    #main(dataset_base)

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))

    for dataset in os.listdir(dataset_base):
        dataset_dir = os.path.join(dataset_base, dataset)
        process.apply_async(main, args=(dataset_dir,))
    process.close()
    process.join()

    print("Done")
