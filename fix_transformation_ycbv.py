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

parser = argparse.ArgumentParser(
                    prog='Converts the data from nerf format to Colmap format',
                    description='Converts the dataset from nerf format to colmap format')

parser.add_argument('--dataset_dir', type=str, required=True,
                    help='Path to the dataset directory')

parser.add_argument('--transformation_dir', type=str, required=True,
                    help='Path to the transformation directory')

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

def main(dataset_dir, transformation_dir):
    mask_dir = os.path.join(dataset_dir, 'mask_mesh')
    transform_json = os.path.join(dataset_dir, 'transforms_all.json')
    with open(transform_json, 'r') as f:
        transforms = json.load(f)

    transformation_json = os.path.join(transformation_dir, 'ngp_weight', 'scale.json')
    with open(transformation_json, 'r') as f:
        scale = json.load(f)

    real_scale = transforms['real_world_scale']

    for frame in transforms['frames']:

        transform_matrix = np.array(frame['transform_matrix'])

        # convert opengl to opencv
        transform_matrix = np.matmul(transform_matrix, flip_mat)

        # scale scene to real world scale
        transform_matrix[:3, 3] *= real_scale

        # scale scene to mm
        transform_matrix[:3, 3] *= 1000

        # convert scene to w2c
        transform_matrix = np.linalg.inv(transform_matrix)

        ycbv_transfomation = np.array(scale["transformation"])
        ycbv_rotation = ycbv_transfomation[:3, :3]

        # apply the ycbv rotation
        transform_matrix[:3, :3] = np.matmul(ycbv_rotation, transform_matrix[:3, :3])

        # convert w2c to c2w
        transform_matrix = np.linalg.inv(transform_matrix)

        # opencv to opengl
        transform_matrix = np.matmul(transform_matrix, flip_mat)

        frame['transform_matrix'] = transform_matrix.tolist()

    with open(os.path.join(dataset_dir, 'transform_new.json'), 'w') as f:
        json.dump(transforms, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_base = args.dataset_dir
    transformation_dir = args.transformation_dir
    main(dataset_base, transformation_dir)
