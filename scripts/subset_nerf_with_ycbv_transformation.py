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


parser = argparse.ArgumentParser(
                    prog='Converts the dataset to scaled nerf format',
                    description='Converts the dataset to scaled nerf format with many subset')

parser.add_argument('--dataset_dir', type=str, required=True,
                    help='Path to the dataset directory')

parser.add_argument('--transform_location', type=str, required=True,
                    help='Path to the transformation file')


def closest_point_2_lines(oa, da, ob, db):
    # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def read_textfile(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def load_json_structure(config_file_path):
    json_file = open(config_file_path)
    camera = json.load(json_file)
    mtx = camera['camera_matrix']
    dist = camera['dist_coeff']
    width, height = camera['img_size']
    fl_x = mtx[0][0]
    fl_y = mtx[1][1]
    cx = mtx[0][2]
    cy = mtx[1][2]
    k1 = dist['k1']
    k2 = dist['k2']
    p1 = dist['p1']
    p2 = dist['p2']
    k3 = dist['k3']

    angle_x = math.atan(width / (fl_x * 2)) * 2
    angle_y = math.atan(height / (fl_y * 2)) * 2
    fovx = angle_x * 180 / math.pi
    fovy = angle_y * 180 / math.pi

    out = {
        "camera_model": "OPENCV",
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        "is_fisheye": False,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "scale": 1.0,
        "aabb_scale": 4,
        "frames": [],
    }
    return out

def main(dataset_dir, trasform_location):
    img_dir = os.path.join(dataset_dir, 'undistorted_images')
    pose_file_path = os.path.join(dataset_dir, 'camera_poses_real_undistorted_centered.json')
    camera_file_path = os.path.join(dataset_dir, 'camera_undistorted.json')

    transform_json = os.path.join(trasform_location, 'ngp_weight', 'scale.json')
    transform = json.load(open(transform_json))
    ycbv_transform = np.array(transform['transformation'])


    train_file_path_3 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                     'image_list_3.txt')
    train_file_path_5 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                     'image_list_5.txt')
    train_file_path_15 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                      'image_list_15.txt')
    train_file_path_20 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                      'image_list_20.txt')
    train_file_path_10 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                      'image_list_10.txt')
    train_file_path_25 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                      'image_list_25.txt')
    train_file_path_50 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                      'image_list_50.txt')
    train_file_path_75 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                      'image_list_75.txt')
    train_file_path_100 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                       'image_list_100.txt')
    train_file_path_150 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                       'image_list_150.txt')
    train_file_path_300 = os.path.join(dataset_dir, 'subsets_fibonacci',
                                       'image_list_300.txt')
    train_file_path_all = os.path.join(dataset_dir, 'subsets_fibonacci',
                                       'image_list_-1.txt')

    out_3 = load_json_structure(camera_file_path)
    out_5 = load_json_structure(camera_file_path)
    out_15 = load_json_structure(camera_file_path)
    out_20 = load_json_structure(camera_file_path)
    out_10 = load_json_structure(camera_file_path)
    out_25 = load_json_structure(camera_file_path)
    out_50 = load_json_structure(camera_file_path)
    out_75 = load_json_structure(camera_file_path)
    out_100 = load_json_structure(camera_file_path)
    out_150 = load_json_structure(camera_file_path)
    out_300 = load_json_structure(camera_file_path)
    out_all = load_json_structure(camera_file_path)

    train_list_3 = read_textfile(train_file_path_3)
    train_list_5 = read_textfile(train_file_path_5)
    train_list_15 = read_textfile(train_file_path_15)
    train_list_20 = read_textfile(train_file_path_20)
    train_list_10 = read_textfile(train_file_path_10)
    train_list_25 = read_textfile(train_file_path_25)
    train_list_50 = read_textfile(train_file_path_50)
    train_list_75 = read_textfile(train_file_path_75)
    train_list_100 = read_textfile(train_file_path_100)
    train_list_150 = read_textfile(train_file_path_150)
    train_list_300 = read_textfile(train_file_path_300)
    train_list_all = read_textfile(train_file_path_all)

    # use 0.2 of random sample of train images as test images
    test_list_3 = random.sample(train_list_3, int(len(read_textfile(train_file_path_3)) * 0.2))
    test_list_5 = random.sample(train_list_5, int(len(read_textfile(train_file_path_5)) * 0.2))
    test_list_15 = random.sample(train_list_15, int(len(read_textfile(train_file_path_15)) * 0.2))
    test_list_20 = random.sample(train_list_20, int(len(read_textfile(train_file_path_20)) * 0.2))
    test_list_10 = random.sample(train_list_10, int(len(read_textfile(train_file_path_10)) * 0.2))
    test_list_25 = random.sample(train_list_25, int(len(read_textfile(train_file_path_25)) * 0.2))
    test_list_50 = random.sample(train_list_50, int(len(read_textfile(train_file_path_50)) * 0.2))
    test_list_75 = random.sample(train_list_75, int(len(read_textfile(train_file_path_75)) * 0.2))
    test_list_100 = random.sample(train_list_100, int(len(read_textfile(train_file_path_100)) * 0.2))
    test_list_150 = random.sample(train_list_150, int(len(read_textfile(train_file_path_150)) * 0.2))
    test_list_300 = random.sample(train_list_300, int(len(read_textfile(train_file_path_300)) * 0.2))
    test_list_all = random.sample(train_list_all, int(len(read_textfile(train_file_path_all)) * 0.2))

    train_list_path_3 = []
    train_list_path_5 = []
    train_list_path_15 = []
    train_list_path_20 = []
    train_list_path_10 = []
    train_list_path_25 = []
    train_list_path_50 = []
    train_list_path_75 = []
    train_list_path_100 = []
    train_list_path_150 = []
    train_list_path_300 = []
    train_list_path_all = []

    test_list_path_3 = []
    test_list_path_5 = []
    test_list_path_15 = []
    test_list_path_20 = []
    test_list_path_10 = []
    test_list_path_25 = []
    test_list_path_50 = []
    test_list_path_75 = []
    test_list_path_100 = []
    test_list_path_150 = []
    test_list_path_300 = []
    test_list_path_all = []

    images_sub_path = os.path.split(img_dir)[1] + "/"

    for train_list_item in train_list_3:
        train_list_item = images_sub_path + train_list_item
        train_list_path_3.append(train_list_item)

    for train_list_item in train_list_5:
        train_list_item = images_sub_path + train_list_item
        train_list_path_5.append(train_list_item)

    for train_list_item in train_list_15:
        train_list_item = images_sub_path + train_list_item
        train_list_path_15.append(train_list_item)

    for train_list_item in train_list_20:
        train_list_item = images_sub_path + train_list_item
        train_list_path_20.append(train_list_item)

    for train_list_item in train_list_10:
        train_list_item = images_sub_path + train_list_item
        train_list_path_10.append(train_list_item)

    for train_list_item in train_list_25:
        train_list_item = images_sub_path + train_list_item
        train_list_path_25.append(train_list_item)

    for train_list_item in train_list_50:
        train_list_item = images_sub_path + train_list_item
        train_list_path_50.append(train_list_item)

    for train_list_item in train_list_75:
        train_list_item = images_sub_path + train_list_item
        train_list_path_75.append(train_list_item)

    for train_list_item in train_list_100:
        train_list_item = images_sub_path + train_list_item
        train_list_path_100.append(train_list_item)

    for train_list_item in train_list_150:
        train_list_item = images_sub_path + train_list_item
        train_list_path_150.append(train_list_item)

    for train_list_item in train_list_300:
        train_list_item = images_sub_path + train_list_item
        train_list_path_300.append(train_list_item)

    for train_list_item in train_list_all:
        train_list_item = images_sub_path + train_list_item
        train_list_path_all.append(train_list_item)

    for test_list_item in test_list_3:
        test_list_item = images_sub_path + test_list_item
        test_list_path_3.append(test_list_item)

    for test_list_item in test_list_5:
        test_list_item = images_sub_path + test_list_item
        test_list_path_5.append(test_list_item)

    for test_list_item in test_list_15:
        test_list_item = images_sub_path + test_list_item
        test_list_path_15.append(test_list_item)

    for test_list_item in test_list_20:
        test_list_item = images_sub_path + test_list_item
        test_list_path_20.append(test_list_item)

    for test_list_item in test_list_10:
        test_list_item = images_sub_path + test_list_item
        test_list_path_10.append(test_list_item)

    for test_list_item in test_list_25:
        test_list_item = images_sub_path + test_list_item
        test_list_path_25.append(test_list_item)

    for test_list_item in test_list_50:
        test_list_item = images_sub_path + test_list_item
        test_list_path_50.append(test_list_item)

    for test_list_item in test_list_75:
        test_list_item = images_sub_path + test_list_item
        test_list_path_75.append(test_list_item)

    for test_list_item in test_list_100:
        test_list_item = images_sub_path + test_list_item
        test_list_path_100.append(test_list_item)

    for test_list_item in test_list_150:
        test_list_item = images_sub_path + test_list_item
        test_list_path_150.append(test_list_item)

    for test_list_item in test_list_300:
        test_list_item = images_sub_path + test_list_item
        test_list_path_300.append(test_list_item)

    for test_list_item in test_list_all:
        test_list_item = images_sub_path + test_list_item
        test_list_path_all.append(test_list_item)

    # for 3 images
    out_3['train_filenames'] = train_list_path_3
    out_3['test_filenames'] = test_list_path_3
    out_3['val_filenames'] = test_list_path_3
    out_3['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_3['orientation_override'] = 'none'
    out_3['applied_scale'] = 1.0

    # for 5 images
    out_5['train_filenames'] = train_list_path_5
    out_5['test_filenames'] = test_list_path_5
    out_5['val_filenames'] = test_list_path_5
    out_5['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_5['orientation_override'] = 'none'
    out_5['applied_scale'] = 1.0

    # for 15 images
    out_15['train_filenames'] = train_list_path_15
    out_15['test_filenames'] = test_list_path_15
    out_15['val_filenames'] = test_list_path_15
    out_15['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_15['orientation_override'] = 'none'
    out_15['applied_scale'] = 1.0

    # for 20 images
    out_20['train_filenames'] = train_list_path_20
    out_20['test_filenames'] = test_list_path_20
    out_20['val_filenames'] = test_list_path_20
    out_20['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_20['orientation_override'] = 'none'
    out_20['applied_scale'] = 1.0

    # for 10 images
    out_10['train_filenames'] = train_list_path_10
    out_10['test_filenames'] = test_list_path_10
    out_10['val_filenames'] = test_list_path_10
    out_10['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_10['orientation_override'] = 'none'
    out_10['applied_scale'] = 1.0

    # for 25 images
    out_25['train_filenames'] = train_list_path_25
    out_25['test_filenames'] = test_list_path_25
    out_25['val_filenames'] = test_list_path_25
    out_25['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_25['orientation_override'] = 'none'
    out_25['applied_scale'] = 1.0

    # for 50 images
    out_50['train_filenames'] = train_list_path_50
    out_50['test_filenames'] = test_list_path_50
    out_50['val_filenames'] = test_list_path_50
    out_50['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_50['orientation_override'] = 'none'
    out_50['applied_scale'] = 1.0

    # for 75 images
    out_75['train_filenames'] = train_list_path_75
    out_75['test_filenames'] = test_list_path_75
    out_75['val_filenames'] = test_list_path_75
    out_75['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_75['orientation_override'] = 'none'
    out_75['applied_scale'] = 1.0

    # for 100 images
    out_100['train_filenames'] = train_list_path_100
    out_100['test_filenames'] = test_list_path_100
    out_100['val_filenames'] = test_list_path_100
    out_100['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_100['orientation_override'] = 'none'
    out_100['applied_scale'] = 1.0

    # for 150 images
    out_150['train_filenames'] = train_list_path_150
    out_150['test_filenames'] = test_list_path_150
    out_150['val_filenames'] = test_list_path_150
    out_150['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_150['orientation_override'] = 'none'
    out_150['applied_scale'] = 1.0

    # for 300 images
    out_300['train_filenames'] = train_list_path_300
    out_300['test_filenames'] = test_list_path_300
    out_300['val_filenames'] = test_list_path_300
    out_300['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_300['orientation_override'] = 'none'
    out_300['applied_scale'] = 1.0

    # for all images
    out_all['train_filenames'] = train_list_path_all
    out_all['test_filenames'] = test_list_path_all
    out_all['val_filenames'] = test_list_path_all
    out_all['applied_transform'] = np.eye(4)[:3, :].tolist()
    out_all['orientation_override'] = 'none'
    out_all['applied_scale'] = 1.0

    file = open(pose_file_path)
    data = json.load(file)
    # surgar_box_transform = np.array([[0.991475, 0.116089, -0.0591719, -2.68266],
    #                                  [-0.114243, 0.992881, 0.0337031, -9.05489],
    #                                  [0.0626632, -0.0266558, 0.997679, 29.3344],
    #                                  [0, 0, 0, 1]])
    #
    # pitcher_transform = np.array([[0.682047, 0.730187, -0.0404812, -4.48842],
    #                              [-0.73099, 0.682338, -0.00827062, -17.5149],
    #                               [0.0215827, 0.0352323, 0.999146, -76.9252],
    #                               [0, 0, 0, 1 ]])
    #
    # drill_transform = np.array([[0.099503, 0.994734, -0.0245527, 6.33158],
    #                             [0.0244238, 0.0222261, 0.999453, -55.1966],
    #                             [0.994736, -0.100049, -0.0220835, 10.1362],
    #                             [0, 0, 0, 1 ]])

    # print("Creating the subset")
    pbar = tqdm(total=len(data), ncols=100, desc="Progress", unit=" files")
    for key in data:
        img_path = os.path.join(img_dir, key)
        # K = data[key]['K']
        W2C = np.array(data[key]["W2C"])

        # convert into mm scale
        W2C[0:3, 3] *= 1000

        # W2C = np.matmul(W2C, np.linalg.inv(surgar_box_transform))
        # W2C = np.matmul(W2C, np.linalg.inv(pitcher_transform))
        # W2C = np.matmul(W2C, np.linalg.inv(drill_transform))
        W2C = np.matmul(W2C, np.linalg.inv(ycbv_transform))


        # to keep the object with same orientation as the original
        r = R.from_euler('zyx', [-90,0,-90], degrees=True)
        W2C[:3,:3] = np.matmul(W2C[:3,:3], r.as_matrix())



        # convert into m scale
        W2C[0:3, 3] *= 0.001

        c2w = np.linalg.inv(W2C)

        tran = np.matmul(c2w, flip_mat)
        sharp = sharpness(imagePath=img_path)

        frame = {"file_path": os.path.join(images_sub_path, key), "sharpness": sharp, "transform_matrix": tran}

        if key in train_list_3:
            out_3["frames"].append(frame.copy())

        if key in train_list_5:
            out_5["frames"].append(frame.copy())

        if key in train_list_15:
            out_15["frames"].append(frame.copy())

        if key in train_list_20:
            out_20["frames"].append(frame.copy())

        if key in train_list_10:
            out_10["frames"].append(frame.copy())

        if key in train_list_25:
            out_25["frames"].append(frame.copy())

        if key in train_list_50:
            out_50["frames"].append(frame.copy())

        if key in train_list_75:
            out_75["frames"].append(frame.copy())

        if key in train_list_100:
            out_100["frames"].append(frame.copy())

        if key in train_list_150:
            out_150["frames"].append(frame.copy())

        if key in train_list_300:
            out_300["frames"].append(frame.copy())

        if key in train_list_all:
            out_all["frames"].append(frame.copy())
        pbar.update(1)

    pbar.close()

    # print("Calculating the center of the scene")
    # # find the center of the scene
    # totw = 0.0
    # totp = np.array([0.0, 0.0, 0.0])
    # pbar = tqdm(total=len(out_all["frames"]), ncols=100, desc="Progress", unit=" files")
    # for f in out_all["frames"]:
    #     mf = f["transform_matrix"][0:3, :]
    #     for g in out_all["frames"]:
    #         mg = g["transform_matrix"][0:3, :]
    #         p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
    #         if w > 0.00001:
    #             totp += p * w
    #             totw += w
    #     pbar.update(1)
    # pbar.close()
    # if totw > 0.0:
    #     totp /= totw

    # # scale the scene to fit into -1 to 1 scale and center it
    outs = [out_3, out_5, out_15, out_20, out_10, out_25, out_50, out_75, out_100, out_150, out_300, out_all]

    # print("Scaling and recentering the scene")
    pbar = tqdm(total=len(outs), ncols=100, desc="Progress", unit=" files")
    for out in outs:
        out["real_world_scale"] = 0.300
        # out["real_world_center"] = totp.tolist()
        for frame in out["frames"]:
            frame["transform_matrix"] = np.array(frame["transform_matrix"])
            frame["transform_matrix"][0:3, 3] *= 1/ 0.300
            # frame["transform_matrix"][0:3, 3] -= totp
        pbar.update(1)
    pbar.close()

    # Convert the numpy array to list for json
    pbar = tqdm(total=len(outs), ncols=100, desc="Progress", unit=" files")
    for out in outs:
        for f in out["frames"]:
            if not type(f["transform_matrix"]) == list:
                f["transform_matrix"] = f["transform_matrix"].tolist()
        pbar.update(1)
    pbar.close()

    # save the dataset
    pbar = tqdm(total=len(outs), ncols=100, desc="Progress", unit=" files")
    for out in outs:
        subset = len(out["train_filenames"])
        if subset > 300:
            subset = "all"
        out_file_path = os.path.join(dataset_dir, 'transforms_{}.json'.format(subset))
        with open(out_file_path, 'w') as outfile:
            json.dump(out, outfile, indent=4)
        pbar.update(1)

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_base = args.dataset_dir
    transform_location = args.transform_location

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))

    for dataset in os.listdir(dataset_base):
        dataset_dir = os.path.join(dataset_base, dataset)
        transform_dir = os.path.join(transform_location, dataset)
        if dataset != "16_wood_block":
            continue
        process.apply_async(main, args=(dataset_dir,transform_dir,))
    process.close()
    process.join()

    print("Done")