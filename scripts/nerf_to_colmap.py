import numpy as np
import os
import argparse
import json
import multiprocessing
import collections
from scipy.spatial.transform import Rotation as R
import shutil
import sys
import sqlite3
import subprocess

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

class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(
            CREATE_CAMERAS_TABLE
        )
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(
            CREATE_IMAGES_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(
            CREATE_KEYPOINTS_TABLE
        )
        self.create_matches_table = lambda: self.executescript(
            CREATE_MATCHES_TABLE
        )
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        prior_q=np.full(4, np.NaN),
        prior_t=np.full(3, np.NaN),
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                image_id,
                name,
                camera_id,
                prior_q[0],
                prior_q[1],
                prior_q[2],
                prior_q[3],
                prior_t[0],
                prior_t[1],
                prior_t[2],
            ),
        )
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )


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

        c2w = np.array(frame['transform_matrix'])

        # convert opengl to opencv
        c2w = np.matmul(c2w, flip_mat)

        # scale scene to real world scale
        c2w[:3, 3] *= real_scale

        # convert scene to w2c
        w2c = np.linalg.inv(c2w)

        Rot = w2c[:3, :3]
        T = w2c[:3, 3]

        # to transform from opengl to opencv
        r = R.from_euler('zyx', [-90, 0,-90], degrees=True)
        Rot = np.matmul(Rot, np.linalg.inv(r.as_matrix()))

        images[img_id] = Image(id=img_id, camera_id=0, qvec=rotmat2qvec(Rot), tvec=T, name=file_name, xys=[0], point3D_ids=[])

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
        '0': Camera(id=0, model="OPENCV",  width=width, height=height,
                    params=[fl_x, fl_y, cx, cy, k1, k2, p1, p2])}
    write_cameras_text(cameras, os.path.join(path, "cameras" + ".txt"))

def write_empty_points3d_file(path):
    path = os.path.join(path, "points3D.txt")
    with open(path, "w") as fid:
        fid.write("\n")

def edit_database(database_path, transforms):

    db = COLMAPDatabase.connect(database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()
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

    parameters = [fl_x, fl_y, cx, cy, k1, k2, p1, p2]


    model, width, height, params = (
        4,
        width,
        height,
        parameters,
    )
    camera_id1 = db.add_camera(model, width, height, params)


    img_ids = {}

    real_scale = transforms["real_world_scale"]
    images = {}
    for frame in transforms['frames']:
        file_path = frame['file_path']
        file_name = os.path.basename(file_path)
        img_id = file_name.split('.')[0]
        img_id = img_id.split('_')[1]
        img_id = int(img_id)

        c2w = np.array(frame['transform_matrix'])

        # convert opengl to opencv
        c2w = np.matmul(c2w, flip_mat)

        # scale scene to real world scale
        c2w[:3, 3] *= real_scale

        # convert scene to w2c
        w2c = np.linalg.inv(c2w)

        Rot = w2c[:3, :3]
        T = w2c[:3, 3]

        # to transform from opengl to opencv
        r = R.from_euler('zyx', [-90, 0, -90], degrees=True)
        Rot = np.matmul(Rot, np.linalg.inv(r.as_matrix()))

        img_ids[img_id] = db.add_image(file_name, camera_id1, prior_q=rotmat2qvec(Rot), prior_t=T)

    db.commit()

def run_colmap(database_path, image_path, sparse_input_dir  ,sparse_output_dir, dense_path):

    # feature extraction
    subprocess.run(["colmap", "feature_extractor",
                    "--database_path", database_path,
                    "--image_path", image_path])

    # exhaustive matching
    subprocess.run(["colmap", "exhaustive_matcher",
                    "--database_path", database_path])

    # point triangulation
    subprocess.run(["colmap", "point_triangulator",
                    "--database_path", database_path,
                    "--image_path", image_path,
                    "--input_path", sparse_input_dir,
                    "--output_path", sparse_output_dir])

    # image undistortion
    subprocess.run(["colmap", "image_undistorter",
                    "--image_path", image_path,
                    "--input_path", sparse_input_dir,
                    "--output_path", dense_path])

    # patch match stereo
    subprocess.run(["colmap", "patch_match_stereo",
                    "--workspace_path", dense_path])

    dense_model_path = os.path.join(dense_path, "fused.ply")
    # stereo fusion
    subprocess.run(["colmap", "stereo_fusion",
                    "--workspace_path", dense_path,
                    "--output_path", dense_model_path])

def main(dataset_dir):
    # object_name = os.path.basename(dataset_dir)
    # object_id = YCB_data[object_name]
    mask_dir = os.path.join(dataset_dir, 'mask_mesh')
    image_path = os.path.join(dataset_dir, 'undistorted_images')
    transform_json = os.path.join(dataset_dir, 'optimise_pose_transforms_all.json')
    output_dir = os.path.join(dataset_dir, 'colmap_format')
    sparse_input_dir = os.path.join(output_dir, 'sparse_input')
    sparse_output_dir = os.path.join(output_dir, 'sparse_output')
    database_path = os.path.join(output_dir, "database.db")
    dense_path = os.path.join(output_dir, 'dense')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sparse_input_dir, exist_ok=True)
    os.makedirs(sparse_output_dir, exist_ok=True)
    os.makedirs(dense_path, exist_ok=True)

    with open(transform_json, 'r') as f:
        transforms = json.load(f)

    write_cam_file(transforms, sparse_input_dir)
    write_image_text(transforms, sparse_input_dir)
    write_empty_points3d_file(sparse_input_dir)
    edit_database(database_path, transforms)

    # To run the colmap commands
    # run_colmap(database_path, image_path, sparse_input_dir, sparse_output_dir,dense_path)


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_base = args.dataset_dir

    # # debug one dataset
    # dataset = os.path.join(dataset_base, '02_cracker_box')
    # main(dataset)

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))

    for dataset in os.listdir(dataset_base):
        print(f"Processing {dataset}")
        dataset_dir = os.path.join(dataset_base, dataset)
        process.apply_async(main, args=(dataset_dir,))
    process.close()
    process.join()

    print("Done")