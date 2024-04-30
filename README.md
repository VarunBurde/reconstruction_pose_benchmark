# Reconstruction and Pose Estimation Benchmark Dataset

![setup.png](webpage_content%2Fsetup.png)

## Dataset Structure

### Calibrated images

The dataset consists of images of the 21
[YCB-V](https://arxiv.org/abs/1711.00199) objects captured using a robotic
manipulator Kuka IIWA 14 with a Basler camera with 2K resolution. The image
poses are calculated from the kinematic chain of the robot, the camera is
calibrated with hand-eye configuration and the OpenCV routine, and the image
undistortion is done with [COLMAP](https://colmap.github.io/). 

The dataset contains one folder per object containing undisorted images, object
masks, camera calibration and camera poses in multiple formats compatible with
[NerfStudio](https://docs.nerf.studio/) and
[COLMAP](https://colmap.github.io/). 

The data is oragnized as folllows:

<pre>├── Object
│   ├── transforms_100.json
│   ├── transforms_10.json
│   ├── transforms_150.json
│   ├── transforms_15.json
│   ├── transforms_20.json
│   ├── transforms_25.json
│   ├── transforms_300.json
│   ├── transforms_3.json
│   ├── transforms_50.json
│   ├── transforms_5.json
│   ├── transforms_75.json
│   ├── transforms_all.json
│   └── undistorted_images
│       ├── img_0001.png
│       ├── img_0002.png
│       ├── img_0003.png
│       ├── img_0004.png
│       ├── ...
│       └── ...
│   └── masks
│       ├── img_0001.png
│       ├── img_0002.png
│       ├── img_0003.png
│       ├── img_0004.png
│       ├── ...
│       └── ...
│   └── colmap
│       ├── points3D.txt
│       ├── images.txt
│       └── cameras.txt
...............
</pre>

- `transforms_N.json`: contains the camera poses and intrinsics for a subset of
  `N` image in the format expected by the [NerfStudio](https://docs.nerf.studio/reference/api/data/dataparsers.html) dataparser.
  The poses have already been pre-processed to fit inside a unit cube to facilitate the Nerf-based reconstruction.
  The subset of images are in the range of 3, 5, 10, 15, 20, 25, 50, 75, 100, 150, 300, and all images. 
  The poses can be converted into meter scale by multiplying the poses with the
  parameter `real_world_scale` in the JSON file. 
  All camera poses are registered to the coordinate frame of the object defined by the 
  [BOP release of the YCB-V meshes](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip).
- `colmap`: contains an empty 3D model with the camera calibration and the image poses written using the [COLMAP convention](https://colmap.github.io/format.html#text-format).
- `undistorted_images`: contains the undistorted images
- `masks`: contains the object's masks


### Reconstructed Meshes

The mesh dataset consists of the meshes of the 21
[YCB-V](https://arxiv.org/abs/1711.00199) objects reconstructed from the
abovementioned calibrated images and using the following methods.
The codebase used for each method is specified between `( )`.

~~~

1.  BakedSDF ( SDFstudio)
2.  Capture Reality (Native)
3.  Colmap (Native)
4.  MonoSDF (SDFstudio)
5.  Nerfacto (NerfStudio)
6.  Neuralangelo (Native)
7.  NeUS (SDfStudio)
8.  Instant-NGP (Native)
9.  Plenoxels (Native)
10. VolSDF (SDFstudio)
12. UniSurf (SDFstudio)

~~~

For a given method, each object mesh is contained in one folder either in `.obj` or `.ply` format.
The folder also holds a texture file whenever the reconstrcution method
produced one, else the mesh is colored only.
As for the camera psoes, the meshes are already registered to the coordinate frame of the object defined by the 
  [BOP release of the YCB-V meshes](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip) 

<pre>├── Method_subset
│   ├── 01_master_chef_can
│   │   ├── mesh_0.png
│   │   ├── mesh.mtl
│   │   └── mesh.obj
│   ├── 02_cracker_box
│   │   ├── mesh_0.png
│   │   ├── mesh.mtl
│   │   └── mesh.obj
│   ├── 04_tomatoe_soup_can
│   │   ├── mesh_0.png
│   │   ├── mesh.mtl
│   │   └── mesh.obj
│   ├── 05_mustard_bottle
│   ....
..............
</pre>


## Download 

### Image Dataset
The image dataset can be downloaded from the [data page](https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/Image_dataset/) with:

```bash
# e.g. to download the 19_large_clamp object,

mkdir -p data/19_large_clamp
wget https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/Image_dataset/19_large_clamp.zip -P data/
unzip data/19_large_clamp.zip -d data/19_large_clamp
```

### Mesh Dataset

The mesh dataset can be downloded from the [data_page](https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/reconstructed_meshes/) with:

```bash
wget https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/reconstructed_meshes/<method>_<dataset_size>.zip"

# e.g. To download meshes reconstructed by Nerfacto trained on all images
wget https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/reconstructed_meshes/<nerfacto>_<all>.zip"
```


## Scripts

We provide scripts to convert the poses from the 
[NerfStudio](https://docs.nerf.studio/reference/api/data/dataparsers.html)
convention to the 
[COLMAP](https://colmap.github.io/format.html#text-format) and 
[BOP-Benchmark](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md) ones.

ones:

```bash

python3 -m scripts.nerf_to_colmap --dataset_dir <path to object image folder>

python3 -m scripts.nerf_to_bop --dataset_dir <path to object image folder>
```
