# Dataset instruction

![setup.png](webpage_content%2Fsetup.png)

## Image Dataset
The dataset consists of images of the updated 21 YCB-V objects captured using a robotic manipulator Kuka IIWA 14 with a 
Baysler camera with 2K resolution. The poses are calculated from the kinematic chain of the robot and the camera 
calibrated with hand-eye configuration. 

The directory structure of the images is as follows. 
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
...............
</pre>

Each dataset pose has been post-processed to fit inside a unit cube to facilitate the Nerf-based reconstruction and used 
Nerfstudio Parser as described [here](https://docs.nerf.studio/reference/api/data/dataparsers.html). The JSON files consist 
of a subset of images used for the benchmark. All images are undistorted according to the estimated distortion parameters.
The subset of images are in the range of 3, 5, 10, 15, 20, 25, 50, 75, 100, 150, 300, and all images. Intrinsic parameters are 
also provided in the JSON files. The poses can be converted into meter scale by multiplying the poses with with parameter 
"real_world_scale" in the JSON file. All camera poses are aligned with the BOP dataset mesh world frame which means the 
reconstructed mesh such be roughly in the same world frame as meshes specified 
[here](https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/ycbv/ycbv_models.zip)

## Downloading the dataset 

The dataset can be downloaded from the following link: [YCB-V Dataset](https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/Bop_images/)

To download each object, wget or other similar command can be used. For example, to download the 19_large_clamp object,
the following command can be used:

```bash
wget https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/Image_dataset/19_large_clamp.zip
```

## Mesh Dataset
The mesh dataset consists of the reconstructed mesh of the updated 21 YCB-V objects with the following methods with their
implementation in the parenthesis.

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

The directory structure of the meshes is as follows.

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

Each meshes are in either .obj or .ply format. Directory may consist of meshes with Texture file or either color already 
embedded in the mesh vertices. Each meshes are scaled, cropped and aligned to fit the GT meshes from YCB-V dataset. 
The alignement is done using the ICP algorithm from the Open3D library. 

To download the mesh dataset, the following command can be used:


```bash
wget https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/reconstructed_meshes/"object_subset.zip"
```

We provide the sample script to convert the Nerf data to COLMAP and BOP format. It can be found under scripts
```bash
python3 nerf_to_bop.py --dataset_dir path/to/direactory
python3 nerf_to_colmap.py --dataset_dir path/to/direactory
```