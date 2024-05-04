import os
import multiprocessing
import json
import shutil

def main(dataset_dir):
    print(dataset_dir)
    nerf_dir = os.path.join(dataset_dir, 'nerf')
    if os.path.exists(nerf_dir):
        shutil.rmtree(nerf_dir)
    os.makedirs(nerf_dir, exist_ok=True)

    optimised_pose_path = os.path.join(dataset_dir, 'optimise_pose_transforms_all.json')
    optimised_pose = json.load(open(optimised_pose_path, 'r'))

    transfrom_subset = ["transforms_{}.json".format(subset) for subset in
                        [3, 5, 10, 15, 20, 25, 50, 75, 100, 150, 300, "all"]]
    for subset in transfrom_subset:
        subset_path = os.path.join(dataset_dir, subset)
        subset_data = json.load(open(subset_path, 'r'))
        for frame in subset_data['frames']:
            for frame_o in optimised_pose['frames']:
                if frame['file_path'] == frame_o['file_path']:
                    frame['transform_matrix'] = frame_o['transform_matrix']


        with open(os.path.join(nerf_dir, subset), 'w') as f:
            json.dump(subset_data, f, indent=4)


if __name__ == '__main__':
    dataset_base = "/home/varun/PycharmProjects/remote_dir/ciirc/ciirc_nfs/datasets/burdeva1/img_dataset/raw_data_fibonacci"

    process = multiprocessing.Pool(len(os.listdir(dataset_base)))
    for dataset in os.listdir(dataset_base):
        print(f"Processing {dataset}")
        dataset_dir = os.path.join(dataset_base, dataset)
        process.apply_async(main, args=(dataset_dir,))
    process.close()
    process.join()








