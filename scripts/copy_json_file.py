import os

dataset_base = "/home/varun/PycharmProjects/remote_dir/ciirc/ciirc_nfs/datasets/burdeva1/img_dataset/raw_data_fibonacci"

for dataset in os.listdir(dataset_base):
    for subset in [3, 5, 10, 15, 20, 25, 50, 75, 100, 150, 300, "all"]:
        subset_path = os.path.join(dataset_base, dataset,"nerf", f"transforms_{subset}.json")

        new_subset_path = os.path.join(dataset_base, dataset, f"transforms_{subset}.json")

        # copyt subset path to new subset path
        os.system(f"cp {subset_path} {new_subset_path}")




