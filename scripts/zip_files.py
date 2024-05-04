import os
import argparse
import multiprocessing
import subprocess

parser = argparse.ArgumentParser(description='Zip files in a directory')
parser.add_argument('--base', type=str, help='Base directory of datset', required=True)
parser.add_argument('--destination', type=str, help='Location to save zipped files', required=True)

def zip_files(destination, base, dataset):
    files_to_zip = ["undistorted_images" , "transforms_3.json" , "transforms_5.json", "transforms_10.json" ,
    "transforms_15.json" , "transforms_20.json" , "transforms_25.json" , "transforms_50.json" , "transforms_75.json",
    "transforms_100.json" , "transforms_150.json", "transforms_300.json", "transforms_all.json" ]
    for files in files_to_zip:
        subprocess.run(['zip', '-ur', os.path.join(destination, dataset + '.zip'), os.path.join(base, dataset, files)])

if __name__ == '__main__':
    args = parser.parse_args()
    base = args.base
    destination = args.destination

    process = multiprocessing.Pool(len(os.listdir(base)))

    for dataset in os.listdir(base):
        zip_files(destination, base, dataset)
