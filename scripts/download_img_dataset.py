from bs4 import BeautifulSoup
import requests
import subprocess
import threading
import os
import argparse


parser = argparse.ArgumentParser(
    prog='Downloads the files from the given url',
    description='Downloads the files from the given url')
parser.add_argument('--dataset_destination', type=str, required=True,
                    help='Path to the dataset directory')
args = parser.parse_args()



url = 'https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/Image_dataset/'
ext = 'zip'


def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


if __name__ == '__main__':
    files = listFD(url, ext)

    dataset_destination = args.dataset_destination

    threads = []

    for file in files:
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0]

        subprocess.run(['mkdir', os.path.join(dataset_destination, file_name)])

        # download the file without certificate verification
        t = threading.Thread(target=subprocess.run, args=(['wget', '-O', os.path.join(dataset_destination, file_name,
                                                            file_name + '.' + ext), file, '--no-check-certificate'],))


        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    threads = []

    for file in files:
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0]

        t = threading.Thread(target=subprocess.run, args=(['unzip', os.path.join(dataset_destination, file_name,
                                                                                 file_name + '.' + ext), '-d',
                                                          os.path.join(dataset_destination, file_name)],))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for file in files:
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0]
        os.remove(os.path.join(dataset_destination, file_name, file_name + '.' + ext))

    print("Done")

