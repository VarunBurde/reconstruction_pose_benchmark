import requests
import subprocess
import multiprocessing
import os

url = 'https://data.ciirc.cvut.cz/public/projects/2023BenchmarkPoseEstimationReconstructedMesh/Image_dataset/'
ext = 'zip'

def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]


if __name__ == '__main__':
    files = listFD(url, ext)

    process = multiprocessing.Pool(len(files))

    for file in files:
        # get basename of the file
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0]

        # create a directory for the file
        subprocess.run(['mkdir', file_name])

        # use multiprocessing to download the files faster
        process.apply_async(subprocess.run, args=(['wget', file],))

    process.close()
    process.join()

    process = multiprocessing.Pool(len(files))

    for file in files:
        # get basename of the file
        file_name = file.split('/')[-1]
        file_name = file_name.split('.')[0]

        # use subprocess to unzip the files
        process.apply_async(subprocess.run, args=(['unzip', file.split('/')[-1], '-d', file_name],))

    process.close()
    process.join()

    current_dir = os.getcwd()

    # delete the zip files from current directory
    for files in os.listdir(current_dir):
        if files.endswith('.zip'):
            print("removing file: ", files)
            os.remove(files)