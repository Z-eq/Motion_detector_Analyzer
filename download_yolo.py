import os
import urllib.request

def download_file(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {output_path}.")
    else:
        print(f"{output_path} already exists.")

def download_yolo_files():
    yolo_files = {
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }

    for file_name, url in yolo_files.items():
        download_file(url, file_name)

if __name__ == "__main__":
    download_yolo_files()
