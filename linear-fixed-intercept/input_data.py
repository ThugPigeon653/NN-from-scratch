import requests
import gzip
import json
import os
import random
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DataManager():
    # DICT FORMAT: train_data={index:{name:<name>, inputs:[<input_values>], label:<label>}}
    config_path:str="config/shared-runtime-config.json"
    train_data:{}
    # default behaviour added, to train on an open-source data set
    def __init__(self, data_path:str=None) -> None:
        if data_path==None and not os.path.exists(self.config_path):
            sys.stdout.write("Using default training data...")
            self.download_and_save_mnist_to_json(self.config_path)
        else:
            # TODO: alternative datasets should be loaded here.
            pass
        try:
            with open(self.config_path, "r") as file:
                self.train_data=json.load(file)
        except Exception as e:
            sys.stdout.write(f"Could not find config file for '{self.config_path}'\n{e}")

    @staticmethod
    def download_and_save_mnist_to_json(json_file_path):
        url = "http://yann.lecun.com/exdb/mnist/"
        files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
        train_data = {}

        for file in files:
            file_url = url + file
            response = requests.get(file_url, stream=True)
            with open(file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)

        with gzip.open(files[0], 'rb') as image_file, gzip.open(files[1], 'rb') as label_file:
            magic_number = int.from_bytes(label_file.read(4), 'big')
            size = int.from_bytes(label_file.read(4), 'big')
            image_file.read(8)  

            for i in range(size):
                label = int.from_bytes(label_file.read(1), 'big')
                image = [int.from_bytes(image_file.read(1), 'big') / 255.0 for _ in range(28 * 28)]
                train_data[i] = {'name': f'{i}', 'inputs': image, 'label': label}

        with open(json_file_path, 'w') as json_file:
            json.dump(train_data, json_file)

    def get_random_training_data_point(self)->{}:
        return self.train_data[str(random.randint(0, len(self.train_data) - 1))]
