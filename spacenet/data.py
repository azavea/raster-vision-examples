import csv

def get_rio_training_scene_info():
    data_path = '/opt/data/spacenet/rio/training_scenes.csv'
    with open(data_path) as f:
        reader = csv.reader(f)
        return list(reader)

def get_rio_val_scene_info():
    data_path = '/opt/data/spacenet/rio/val_scenes.csv'
    with open(data_path) as f:
        reader = csv.reader(f)
        return list(reader)
