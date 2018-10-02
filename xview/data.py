import csv


def get_xview_training_scene_info():
    data_path = '/opt/data/xview/training_scenes.csv'
    with open(data_path) as f:
        reader = csv.reader(f)
        return list(reader)


def get_xview_val_scene_info():
    data_path = '/opt/data/xview/val_scenes.csv'
    with open(data_path) as f:
        reader = csv.reader(f)
        return list(reader)
