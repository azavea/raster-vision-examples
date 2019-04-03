import csv
from io import StringIO

from rastervision.utils.files import file_to_str


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


def get_scene_info(csv_uri):
    csv_str = file_to_str(csv_uri)
    reader = csv.reader(StringIO(csv_str), delimiter=',')
    return list(reader)
