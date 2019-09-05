import subprocess
from os.path import join
import json
import pprint
import glob
import os

import click

from rastervision.utils.files import (
    list_paths, download_or_copy, file_exists, make_dir, file_to_json)


cfg = [
    {
        'key': 'cowc-object-detection',
        'module': 'examples.cowc.object_detection',
        'local': {
            'raw_uri': '/opt/data/raw-data/isprs-potsdam/',
            'processed_uri': '/opt/data/examples/cowc-potsdam/processed-data',
        },
        'remote': {
            'raw_uri': 's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri': 's3://raster-vision-lf-dev/examples/cowc-potsdam/processed-data',
        },
        'rv_profile': 'tf',
    },
    {
        'key': 'potsdam-semantic-segmentation-pytorch',
        'module': 'examples.potsdam.semantic_segmentation',
        'local': {
            'raw_uri': '/opt/data/raw-data/isprs-potsdam/',
            'processed_uri': '/opt/data/examples/potsdam/processed-data',
        },
        'remote': {
            'raw_uri': 's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri': 's3://raster-vision-lf-dev/examples/potsdam/processed-data',
        }
    },
    {
        'key': 'potsdam-semantic-segmentation-tf',
        'module': 'examples.potsdam.semantic_segmentation',
        'local': {
            'raw_uri': '/opt/data/raw-data/isprs-potsdam/',
            'processed_uri': '/opt/data/examples/potsdam/processed-data',
        },
        'remote': {
            'raw_uri': 's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri': 's3://raster-vision-lf-dev/examples/potsdam/processed-data',
        },
        'extra_args': [['use_tf', 'True']],
        'rv_profile': 'tf',
    },
    {
        'key': 'spacenet-rio-chip-classification-pytorch',
        'module': 'examples.spacenet.rio.chip_classification',
        'local': {
            'raw_uri': '/opt/data/raw-data/spacenet-dataset',
            'processed_uri': '/opt/data/examples/spacenet/rio/processed-data',
        },
        'remote': {
            'raw_uri': 's3://spacenet-dataset/',
            'processed_uri': 's3://raster-vision-lf-dev/examples/spacenet/rio/processed-data',
        },
    },
    {
        'key': 'spacenet-rio-chip-classification-tf',
        'module': 'examples.spacenet.rio.chip_classification',
        'local': {
            'raw_uri': '/opt/data/raw-data/spacenet-dataset',
            'processed_uri': '/opt/data/examples/spacenet/rio/processed-data',
        },
        'remote': {
            'raw_uri': 's3://spacenet-dataset/',
            'processed_uri': 's3://raster-vision-lf-dev/examples/spacenet/rio/processed-data',
        },
        'extra_args': [['use_tf', 'True']],
        'rv_profile': 'tf',
    },
    {
        'key': 'spacenet-vegas-simple-segmentation',
        'module': 'examples.spacenet.vegas.simple_segmentation',
        'local': {
            'raw_uri': '/opt/data/raw-data/spacenet-dataset',
        },
        'remote': {
            'raw_uri': 's3://spacenet-dataset/',
        },
    },
    {
        'key': 'xview-object-detection',
        'module': 'examples.xview.object_detection',
        'local': {
            'raw_uri': 's3://raster-vision-xview-example/raw-data',
            'processed_uri': '/opt/data/examples/xview/processed-data',
        },
        'remote': {
            'raw_uri': 's3://raster-vision-xview-example/raw-data',
            'processed_uri': 's3://raster-vision-lf-dev/examples/xview/processed-data',
        },
        'rv_profile': 'tf',
    },
]


def run_experiment(exp_cfg, root_uri, test=True, remote=False, commands=None):
    uris = exp_cfg['remote'] if remote else exp_cfg['local']
    cmd = ['rastervision']
    rv_profile = exp_cfg.get('rv_profile')
    if rv_profile is not None:
        cmd += ['-p', rv_profile]
    cmd += ['run', 'aws_batch' if remote else 'local', '-e', exp_cfg['module']]
    cmd += ['-a', 'raw_uri', uris['raw_uri']]
    if 'processed_uri' in uris:
        cmd += ['-a', 'processed_uri', uris['processed_uri']]
    cmd += ['-a', 'root_uri', join(root_uri, exp_cfg['key'])]
    cmd += ['-a', 'test', 'True' if test else 'False']
    extra_args = exp_cfg.get('extra_args')
    if extra_args:
        for k, v in extra_args:
            cmd += ['-a', str(k), str(v)]
    cmd += ['--splits', '2']
    if commands is not None:
        cmd += ['-r'] + commands

    print('running command:')
    print(' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('failure!')
        print(' '.join(cmd))
        exit()


def collect_experiment(key, root_uri, output_dir, get_pred_package=False):
    print('\nCollecting experiment {}...\n'.format(key))

    if root_uri.startswith('s3://'):
        predict_package_uris = list_paths(join(root_uri, key, 'bundle'), ext='predict_package.zip')
        eval_json_uris = list_paths(join(root_uri, key, 'eval'), ext='eval.json')
    else:
        predict_package_uris = glob.glob(join(root_uri, key, 'bundle', '*', 'predict_package.zip'))
        eval_json_uris = glob.glob(join(root_uri, key, 'eval', '*', 'eval.json'))

    if len(predict_package_uris) > 1 or len(eval_json_uris) > 1:
        print('Cannot collect from key with multiple experiments!!!')
        return

    if len(predict_package_uris) == 0 or len(eval_json_uris) == 0:
        print('Missing output!!!')
        return

    predict_package_uri = predict_package_uris[0]
    eval_json_uri = eval_json_uris[0]
    make_dir(join(output_dir, key))
    if get_pred_package:
        download_or_copy(predict_package_uri, join(output_dir, key))

    download_or_copy(eval_json_uri, join(output_dir, key))

    eval_json = file_to_json(join(output_dir, key, 'eval.json'))
    pprint.pprint(eval_json['overall'], indent=4)


def validate_keys(keys):
    exp_keys = [exp_cfg['key'] for exp_cfg in cfg]
    invalid_keys = set(keys).difference(exp_keys)
    if invalid_keys:
        raise ValueError('{} are invalid keys'.format(', '.join(keys)))


@click.group()
def test():
    pass


@test.command()
@click.argument('root_uri')
@click.argument('keys', nargs=-1)
@click.option('--test', is_flag=True)
@click.option('--remote', is_flag=True)
@click.option('--commands')
def run(root_uri, keys, test, remote, commands):
    run_all = len(keys) == 0
    validate_keys(keys)

    if commands is not None:
        commands = commands.split(' ')
    for exp_cfg in cfg:
        if run_all or exp_cfg['key'] in keys:
            run_experiment(exp_cfg, root_uri, test=test, remote=remote,
                           commands=commands)


@test.command()
@click.argument('root_uri')
@click.argument('output_dir')
@click.argument('keys', nargs=-1)
@click.option('--get-pred-package', is_flag=True)
def collect(root_uri, output_dir, keys, get_pred_package):
    run_all = len(keys) == 0
    validate_keys(keys)

    for exp_cfg in cfg:
        key = exp_cfg['key']
        if run_all or key in keys:
            collect_experiment(key, root_uri, output_dir, get_pred_package)

if __name__ == '__main__':
    test()