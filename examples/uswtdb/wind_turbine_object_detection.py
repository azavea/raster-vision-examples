import os
import random

import rastervision as rv
from rastervision.utils.files import list_paths
from . import constants

def label_path_to_scene_properties(label_path):
    '''
    Given the path to a label source, extract the corresponding image
    uri and the scene id
    '''

    # find the scene id
    scene_id = os.path.basename(label_path).replace('.geojson', '')

    # this is the base uri that all of the naip imagery is on
    image_base_uri = 's3://rasterfoundry-production-data-us-east-1/naip-visualization/'

    # extract the state, year and folder name for the image
    parts = label_path.split('/')
    state = parts[-2]
    year = constants.STATE_YEARS[state]
    folder = scene_id.split('_')[1][:5]

    # construct the image uri
    image_uri = '{}{}/{}/100cm/rgb/{}/{}.tif'.format(
        image_base_uri, state, year, folder, scene_id)

    return image_uri, label_path, scene_id


class ObjectDetectionExperiments(rv.ExperimentSet):

    def exp_uswtdb(self, version, experiment_id, states=None, steps=None, test='False'):

        # we will use the label geojson files to dictate which scenes to create
        # this is the folder where the labels for each of the three states are
        base_uri = 's3://raster-vision-wind-turbines/labels'

        # find all geojson label stores within that base uri for selected states
        avail_states = list(constants.STATE_YEARS.keys())
        if states:
            state_list = states.split(',')
        else:
            state_list = avail_states
        
        label_paths = []
        for state in state_list:
            state = state.lower()
            if state in avail_states:
                state_uri = os.path.join(base_uri, state)
                label_paths += list_paths(state_uri, ext='.geojson')
            else:
                raise Exception(
                    'states must be a comma-separated list of any one of the '\
                    'following state abbreviations {}, found "{}".'.format(avail_states, state))

        # randomize the order of label paths
        random.seed(5678)
        random.shuffle(label_paths)
        
        # specify step default and batch size
        NUM_STEPS = 100000
        BATCH_SIZE = 16

        if steps:
            try:
                NUM_STEPS = int(steps.replace(',', ''))
            except ValueError:
                raise ValueError('The "steps" parameter must be a positive integer, got {}'.format(steps))
            if NUM_STEPS < 0:
                raise ValueError('The "steps" parameter must be positive, got {}'.format(NUM_STEPS))

        # option to run a small subset of the entire experiment
        if test == 'True':
            label_paths = label_paths[0:4]
            NUM_STEPS = 1
            BATCH_SIZE = 1
        
        # divide label paths into traing and validation sets
        num_train_label_paths = round(len(label_paths) * 0.8)
        train_label_paths = label_paths[0:num_train_label_paths]
        val_label_paths = label_paths[num_train_label_paths:]

        def build_scene(label_path):
            '''
            Build a scene from a label path
            '''

            # extract image uri, label uri and scene id from label path
            train_image_uri, label_source_uri, id = label_path_to_scene_properties(
                label_path)

            # build raster source
            raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .build()

            # build label source
            label_source = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION_GEOJSON) \
                .with_uri(label_source_uri) \
                .build()

            # build scene
            scene = rv.SceneConfig.builder() \
                .with_task(task) \
                .with_id(id) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

            return scene


        # build task config
        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({"turbine": (1, "red")}) \
                            .with_chip_options(neg_ratio=2.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        # build backend config
        resnet = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
            .with_task(task) \
            .with_debug(True) \
            .with_batch_size(BATCH_SIZE)\
            .with_num_steps(NUM_STEPS) \
            .with_train_options(do_monitoring=True,
                                replace_model=True) \
            .with_model_defaults(rv.FASTER_RCNN_RESNET50_COCO) \
            .build()

        # create training and validation scenes
        train_scenes = [build_scene(label_path) for label_path in train_label_paths]
        val_scenes = [build_scene(label_path) for label_path in val_label_paths]

        # build dataset config
        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        # get the root uri
        root_uri = os.path.join('s3://raster-vision-wind-turbines/versions', version)

        # build experiment
        rn_experiment = rv.ExperimentConfig.builder() \
            .with_root_uri(root_uri) \
            .with_task(task) \
            .with_dataset(dataset) \
            .with_id(experiment_id) \
            .with_backend(resnet) \
            .build()

        return rn_experiment

if __name__ == '__main__':
    rv.main()
