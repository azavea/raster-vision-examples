import os
import random

import rastervision as rv
from rastervision.utils.files import list_paths


def label_path_to_scene_properties(label_path):
    '''
    Given the path to a label source, extract the corresponding image
    uri and the scene id
    '''

    # imagery for these three states come from different years
    state_years = {
        'ia': '2017',
        'ok': '2017',
        'tx': '2016'
    }

    # find the scene id
    scene_id = os.path.basename(label_path).replace('.geojson', '')

    # this is the base uri that all of the naip imagery is on
    image_base_uri = 's3://rasterfoundry-production-data-us-east-1/naip-visualization/'

    # extract the state, year and folder name for the image
    parts = label_path.split('/')
    state = parts[-2]
    year = state_years[state]
    folder = scene_id.split('_')[1][:5]

    # construct the image uri
    image_uri = '{}{}/{}/100cm/rgb/{}/{}.tif'.format(
        image_base_uri, state, year, folder, scene_id)

    return image_uri, label_path, scene_id


class ObjectDetectionExperiments(rv.ExperimentSet):

    def exp_uswtdb(self, root_uri, test='False'):

        # we will use the label geojson files to dictate which scenes to create
        # this is the folder where the labels for each of the three states are
        base_uri = 's3://raster-vision-wind-turbines/labels'

        # find all geojson label stores within that base uro
        label_paths = list_paths(base_uri, ext='.geojson')

        # option to run a small subset of the entire experiment
        if test == 'True':
            label_paths = label_paths[0:10]

        # divide label paths into traing and validation sets
        random.seed(5678)
        random.shuffle(label_paths)
        num_train_label_paths = round(len(label_paths) * 0.8)
        train_label_paths = label_paths[0:num_train_label_paths]
        val_label_paths = label_paths[num_train_label_paths:]

        # specify steps and batch size
        # NUM_STEPS = 817
        NUM_STEPS = 100000
        BATCH_SIZE = 8
        if test == 'True':
            NUM_STEPS = 1
            BATCH_SIZE = 1


        def build_scene(label_path):
            '''
            Build a scene from a label path
            '''

            # extract image uri, label uri and scene id from label path
            train_image_uri, label_source_uri, id = label_path_to_scene_properties(
                label_path)

            # build raster source
            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .build()

            # build label sourcs
            label_source = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
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
                                replace_model=False) \
            .with_model_defaults(rv.FASTER_RCNN_RESNET50_COCO) \
            .build()

        mobilenet = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
            .with_task(task) \
            .with_debug(True) \
            .with_batch_size(BATCH_SIZE)\
            .with_num_steps(NUM_STEPS) \
            .with_train_options(do_monitoring=True,
                                replace_model=False) \
            .with_model_defaults(rv.SSD_MOBILENET_V2_COCO) \
            .build()

        mobilenet_v1 = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
            .with_task(task) \
            .with_debug(True) \
            .with_batch_size(BATCH_SIZE)\
            .with_num_steps(NUM_STEPS) \
            .with_train_options(do_monitoring=True,
                                replace_model=False) \
            .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
            .build()

        frcnn_inception = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
            .with_task(task) \
            .with_debug(True) \
            .with_batch_size(BATCH_SIZE)\
            .with_num_steps(NUM_STEPS) \
            .with_train_options(do_monitoring=True,
                                replace_model=False) \
            .with_model_defaults(rv.FASTER_RCNN_INCEPTION_V2_COCO) \
            .build()

        # create training and validation scenes
        train_scenes = [build_scene(label_path) for label_path in train_label_paths]
        val_scenes = [build_scene(label_path) for label_path in val_label_paths]

        # build dataset config
        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        # build experiment
        rn_experiment = rv.ExperimentConfig.builder() \
            .with_root_uri(root_uri) \
            .with_task(task) \
            .with_dataset(dataset) \
            .with_id('uswtdb-object-detection-resnet-ia-ok-tx') \
            .with_backend(resnet) \
            .build()

        mn_experiment = rv.ExperimentConfig.builder() \
            .with_root_uri(root_uri) \
            .with_task(task) \
            .with_dataset(dataset) \
            .with_chip_key('uswtdb-object-detection-resnet-ia-ok-tx') \
            .with_id('uswtdb-object-detection-mobilenet-ia-ok-tx') \
            .with_backend(mobilenet) \
            .build()

        mn1_experiment = rv.ExperimentConfig.builder() \
            .with_root_uri(root_uri) \
            .with_task(task) \
            .with_dataset(dataset) \
            .with_chip_key('uswtdb-object-detection-resnet-ia-ok-tx') \
            .with_id('uswtdb-object-detection-mobilenet_v1-ia-ok-tx') \
            .with_backend(mobilenet) \
            .build()

        frcnn_inception_experiment = rv.ExperimentConfig.builder() \
            .with_root_uri(root_uri) \
            .with_task(task) \
            .with_dataset(dataset) \
            .with_chip_key('uswtdb-object-detection-resnet-ia-ok-tx') \
            .with_id('uswtdb-object-detection-frcnn_inception-ia-ok-tx') \
            .with_backend(frcnn_inception) \
            .build()

        return [rn_experiment, mn_experiment, mn1_experiment, frcnn_inception_experiment]

if __name__ == '__main__':
    rv.main()
