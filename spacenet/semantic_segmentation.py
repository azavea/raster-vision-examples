import re
import random
import os
from abc import abstractmethod

import rastervision as rv
from rastervision.utils.files import list_paths


BUILDINGS = 'buildings'
ROADS = 'roads'


class SpacenetConfig(object):
    @staticmethod
    def create(use_remote_data, target):
        if target.lower() == BUILDINGS:
            return VegasBuildings(use_remote_data)
        elif target.lower() == ROADS:
            return VegasRoads(use_remote_data)
        else:
            raise ValueError('{} is not a valid target.'.format(target))

    def get_raster_source_uri(self, id):
        return os.path.join(
            self.base_uri, self.raster_dir,
            '{}{}.tif'.format(self.raster_fn_prefix, id))

    def get_label_source_uri(self, id):
        return os.path.join(
            self.base_uri, self.label_dir,
            '{}{}.geojson'.format(self.label_fn_prefix, id))

    def get_scene_ids(self):
        label_dir = os.path.join(self.base_uri, self.label_dir)
        label_paths = list_paths(label_dir, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(self.label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]
        return scene_ids

    @abstractmethod
    def get_class_map(self):
        pass


class VegasRoads(SpacenetConfig):
    def __init__(self, use_remote_data):
        self.base_uri = '/opt/data/AOI_2_Vegas_Roads_Train'
        if use_remote_data:
            self.base_uri = 's3://spacenet-dataset/SpaceNet_Roads_Competition/Train/AOI_2_Vegas_Roads_Train'  # noqa

        self.raster_dir = 'RGB-PanSharpen'
        self.label_dir = 'geojson/spacenetroads'
        self.raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        self.label_fn_prefix = 'spacenetroads_AOI_2_Vegas_img'

    def get_class_map(self):
        # First class should be background when using GeoJSONRasterSource
        return {
            'Road': (1, 'orange'),
            'Background': (2, 'black')
        }


class VegasBuildings(SpacenetConfig):
    def __init__(self, use_remote_data):
        self.base_uri = '/opt/data/AOI_2_Vegas_Train'
        if use_remote_data:
            self.base_uri = 's3://spacenet-dataset/SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas'  # noqa

        self.raster_dir = 'RGB-PanSharpen'
        self.label_dir = 'geojson/buildings'
        self.raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        self.label_fn_prefix = 'buildings_AOI_2_Vegas_img'

    def get_class_map(self):
        # First class should be background when using GeoJSONRasterSource
        return {
            'Building': (1, 'orange'),
            'Background': (2, 'black')
        }


def build_scene(task, spacenet_config, id, channel_order=None):
    # Need to use stats_transformer because imagery is uint16.
    raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                      .with_uri(spacenet_config.get_raster_source_uri(id)) \
                      .with_channel_order(channel_order) \
                      .with_stats_transformer() \
                      .build()

    background_class_id = 2
    line_buffer = 15
    label_raster_source = rv.RasterSourceConfig.builder(rv.GEOJSON_SOURCE) \
        .with_uri(spacenet_config.get_label_source_uri(id)) \
        .with_rasterizer_options(background_class_id, line_buffer=line_buffer) \
        .build()

    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_raster_source(label_raster_source) \
        .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source) \
                          .with_label_source(label_source) \
                          .build()

    return scene


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


class SpacenetSemanticSegmentation(rv.ExperimentSet):
    def exp_main(self, root_uri, target=BUILDINGS, use_remote_data=True, test=False):
        """Run an experiment on the Spacenet Vegas road or building segmentation dataset.

        Args:
            root_uri: (str): root of where to put output
            target: (str) 'buildings' or 'roads'
            use_remote_data: (bool or str) if True or 'True', then use data from S3,
                else local
            test: (bool or str) if True or 'True', run a very small experiment as a
                test and generate debug output
        """
        test = str_to_bool(test)
        use_remote_data = str_to_bool(use_remote_data)
        spacenet_config = SpacenetConfig.create(use_remote_data, target)
        experiment_id = target

        scene_ids = spacenet_config.get_scene_ids()
        if len(scene_ids) == 0:
            raise ValueError('No scenes found. Something is configured incorrectly.')
        random.seed(1234)
        random.shuffle(scene_ids)
        split_ratio = 0.8
        nb_train_ids = round(len(scene_ids) * split_ratio)
        train_ids = scene_ids[0:nb_train_ids]
        val_ids = scene_ids[nb_train_ids:]
        channel_order = [0, 1, 2]

        debug = False
        batch_size = 8
        chips_per_scene = 9
        num_steps = 1e5
        model_type = rv.MOBILENET_V2
        # Use full dataset.
        nb_train_scenes = len(train_ids)
        nb_val_scenes = len(val_ids)

        # Slightly better results can be obtained at a greater computational expense
        # using the following config. Takes 24 hours to train on P3 instance.
        # num_steps = int(1.5e5)
        # model_type = rv.XCEPTION_65

        if test:
            debug = True
            num_steps = 1
            batch_size = 1
            chips_per_scene = 9
            nb_train_scenes = 8
            nb_val_scenes = 2

        train_ids = train_ids[0:nb_train_scenes]
        val_ids = val_ids[0:nb_val_scenes]

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes(spacenet_config.get_class_map()) \
                            .with_chip_options(
                                chips_per_scene=chips_per_scene,
                                debug_chip_probability=1.0,
                                negative_survival_probability=0.25,
                                target_classes=[1],
                                target_count_threshold=1000) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        train_scenes = [build_scene(task, spacenet_config, id, channel_order)
                        for id in train_ids]
        val_scenes = [build_scene(task, spacenet_config, id, channel_order)
                      for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()

        # Need to use stats_analyzer because imagery is uint16.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(experiment_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
