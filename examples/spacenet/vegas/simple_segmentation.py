import re
import random
import os
from os.path import join

import rastervision as rv
from rastervision.utils.files import list_paths
from examples.utils import str_to_bool


class SpacenetVegasSimpleSegmentation(rv.ExperimentSet):
    def exp_main(self, raw_uri, root_uri, test_run=False):
        """Run an experiment on the Spacenet Vegas building dataset.

        This is a simple example of how to do semantic segmentation on data that
        doesn't require any pre-processing or special permission to access.

        Args:
            raw_uri: (str) directory of raw data (the root of the Spacenet dataset)
            root_uri: (str) root directory for experiment output
            test_run: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        base_uri = join(
            raw_uri, 'SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas')
        raster_uri = join(base_uri, 'RGB-PanSharpen')
        label_uri = join(base_uri, 'geojson/buildings')
        raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        label_fn_prefix = 'buildings_AOI_2_Vegas_img'
        label_paths = list_paths(label_uri, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]

        test_run = str_to_bool(test_run)
        exp_id = 'spacenet-simple-seg'
        num_steps = 1e5
        batch_size = 8
        debug = False
        if test_run:
            exp_id += '-test'
            num_steps = 1
            batch_size = 1
            debug = True
            scene_ids = scene_ids[0:10]

        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)
        # Workaround to handle scene 1000 missing on S3.
        if '1000' in scene_ids:
            scene_ids.remove('1000')
        num_train_ids = round(len(scene_ids) * 0.8)
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes({
                                'Building': (1, 'orange'),
                                'Background': (2, 'black')
                            }) \
                            .with_chip_options(
                                chips_per_scene=9,
                                debug_chip_probability=1.0,
                                negative_survival_probability=0.25,
                                target_classes=[1],
                                target_count_threshold=1000) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(rv.MOBILENET_V2) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        def make_scene(id):
            train_image_uri = os.path.join(raster_uri,
                                           '{}{}.tif'.format(raster_fn_prefix, id))

            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .with_stats_transformer() \
                .build()

            vector_source = os.path.join(
                label_uri, '{}{}.geojson'.format(label_fn_prefix, id))
            label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                .with_vector_source(vector_source) \
                .with_rasterizer_options(2) \
                .build()

            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_raster_source(label_raster_source) \
                .build()

            scene = rv.SceneConfig.builder() \
                .with_task(task) \
                .with_id(id) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

            return scene

        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()

        # Need to use stats_analyzer because imagery is uint16.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
