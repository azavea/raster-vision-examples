import re
import random
import os

import rastervision as rv
from rastervision.utils.files import list_paths

class SpacenetVegasSimpleSegmentation(rv.ExperimentSet):

    def exp_main(self, root_uri, test='False'):
        """Run an experiment on the Spacenet Vegas building dataset.

        This is a simple example of how to do semantic segmentation.

        Args:
            root_uri: (str): root of where to put output
            test: (str): 'True' or 'False', whether or not to run a small
                subset of the experiment
        """
        base_uri = 's3://spacenet-dataset/SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas'

        raster_dir = 'RGB-PanSharpen'
        label_dir = 'geojson/buildings'
        raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        label_fn_prefix = 'buildings_AOI_2_Vegas_img'

        label_dir = os.path.join(base_uri, label_dir)
        label_paths = list_paths(label_dir, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]

        if test == 'True':
            scene_ids = scene_ids[0:10]
        random.seed(5678)
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

        num_steps = 1e5
        batch_size = 8
        if test == 'True':
            num_steps = 1
            batch_size = 1

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                    .with_task(task) \
                                    .with_model_defaults(rv.MOBILENET_V2) \
                                    .with_num_steps(num_steps) \
                                    .with_batch_size(batch_size) \
                                    .with_debug(False) \
                                    .build()

        def build_scene(id):
            train_image_uri = os.path.join(base_uri, raster_dir,
                                           '{}{}.tif'.format(raster_fn_prefix, id))

            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .with_stats_transformer() \
                .build()

            vector_source = os.path.join(
                label_dir, '{}{}.geojson'.format(label_fn_prefix, id))
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

        train_scenes = [build_scene(id) for id in train_ids]
        val_scenes = [build_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()

        # Need to use stats_analyzer because imagery is uint16.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('simple_semantic_segmentation') \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
