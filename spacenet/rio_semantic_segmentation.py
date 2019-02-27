import os

import rastervision as rv

from .data import (get_rio_training_scene_info, get_rio_val_scene_info)

AOI_URI = 's3://spacenet-dataset/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


class SemanticSegmentationExperiments(rv.ExperimentSet):
    """Semantic segmentation experiments on SpaceNet Rio data.
    Be sure you've run the data prep notebook before running this experiment.
    """

    def scene_maker(self, task):
        def f(x):
            (raster_uri, label_uri) = x
            id = os.path.splitext(os.path.basename(raster_uri))[0]

            background_class_id = 2
            label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                .with_vector_source(label_uri) \
                .with_rasterizer_options(background_class_id) \
                .build()
            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_raster_source(label_raster_source) \
                .build()

            return rv.SceneConfig.builder() \
                                 .with_task(task) \
                                 .with_id(id) \
                                 .with_raster_source(raster_uri) \
                                 .with_label_source(label_source) \
                                 .with_aoi_uri(AOI_URI) \
                                 .build()
        return f

    def exp_main(self, root_uri, test='False'):
        """Run a semantic segmentation experiment on the Spacenet Rio building dataset.

        Run a small test experiment if test is 'True'.

        Args:
            root_uri: (str): root of where to put output
            test: (str) 'True' or 'False'
        """
        exp_id = 'spacenet-rio-building-semseg'
        debug = False
        batch_size = 8
        num_steps = 150000
        model_type = rv.MOBILENET_V2

        if test == 'True':
            debug = True
            num_steps = 1
            batch_size = 1

        class_map = {
            'Building': (1, 'orange'),
            'Background': (2, 'black')
        }

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes(class_map) \
                            .with_chip_options(
                                stride=300,
                                window_method='sliding',
                                debug_chip_probability=0.25) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_config({
                                    'min_scale_factor': '0.75',
                                    'max_scale_factor': '1.25'},
                                    ignore_missing_keys=True, set_missing_keys=True) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        make_scene = self.scene_maker(task)
        train_info = get_rio_training_scene_info()
        val_info = get_rio_val_scene_info()
        if test == 'True':
            train_info = train_info[0:1]
            val_info = val_info[0:1]
        train_scenes = list(map(make_scene, train_info))
        val_scenes = list(map(make_scene, val_info))

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
