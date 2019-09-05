import os
from os.path import join

import rastervision as rv
from examples.utils import get_scene_info, str_to_bool, save_image_crop

aoi_path = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


class SemanticSegmentationExperiments(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False):
        """Semantic segmentation experiment on Spacenet Rio dataset.

        Run the data prep notebook before running this experiment. Note all URIs can be
        local or remote.

        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        exp_id = 'spacenet-rio-semseg'
        debug = False
        batch_size = 8
        num_epochs = 20

        train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
        val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))

        if test:
            exp_id += '-test'
            debug = True
            num_epochs = 1
            batch_size = 2
            train_scene_info = train_scene_info[0:1]
            val_scene_info = val_scene_info[0:1]

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
                                debug_chip_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_SEMANTIC_SEGMENTATION) \
            .with_task(task) \
            .with_train_options(
                lr=1e-4,
                batch_size=batch_size,
                num_epochs=num_epochs,
                model_arch='resnet18',
                debug=debug) \
            .build()

        def make_scene(scene_info):
            (raster_uri, label_uri) = scene_info
            raster_uri = join(raw_uri, raster_uri)
            label_uri = join(processed_uri, label_uri)

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                                size=600)
                raster_uri = crop_uri

            aoi_uri = join(raw_uri, aoi_path)
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
                                 .with_aoi_uri(aoi_uri) \
                                 .build()

        train_scenes = [make_scene(info) for info in train_scene_info]
        val_scenes = [make_scene(info) for info in val_scene_info]

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
