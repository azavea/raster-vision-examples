import os
from os.path import join

import rastervision as rv
from examples.utils import get_scene_info, str_to_bool, save_image_crop

aoi_path = 'AOIs/AOI_1_Rio/srcData/buildingLabels/Rio_OUTLINE_Public_AOI.geojson'


class ChipClassificationExperiments(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False, use_tf=False):
        """Chip classification experiment on Spacenet Rio dataset.

        Run the data prep notebook before running this experiment. Note all URIs can be
        local or remote.

        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
            use_tf: (bool) if True, use Tensorflow Deeplab backend
        """
        test = str_to_bool(test)
        use_tf = str_to_bool(use_tf)
        exp_id = 'spacenet-rio-chip-classification'
        debug = False
        train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
        val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))

        if test:
            exp_id += '-test'
            debug = True
            train_scene_info = train_scene_info[0:1]
            val_scene_info = val_scene_info[0:1]

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(200) \
                            .with_classes({
                                'building': (1, 'red'),
                                'no_building': (2, 'black')
                            }) \
                            .build()

        if use_tf:
            num_epochs = 20
            batch_size = 32
            if test:
                num_epochs = 1
                batch_size = 1

            backend = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                .with_task(task) \
                .with_model_defaults(rv.RESNET50_IMAGENET) \
                .with_debug(debug) \
                .with_batch_size(batch_size) \
                .with_num_epochs(num_epochs) \
                .with_config({
                    'trainer': {
                        'options': {
                            'saveBest': True,
                            'lrSchedule': [
                                {
                                    'epoch': 0,
                                    'lr': 0.0005
                                },
                                {
                                    'epoch': 10,
                                    'lr': 0.0001
                                },
                                {
                                    'epoch': 15,
                                    'lr': 0.00001
                                }
                            ]
                        }
                    }
                }, set_missing_keys=True) \
                .build()
        else:
            num_epochs = 20
            batch_size = 32
            if test:
                num_epochs = 1
                batch_size = 2

            backend = rv.BackendConfig.builder(rv.PYTORCH_CHIP_CLASSIFICATION) \
                .with_task(task) \
                .with_train_options(
                    lr=1e-4,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    model_arch='resnet50',
                    debug=debug) \
                .build()

        def make_scene(scene_info):
            (raster_uri, label_uri) = scene_info
            raster_uri = join(raw_uri, raster_uri)
            label_uri = join(processed_uri, label_uri)
            aoi_uri = join(raw_uri, aoi_path)

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                                size=600, min_features=20)
                raster_uri = crop_uri

            id = os.path.splitext(os.path.basename(raster_uri))[0]
            label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                               .with_uri(label_uri) \
                                               .with_ioa_thresh(0.5) \
                                               .with_use_intersection_over_cell(False) \
                                               .with_pick_min_class_id(True) \
                                               .with_background_class_id(2) \
                                               .with_infer_cells(True) \
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
