import os
from os.path import join

import rastervision as rv
from examples.utils import get_scene_info, str_to_bool, save_image_crop


class ObjectDetectionExperiments(rv.ExperimentSet):
    def exp_xview(self, raw_uri, processed_uri, root_uri, test=False):
        """Object detection experiment on xView data.

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
        exp_id = 'xview-vehicles'
        batch_size = 16
        num_epochs = 20
        debug = False
        train_scene_info = get_scene_info(join(processed_uri, 'train-scenes.csv'))
        val_scene_info = get_scene_info(join(processed_uri, 'val-scenes.csv'))

        if test:
            exp_id += '-test'
            batch_size = 2
            num_epochs = 2
            debug = True
            train_scene_info = train_scene_info[0:1]
            val_scene_info = val_scene_info[0:1]

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({'vehicle': (1, 'red')}) \
                            .with_chip_options(neg_ratio=1.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        backend = rv.BackendConfig.builder(rv.PYTORCH_OBJECT_DETECTION) \
            .with_task(task) \
            .with_train_options(
                lr=1e-4,
                one_cycle=True,
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
                save_image_crop(raster_uri, crop_uri, size=600, min_features=5)
                raster_uri = crop_uri

            id = os.path.splitext(os.path.basename(raster_uri))[0]
            label_source = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
                                               .with_uri(label_uri) \
                                               .build()

            return rv.SceneConfig.builder() \
                                 .with_task(task) \
                                 .with_id(id) \
                                 .with_raster_source(raster_uri) \
                                 .with_label_source(label_source) \
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
