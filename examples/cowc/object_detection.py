import os
from os.path import join

import rastervision as rv
from examples.utils import str_to_bool, save_image_crop


class CowcObjectDetectionExperiments(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False, use_tf=False):
        """Object detection on COWC (Cars Overhead with Context) Potsdam dataset

        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        exp_id = 'cowc-object-detection2'
        num_steps = 100000
        batch_size = 8
        debug = False
        train_scene_ids = ['2_10', '2_11', '2_12', '2_14', '3_11',
                           '3_13', '4_10', '5_10', '6_7', '6_9']
        val_scene_ids = ['2_13', '6_8', '3_10']

        if test:
            exp_id += '-test'
            num_steps = 1
            batch_size = 2
            debug = True

            train_scene_ids = train_scene_ids[0:1]
            val_scene_ids = val_scene_ids[0:1]

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({'vehicle': (1, 'red')}) \
                            .with_chip_options(neg_ratio=1.0,
                                               ioa_thresh=0.5) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        if use_tf:
            backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                        .with_task(task) \
                                        .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
                                        .with_debug(debug) \
                                        .with_batch_size(batch_size) \
                                        .with_num_steps(num_steps) \
                                        .build()
        else:
            batch_size = 16
            num_epochs = 10
            if test:
                batch_size = 2
                num_epochs = 2

            backend = rv.BackendConfig.builder(rv.PYTORCH_OBJECT_DETECTION) \
                .with_task(task) \
                .with_train_options(
                    lr=2e-4,
                    one_cycle=True,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    model_arch='resnet18',
                    debug=debug,
                    run_tensorboard=False) \
                .build()

        def make_scene(id):
            raster_uri = join(
                raw_uri, '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(id))
            label_uri = join(
                processed_uri, 'labels', 'all', 'top_potsdam_{}_RGBIR.json'.format(id))

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                                size=1000, min_features=5)
                raster_uri = crop_uri

            return rv.SceneConfig.builder() \
                                 .with_id(id) \
                                 .with_task(task) \
                                 .with_raster_source(raster_uri, channel_order=[0, 1, 2]) \
                                 .with_label_source(label_uri) \
                                 .build()

        train_scenes = [make_scene(id) for id in train_scene_ids]
        val_scenes = [make_scene(id) for id in val_scene_ids]

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
