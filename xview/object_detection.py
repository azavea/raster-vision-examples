import os

import rastervision as rv

from .data import (get_xview_training_scene_info, get_xview_val_scene_info)


class ObjectDetectionExperiments(rv.ExperimentSet):
    """Object detection experiments on xView data.
    Be sure you've run the data prep notebook before running this experiment.
    """

    def scene_maker(self, task):
        def f(x):
            (raster_uri, label_uri) = x
            id = os.path.splitext(os.path.basename(raster_uri))[0]
            label_source = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION_GEOJSON) \
                                               .with_uri(label_uri) \
                                               .build()

            return rv.SceneConfig.builder() \
                                 .with_task(task) \
                                 .with_id(id) \
                                 .with_raster_source(raster_uri) \
                                 .with_label_source(label_source) \
                                 .build()
        return f

    def exp_xview(self, root_uri):
        # Number of training steps. Increase this for longer train time
        # and better results.
        NUM_STEPS = 100000

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({"vehicle": (1, "red")}) \
                            .with_chip_options(neg_ratio=1.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        # Set up the backend base.
        # Here we create a builder with the configuration that
        # is common between the two experiments. We don't call
        # build, so that we can branch off the builder based on
        # using a mobilenet or faster rcnn resnet model.

        backend_base = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                    .with_task(task) \
                                    .with_debug(True) \
                                    .with_batch_size(16) \
                                    .with_num_steps(NUM_STEPS)

        mobilenet = backend_base \
                    .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
                    .build()

        resnet = backend_base \
                    .with_model_defaults(rv.FASTER_RCNN_RESNET50_COCO) \
                    .build()

        make_scene = self.scene_maker(task)

        train_scenes = list(map(make_scene, get_xview_training_scene_info()))
        val_scenes = list(map(make_scene, get_xview_val_scene_info()))

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        # Set up the experiment base.
        # Notice we set the "chip_key". This allows the two experiments to
        # use the same training chips, so that the chip command is only run
        # once for both experiments.

        experiment_base = rv.ExperimentConfig.builder() \
                                            .with_root_uri(root_uri) \
                                            .with_task(task) \
                                            .with_dataset(dataset) \
                                            .with_chip_key("xview-object_detection")

        mn_experiment = experiment_base \
                        .with_id('xview-object-detection-mobilenet') \
                        .with_backend(mobilenet) \
                        .build()

        rn_experiment = experiment_base \
                        .with_id('xview-object-detection-resnet') \
                        .with_backend(resnet) \
                        .build()

        return [mn_experiment, rn_experiment]

if __name__ == '__main__':
    rv.main()
