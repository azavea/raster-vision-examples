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

    def exp_xview_resnet(self, root_uri):
        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({"vehicle": (1, "red")}) \
                            .with_chip_options(neg_ratio=0.0,
                                               ioa_thresh=1.0,
                                               window_method='sliding') \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        backend_config = 'https://gist.githubusercontent.com/simonkassel/3ed8cd946790be079879a5a738499bbc/raw/017c4e57e24523ebc2308f35c72619291a65b764/mobilenet-xview-vehicles.config'
        backend = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                  .with_task(task) \
                                  .with_template(backend_config)\
                                  .with_debug(True) \
                                  .with_batch_size(16) \
                                  .with_num_steps(150000) \
                                  .build()

        make_scene = self.scene_maker(task)

        train_scenes = list(map(make_scene, get_xview_training_scene_info()))
        val_scenes = list(map(make_scene, get_xview_val_scene_info()))

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('xview-object-detection') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment

if __name__ == '__main__':
    rv.main()
