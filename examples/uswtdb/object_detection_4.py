import os
import random

import rastervision as rv
from rastervision.utils.files import list_paths

class ObjectDetectionExperiments(rv.ExperimentSet):

    def exp_uswtdb(self, root_uri, test='False'):
        base_uri = 's3://raster-vision-iowa-wind-turbines'
        image_dir = 'images'
        label_dir = 'labels'

        label_paths = list_paths(os.path.join(base_uri, label_dir), ext='.geojson')

        scene_ids = [os.path.basename(x).replace('.geojson', '') for x in label_paths]
        
        if test == 'True':
            scene_ids = scene_ids[0:10]
        random.seed(5678)
        random.shuffle(scene_ids)

        num_train_ids = round(len(scene_ids) * 0.8)
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]

        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({"turbine": (1, "red")}) \
                            .with_chip_options(neg_ratio=3.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

        NUM_STEPS = 20000
        BATCH_SIZE = 16
        if test == 'True':
            NUM_STEPS = 1
            BATCH_SIZE = 1

        resnet = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                    .with_task(task) \
                                    .with_debug(True) \
                                    .with_batch_size(BATCH_SIZE)\
                                    .with_num_steps(NUM_STEPS) \
                                    .with_train_options(do_monitoring=True,
                                                        replace_model=True) \
                                    .with_model_defaults(rv.FASTER_RCNN_RESNET50_COCO) \
                                    .build()

        def build_scene(id):
            train_image_uri = os.path.join(base_uri, image_dir, '{}.tif'.format(id))
            raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .build()

            label_source_uri = os.path.join(base_uri, label_dir, '{}.geojson'.format(id))
            label_source = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
                .with_uri(label_source_uri) \
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

        rn_experiment = rv.ExperimentConfig.builder() \
                                            .with_root_uri(root_uri) \
                                            .with_task(task) \
                                            .with_dataset(dataset) \
                                            .with_id('uswtdb-object-detection-resnet-4') \
                                            .with_backend(resnet) \
                                            .build()

        return rn_experiment

if __name__ == '__main__':
    rv.main()
