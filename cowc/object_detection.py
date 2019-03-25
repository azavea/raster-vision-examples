import os

import rastervision as rv


class ObjectDetectionExperiments(rv.ExperimentSet):
    """Object detection experiments on COWC potsdam data.
    """

    def get_task(self):
        return rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({"vehicle": (1, "red")}) \
                            .with_chip_options(neg_ratio=1.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.1,
                                                  score_thresh=0.5) \
                            .build()

    def get_backend_base(self, task):
        return rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                               .with_task(task) \
                               .with_model_defaults(rv.SSD_MOBILENET_V1_COCO) \
                               .with_debug(True)

    def exp_cowc_local(self):
        root_uri = '/opt/data/cowc/potsdam-local/root_uri'

        task = self.get_task()

        backend =  self.get_backend_base(task) \
                       .with_batch_size(4) \
                       .with_num_steps(350) \
                       .build()

        train_scene_info = [("/opt/data/cowc/potsdam-local/cowc-potsdam-test/2-10.tif",
                             "/opt/data/cowc/potsdam-local/cowc-potsdam-test/2-10.json")]
        val_scene_info = [("/opt/data/cowc/potsdam-local/cowc-potsdam-test/2-13.tif",
                           "/opt/data/cowc/potsdam-local/cowc-potsdam-test/2-13.json")]

        def  make_scene(tup):
            image, labels = tup
            id = os.path.splitext(os.path.basename(image))[0]
            return rv.SceneConfig.builder() \
                                 .with_id(id) \
                                 .with_task(task) \
                                 .with_raster_source(image, channel_order=[0,1,2]) \
                                 .with_label_source(labels) \
                                 .build()

        train_scenes = list(map(make_scene, train_scene_info))
        val_scenes = list(map(make_scene, val_scene_info))

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('cowc-object-detection-local') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment

    def exp_cowc_full(self, root_uri):
        train_scene_ids = ['2_10', '2_11', '2_12', '2_14', '3_11',
                           '3_13', '4_10', '5_10', '6_7', '6_9']

        val_scene_ids = ['2_13', '6_8', '3_10']

        task = self.get_task()

        backend = self.get_backend_base(task) \
                      .with_batch_size(8) \
                      .with_num_steps(100000) \
                      .build()

        def make_scene(id, label_uri):
            img_uri = '{}/isprs-potsdam/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(
                root_uri, id)

            return rv.SceneConfig.builder() \
                                 .with_id(id) \
                                 .with_task(task) \
                                 .with_raster_source(img_uri, channel_order=[0,1,2]) \
                                 .with_label_source(label_uri) \
                                 .build()

        train_label_uri = '{}/labels/train.json'.format(root_uri)
        train_scenes = list(map(lambda id: make_scene(id, train_label_uri),
                                train_scene_ids))
        val_label_uri = '{}/labels/test.json'.format(root_uri)
        val_scenes = list(map(lambda id: make_scene(id, val_label_uri),
                              val_scene_ids))

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('cowc-object-detection-full') \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
