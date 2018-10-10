import os

import rastervision as rv


def build_scene(task, data_uri, id, channel_order=None):
    id = id.replace('-', '_')
    raster_source_uri = '{}/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(
        data_uri, id)

    label_source_uri = '{}/5_Labels_for_participants/top_potsdam_{}_label.tif'.format(
        data_uri, id)

    # Using with_rgb_class_map because input TIFFs have classes encoded as RGB colors.
    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_rgb_class_map(task.class_map) \
        .with_raster_source(label_source_uri) \
        .build()

    # URI will be injected by scene config.
    # Using with_rgb(True) because we want prediction TIFFs to be in RGB format.
    label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
        .with_rgb(True) \
        .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(id) \
                          .with_raster_source(raster_source_uri,
                                              channel_order=channel_order) \
                          .with_label_source(label_source) \
                          .with_label_store(label_store) \
                          .build()

    return scene


class PotsdamSemanticSegmentation(rv.ExperimentSet):
    def exp_main(self, root_uri, data_uri, test_run=False):
        """Run an experiment on the ISPRS Potsdam dataset.

        Uses Tensorflow Deeplab backend with Mobilenet architecture. Should get to
        F1 score of ~0.86 (including clutter class) after 6 hours of training on P3
        instance.

        Args:
            root_uri: (str) root directory for experiment output
            data_uri: (str) root directory of Potsdam dataset
            test_run: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        if test_run == 'True':
            test_run = True
        elif test_run == 'False':
            test_run = False

        train_ids = ['2-10', '2-11', '3-10', '3-11', '4-10', '4-11', '4-12', '5-10',
                     '5-11', '5-12', '6-10', '6-11', '6-7', '6-9', '7-10', '7-11',
                     '7-12', '7-7', '7-8', '7-9']
        val_ids = ['2-12', '3-12', '6-12']
        # infrared, red, green
        channel_order = [3, 0, 1]

        debug = False
        batch_size = 16
        chips_per_scene = 500
        num_steps = 100000
        model_type = rv.MOBILENET_V2

        # Better results can be obtained at a greater computational expense using
        # num_steps = 150000
        # model_type = rv.XCEPTION_65

        if test_run:
            debug = True
            num_steps = 1
            batch_size = 1
            chips_per_scene = 50
            train_ids = train_ids[0:1]
            val_ids = val_ids[0:1]

        classes = {
            'Car': (1, '#ffff00'),
            'Building': (2, '#0000ff'),
            'Low Vegetation': (3, '#00ffff'),
            'Tree': (4, '#00ff00'),
            'Impervious': (5, "#ffffff"),
            'Clutter': (6, "#ff0000")
        }

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(256) \
                            .with_classes(classes) \
                            .with_chip_options(
                                chips_per_scene=chips_per_scene,
                                debug_chip_probability=0.2,
                                negative_survival_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        train_scenes = [build_scene(task, data_uri, id, channel_order)
                        for id in train_ids]
        val_scenes = [build_scene(task, data_uri, id, channel_order)
                      for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id('potsdam-seg') \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
