import os
from os.path import join

import rastervision as rv
from examples.utils import str_to_bool, save_image_crop


class PotsdamSemanticSegmentation(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False):
        """Run an experiment on the ISPRS Potsdam dataset.

        Uses Tensorflow Deeplab backend with Mobilenet architecture. Should get to
        F1 score of ~0.86 (including clutter class) after 6 hours of training on a P3
        instance.

        Args:
            raw_uri: (str) directory of raw data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        exp_id = 'potsdam-seg'
        train_ids = ['2-10', '2-11', '3-10', '3-11', '4-10', '4-11', '4-12', '5-10',
                     '5-11', '5-12', '6-10', '6-11', '6-7', '6-9', '7-10', '7-11',
                     '7-12', '7-7', '7-8', '7-9']
        val_ids = ['2-12', '3-12', '6-12']
        # infrared, red, green
        channel_order = [3, 0, 1]
        debug = False
        batch_size = 8
        num_steps = 100000
        model_type = rv.MOBILENET_V2

        if test:
            debug = True
            num_steps = 1
            batch_size = 1
            train_ids = train_ids[0:1]
            val_ids = val_ids[0:1]
            exp_id += '-test'

        classes = {
            'Car': (1, '#ffff00'),
            'Building': (2, '#0000ff'),
            'Low Vegetation': (3, '#00ffff'),
            'Tree': (4, '#00ff00'),
            'Impervious': (5, "#ffffff"),
            'Clutter': (6, "#ff0000")
        }

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes(classes) \
                            .with_chip_options(window_method='sliding',
                                               stride=300, debug_chip_probability=1.0) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        def make_scene(id):
            id = id.replace('-', '_')
            raster_uri = '{}/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(
                raw_uri, id)

            label_uri = '{}/5_Labels_for_participants/top_potsdam_{}_label.tif'.format(
                raw_uri, id)

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, size=600)
                raster_uri = crop_uri

            # Using with_rgb_class_map because label TIFFs have classes encoded as RGB colors.
            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_rgb_class_map(task.class_map) \
                .with_raster_source(label_uri) \
                .build()

            # URI will be injected by scene config.
            # Using with_rgb(True) because we want prediction TIFFs to be in RGB format.
            label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                .with_rgb(True) \
                .build()

            scene = rv.SceneConfig.builder() \
                                  .with_task(task) \
                                  .with_id(id) \
                                  .with_raster_source(raster_uri,
                                                      channel_order=channel_order) \
                                  .with_label_source(label_source) \
                                  .with_label_store(label_store) \
                                  .build()

            return scene

        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
