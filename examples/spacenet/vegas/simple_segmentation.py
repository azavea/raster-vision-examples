import re
import random
import os
from os.path import join

import rastervision as rv
from rastervision.utils.files import list_paths
from examples.utils import str_to_bool

# Check out the docs for more information about the Raster Vision API:
# https://docs.rastervision.io/en/0.9/index.html#documentation

# Raster Vision ExperimentSets run similarly to the unittest.TestSuite.
# When you pass this script to the Raster Vision command line tool, it
# will reflexively run any ExperimentSet method that starts with 'exp_'.
# Return ExperimentConfigs from these methods in order to kick off their
# corresponding experiments.
class SpacenetVegasSimpleSegmentation(rv.ExperimentSet):
    def exp_main(self, raw_uri, root_uri, test=False):
        """Run an experiment on the Spacenet Vegas building dataset.

        This is a simple example of how to do semantic segmentation on data that
        doesn't require any pre-processing or special permission to access.

        Args:
            raw_uri: (str) directory of raw data (the root of the Spacenet dataset)
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        # Specify the location of the raw data
        base_uri = join(
            raw_uri, 'SpaceNet_Buildings_Dataset_Round2/spacenetV2_Train/AOI_2_Vegas')
        # The images and labels are in two separate directories within the base_uri
        raster_uri = join(base_uri, 'RGB-PanSharpen')
        label_uri = join(base_uri, 'geojson/buildings')
        # The tiff (raster) and geojson (label) files have have a naming convention of
        # '[prefix]_[image id].geojson.' The prefix indicates the type of data and the
        # image id indicates which scene each is associated with.
        raster_fn_prefix = 'RGB-PanSharpen_AOI_2_Vegas_img'
        label_fn_prefix = 'buildings_AOI_2_Vegas_img'
        # Find all of the image ids that have associated images and labels. Collect
        # these values to use as our scene ids.
        label_paths = list_paths(label_uri, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]

        # Set some trainin parameters:
        # The exp_id will be the label associated with this experiment, it will be used
        # to name the experiment config json.
        exp_id = 'spacenet-simple-seg'
        # Number of times to go through the entire dataset during training.
        num_epochs = 10
        # Number of images in each batch
        batch_size = 8
        # Specify whether or not to make debug chips (a zipped sample of png chips
        # that you can examine to help debug the chipping process)
        debug = False

        # This experiment includes an option to run a small test experiment before
        # running the whole thing. You can set this using the 'test' parameter. If
        # this parameter is set to True it will run a tiny test example with a new
        # experiment id. This will be small enough to run locally. It is recommended
        # to run a test example locally before submitting the whole experiment to AWs
        # Batch.
        test = str_to_bool(test)
        if test:
            exp_id += '-test'
            num_epochs = 1
            batch_size = 2
            debug = True
            scene_ids = scene_ids[0:10]

        # Split the data into training and validation sets:
        # Randomize the order of all scene ids
        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)
        # Workaround to handle scene 1000 missing on S3.
        if '1000' in scene_ids:
            scene_ids.remove('1000')
        # Figure out how many scenes make up 80% of the whole set
        num_train_ids = round(len(scene_ids) * 0.8)
        # Split the scene ids into training and validation lists
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]

        # The TaskConfigBuilder constructs a child class of TaskConfig that
        # corresponds to the type of computer vision task you are taking on.
        # This experiment includes a semantic segmentation task but Raster
        # Vision also has backends for object detection and chip classification.
        # Before building the task config you can also set parameters using
        # 'with_' methods. In the example below we set the chip size, the
        # pixel class names and colors, and addiitonal chip options.
        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes({
                                'Building': (1, 'orange'),
                                'Background': (2, 'black')
                            }) \
            .with_chip_options(
                                chips_per_scene=9,
                                debug_chip_probability=0.25,
                                negative_survival_probability=1.0,
                                target_classes=[1],
                                target_count_threshold=1000) \
            .build()

        # Next we will create a backend that is built on top of a third-party
        # deep learning library. In this case we will construct the
        # BackendConfig for the fastai semantic segmentation backend.
        backend = rv.BackendConfig.builder(rv.FASTAI_SEMANTIC_SEGMENTATION) \
            .with_task(task) \
            .with_train_options(
                lr=1e-4,
                batch_size=batch_size,
                num_epochs=num_epochs,
                model_arch='resnet18',
                debug=debug) \
            .build()

        # We will use this function to create a list of scenes that we will pass
        # to the DataSetConfig builder.
        def make_scene(id):
            """Make a SceneConfig object for each image/label pair

            Args:
                id (str): The id that corresponds to both the .tiff image source
                    and .geojson label source for a given scene

            Returns:
                rv.data.SceneConfig: a SceneConfig object which is composed of
                    images, labels and optionally AOIs
            """
            # Find the uri for the image associated with this is
            train_image_uri = os.path.join(raster_uri,
                                           '{}{}.tif'.format(raster_fn_prefix, id))

            # Construct a raster source from an image uri that can be handled by Rasterio.
            # We also specify the order of image channels by their indices and add a
            # stats transformer which normalizes pixel values into uint8.
            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .with_stats_transformer() \
                .build()

            # Next create a label source config to pair with the raster source:
            # define the geojson label source uri
            vector_source = os.path.join(
                label_uri, '{}{}.geojson'.format(label_fn_prefix, id))

            # Since this is a semantic segmentation experiment and the labels
            # are distributed in a vector-based GeoJSON format, we need to rasterize
            # the labels. We create  aRasterSourceConfigBuilder using
            # `rv.RASTERIZED_SOURCE`
            # indicating that it will come from a vector source. We then specify the uri
            # of the vector source and (in the 'with_rasterizer_options' method) the id
            # of the pixel class we would like to use as background.
            label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                .with_vector_source(vector_source) \
                .with_rasterizer_options(2) \
                .build()

            # Create a semantic segmentation label source from rasterized source config
            # that we built in the previous line.
            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_raster_source(label_raster_source) \
                .build()

            # Finally we can build a scene config object using the scene id and the
            # configs we just defined
            scene = rv.SceneConfig.builder() \
                .with_task(task) \
                .with_id(id) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

            return scene

        # Create lists of train and test scene configs
        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        # Construct a DataSet config using the lists of train and
        # validation scenes
        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        # We will need to convert this imagery from uint16 to uint8
        # in order to use it. We specified that this conversion should take place
        # when we built the train raster source but that process will require
        # dataset-level statistics. To get these stats we need to create an
        # analyzer.
        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()

        # We use the previously-constructed configs to create the constituent
        # parts of the experiment. We also give the builder strings that define
        # the experiment id and and root uri. The root uri indicates where all
        # of the output will be written.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        # Return one or more experiment configs to run the experiment(s)
        return experiment


if __name__ == '__main__':
    rv.main()
