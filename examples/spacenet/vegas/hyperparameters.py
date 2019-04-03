import os

import rastervision as rv
from examples.spacenet.vegas.all import (
    SpacenetConfig, build_dataset, build_task, validate_options, BUILDINGS)
from examples.utils import str_to_bool


def build_backend(task, test_run, learning_rate):
    if test_run:
        debug = True
    else:
        debug = False

    if test_run:
        num_steps = 1
        batch_size = 1
    else:
        batch_size = 12
        num_steps = 1e5

    rate_dict = {'baseLearningRate': str(learning_rate)}
    backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                              .with_task(task) \
                              .with_model_defaults(rv.MOBILENET_V2) \
                              .with_num_steps(num_steps) \
                              .with_batch_size(batch_size) \
                              .with_config(rate_dict) \
                              .with_debug(debug) \
                              .build()

    return backend


class HyperParameterSearch(rv.ExperimentSet):
    def exp_main(self, raw_uri, root_uri, test_run=False, learning_rates='0.001'):
        """Run a hyper-parameter search experiment on Spacenet Vegas.

        Generates an experiment for each learning rate using a TF Deeplab semantic
        segmentation backend on the Spacenet Vegas Buildings dataset.

        Args:
            raw_uri: (str) directory of raw data (the root of the Spacenet dataset)
            root_uri: (str) root directory for experiment output
            test_run: (bool) if True, run a very small experiment as a test and
                generate debug output
            learning_rates: (str) comma-delimited list of learning rates to use
        """
        test_run = str_to_bool(test_run)
        target = BUILDINGS
        task_type = rv.SEMANTIC_SEGMENTATION
        learning_rates = learning_rates.split(',')

        task_type = task_type.upper()
        spacenet_config = SpacenetConfig.create(raw_uri, target)
        ac_key = '{}_{}'.format(target, task_type.lower())

        validate_options(task_type, target)

        task = build_task(task_type, spacenet_config.get_class_map())
        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()
        dataset = build_dataset(task, spacenet_config, test_run)

        # Reduce number of scenes
        dataset.train_scenes = dataset.train_scenes[0:2**7]

        exps = []
        for learning_rate in learning_rates:
            backend = build_backend(task, test_run, learning_rate)
            exp_id = '{}_{}_rate={}'.format(target, task_type.lower(),
                                            learning_rate)

            # Need to use stats_analyzer because imagery is uint16.
            # Set the analyze and chip key to share analyze and chip output
            # between the experiments.
            experiment = rv.ExperimentConfig.builder() \
                                            .with_id(exp_id) \
                                            .with_task(task) \
                                            .with_backend(backend) \
                                            .with_analyzer(analyzer) \
                                            .with_dataset(dataset) \
                                            .with_root_uri(root_uri) \
                                            .with_analyze_key(ac_key) \
                                            .with_chip_key(ac_key)

            exps.append(experiment.build())

        return exps


if __name__ == '__main__':
    rv.main()
