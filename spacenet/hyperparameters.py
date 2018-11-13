import os

import rastervision as rv
from spacenet.vegas import (SpacenetConfig, build_dataset, build_task,
                            str_to_bool, validate_options, BUILDINGS)


def build_backend(task, test, learning_rate):
    if test:
        debug = True
    else:
        debug = False

    if not test:
        batch_size = 12
        num_steps = 1e5
    else:
        num_steps = 1
        batch_size = 1

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
    def exp_main(self, root_uri, use_remote_data=True, test=False, learning_rates='0.001'):
        """Run an experiment on the Spacenet Vegas building dataset.

        Args:
            root_uri: (str): root of where to put output
            use_remote_data: (bool or str) if True or 'True', then use
                data from S3, else local
            test: (bool or str) if True or 'True', run a very small
                experiment as a test and generate debug output

        """
        target = BUILDINGS
        task_type = rv.SEMANTIC_SEGMENTATION
        learning_rates = learning_rates.split(',')

        test = str_to_bool(test)
        task_type = task_type.upper()
        use_remote_data = str_to_bool(use_remote_data)
        spacenet_config = SpacenetConfig.create(use_remote_data, target)
        ac_key = '{}_{}'.format(target, task_type.lower())

        validate_options(task_type, target)

        task = build_task(task_type, spacenet_config.get_class_map())
        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()
        dataset = build_dataset(task, spacenet_config, test)

        # Reduce number of scenes
        dataset.train_scenes = dataset.train_scenes[0:2**7]

        exps = []
        for learning_rate in learning_rates:
            backend = build_backend(task, test, learning_rate)
            experiment_id = '{}_{}_rate={}'.format(target, task_type.lower(),
                                                   learning_rate)

            # Need to use stats_analyzer because imagery is uint16.
            experiment = rv.ExperimentConfig.builder() \
                                            .with_id(experiment_id) \
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
