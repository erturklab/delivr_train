# DELIVR Training pipeline
This is the official repository for the DELIVR training pipeline.
## Installation
Requirements can be installed from the requirements.txt, Ranger21 has to be installed according to the [Ranger21 github repository](https://github.com/lessw2020/Ranger21).
## Usage
You can run the training pipeline by `python __main__.py [CONFIG_LOCATION]` where `CONFIG_LOCATION` points towards the corresponding configuration file. 
### Configuration file
The configuration file needs to be adapted to your project. This section gives an overview over the configuration options.
- `dataset``raw_path` : The location of your raw data
- `dataset``gt_path` : The location of your annotation data
- `dataset``output_path` : The path where your results will be saved to
- `dataset``checkpoint_path` : DEPRECATED
- `dataset``delivr_model_path` : The path of the model you want to retrain
- `training``epochs`: The amount of epochs you want to train your model
- `training``learning_rate`: The learning rate for your training
- `training``normalization`: Binary, `true` performs intensity based normalization on the raw data, `false` not
- `training``retrain`: Binary, `true` retrains the model under `dataset``delivr_model_path`. `false` not
- `training``tta`: Binary, `true` performs test time augmentation, `false` not
- `training``test_list`: Can be either the path to a dedicated test set or a list of paths pointing to items in your testset. Will be populated automatically if empty
- `network``batch_size`: Batch size
- `network``num_workers`: Number of workers as laid out [here](https://pytorch.org/docs/stable/data.html#multi-process-data-loading)

