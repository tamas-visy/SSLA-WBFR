# Self-Supervised Learning Approaches for Wearable-Based Fatigue Recognition

Based on the [Homekit2020 repository](https://github.com/behavioral-data/Homekit2020). See the [original README.md](ORIGINAL_README.md) and the [original LICENSE.md](ORIGINAL_LICENSE.md).

For the paper [Systematic Evaluation of Self-Supervised Learning Approaches for Wearable-Based Fatigue Recognition]() _(CHIL 2024)_ by Tamás Visy, Rita Kuznetsova, Christian Holz, Shkurta Gashi.

## Setup

Follow the [original README.md](ORIGINAL_README.md) for the initial setup.

### Env

A ".env" file should be added to repository root:
```
# Environment variables go here, can be read by `python-dotenv` package:
#
#   `src/script.py`
#   ----------------------------------------------------------------
#    import dotenv
#
#    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#    dotenv_path = os.path.join(project_dir, '.env')
#    dotenv.load_dotenv(dotenv_path)
#   ----------------------------------------------------------------
#
# DO NOT ADD THIS FILE TO VERSION CONTROL!
PROJECT_NAME=SSLA-WBFR

# Weights and Biases Parameters
WANDB_USERNAME=???
WANDB_PROJECT=???
```

### Patches

The Petastorm package might have to be updated manually to fix a bug related to [this](https://github.com/numpy/numpy/issues/9464):

In `petastorm/utils.py`, around line 77
```python
if field.numpy_dtype == np.int32:
    decoded_row[field_name] = np.int64(row[field_name])
else:
    decoded_row[field_name] = field.numpy_dtype(row[field_name])
```

## Data

To access the data using the commands utilized below, the templates like `configs/data_temporal._yaml` should be filled out in a file with the correct (".yaml") file type.

Some step values (mostly in the minute-level data) are very strong outliers, setting these to `missing` agrees more with the hour-level data.
Set `--data.fix_step_outliers true` to do so, which uses `DefaultFixerTransformRow` to filter out values (below 0 or above 400 mean steps per minute) by setting values to 0 and missingness to 1.

## Train

#### Sample command
If you follow everything in the README, the following "raw" command should work. However, to simplify setting up runs it is recommended to use configs as [described later](#train-using-configs).

```commandline
python src/models/train.py fit --config configs/tasks/HomekitPredictFluPos.yaml --config configs/models/CNNToTransformerClassifier.yaml --early_stopping_patience 2 --model.val_bootstraps 0 --data.fix_step_outliers true --data.train_path $PROJECT_ROOT$/data/processed/split_2020_02_10/train_7_day/ --data.val_path $PROJECT_ROOT$/data/processed/split_2020_02_10/eval_7_day/ --data.test_path $PROJECT_ROOT$/data/processed/split_2020_02_10/test_7_day/
```
where `$PROJECT_ROOT$` has to be the absolute path to project's root folder (starting with a drive, e.g., `"C:/Users/..."`, with right facing slashes so Petastorm / Parquet / etc. works).

While this is not mentioned in the [original README.md](ORIGINAL_README.md), you should also pass a `data.test_path`, otherwise testing could raise errors and the train could crash.

Set `--model.val_bootstraps=0` [to avoid memory leak](https://github.com/behavioral-data/Homekit2020/issues/13).

Added `--early_stopping_patience`, see "Appendix D. Datasheet" of the [Homekit2020] paper.

Diverging from the original command is the addition of `--data.fix_step_outliers true`, which is described in [the previous section](#data).

##### Self-supervised training sample command

MyCNNToTransformerClassifier can be pre-trained on "unlabeled" data by using an appropriate task. The model's `task_type` should be set to an appropriate value - this can't be avoided due to the structure of the legacy code and PyTorch Lightning.

```commandline
python src/models/train.py fit --config configs/tasks/SSL-Autoencode.yaml --config configs/models/MyCNNtoTransformerClassifier.yaml --early_stopping_patience 2 --model.val_bootstraps 0 --model.task_type autoencoder --data.fix_step_outliers true --data.train_path $PROJECT_ROOT$/data/processed/split_2020_02_10/train_7_day/ --data.val_path $PROJECT_ROOT$/data/processed/split_2020_02_10/eval_7_day/ --data.test_path $PROJECT_ROOT$/data/processed/split_2020_02_10/test_7_day/
```

An example task would be `SSL-Autoencode`, where the model's task is to find a dense but representative embedding from which the input can be reconstructed well.
Since this task has a higher memory footprint compared to other SSL methods, it is recommended to decrease batch size to half with `--model.batch_size 400`.

### Train using configs

Most models can be trained using configs using the template
```commandline
python src/models/train.py fit -c configs/models/MyCNNtoTransformerClassifier.yaml -c configs/tasks/$TASK_NAME$.yaml -c configs/data_temporal_7_day.yaml -c configs/common.yaml [--model.pretrained_ckpt_path $PATH$]
```
where task name and checkpoint path have to be set to correct values. Checkpoint path can be excluded for non-pretrained training runs (so all runs except for ones evaluating a model pretrained with an SSL method).

#### Sample commands
##### Fatigue
```commandline
python src/models/train.py fit -c configs/models/MyCNNtoTransformerClassifier.yaml -c configs/tasks/Fatigue.yaml -c configs/data_temporal_7_day.yaml -c configs/common.yaml
```

##### SSL-DailyFeatures
```commandline
python src/models/train.py fit -c configs/models/MyCNNtoTransformerClassifier.yaml -c configs/tasks/SSL-DailyFeatures.yaml -c configs/data_temporal_7_day.yaml -c configs/common.yaml
```

##### SSL-MultiHead
```commandline
python src/models/train.py fit -c configs/models/MyCNNtoTransformerClassifier.yaml -c configs/tasks/SSL-MultiHead.yaml -c configs/data_temporal_7_day.yaml -c configs/common.yaml
```
