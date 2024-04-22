"""
===========================
SeattleFluStudy experiments  
===========================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

This module contains the code used in the experimental tasks described in the reference paper.
The tasks are classes intended to provide, during training and evaluation, the correct reference to 
the datasets being used and evaluation metrics. 

**Classes**
    :class Task:
    :class ClassificationMixin:
    :class AutoencodeMixin:
    :class ActivityTask: 
    :class GeqMeanSteps: 
    :class PredictFluPos: 
    :class PredictEvidationILI: 
    :class PredictTrigger:
    :class PredictSurveyClause:
    :class SingleWindowActivityTask:
    :class ClassifyObese:
    :class EarlyDetection:
    :class AutoencodeEarlyDetection:
    :class Autoencode:

"""
__docformat__ = 'reStructuredText'

from importlib.resources import path
from typing import Any, List, Optional, Tuple, Callable

from pyarrow.parquet import ParquetDataset
from sklearn.utils import resample

import numpy as np

import pytorch_lightning as pl

from petastorm import make_reader
from petastorm.transform import TransformSpec
from petastorm.etl.dataset_metadata import infer_or_load_unischema
import petastorm.predicates as peta_pred
from petastorm.pytorch import DataLoader as PetastormDataLoader, decimal_friendly_collate

from src.models.eval import classification_eval, regression_eval
from src.data.utils import url_from_path
from src.utils import get_logger, read_yaml
from src.tasks.lablers import DailyFeaturesLabler

from src.data.utils import read_parquet_to_pandas

from src.data.transforms import DefaultTransformRow, DefaultFixerTransformRow

logger = get_logger(__name__)

import pandas as pd

DEFAULT_FIELDS = ['heart_rate',
                  'missing_heart_rate',
                  'missing_steps',
                  'sleep_classic_0',
                  'sleep_classic_1',
                  'sleep_classic_2',
                  'sleep_classic_3',
                  'steps']

###################################################
########### MODULE UTILITY FUNCTIONS ##############
###################################################

SUPPORTED_TASK_TYPES = [
    "classification",
    "autoencoder"
    # regression
]


def get_task_from_config_path(path, **kwargs):
    config = read_yaml(path)
    task_class = get_task_with_name(config["data"]["class_path"])
    task = task_class(dataset_args=config["data"].get(["dataset_args"], {}),
                      **config["data"].get("init_args", {}),
                      **kwargs)
    return task


def stack_keys(keys, row, normalize_numerical=True):
    results = []
    for k in keys:
        feature_vector = row[k]
        is_numerical = np.issubdtype(feature_vector.dtype, np.number)

        if normalize_numerical and is_numerical:
            mu = feature_vector.mean()
            sigma = feature_vector.std()
            if sigma != 0:
                feature_vector = (feature_vector - mu) / sigma

        results.append(feature_vector.T)

    return np.vstack(results).T


class Task(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        for task_type in SUPPORTED_TASK_TYPES:
            setattr(self, f"is_{task_type}", False)

        # only computes full dataset if dataset getter methods are invoked
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        raise NotImplementedError

    def get_val_dataset(self):
        raise NotImplementedError

    def get_test_dataset(self):
        raise NotImplementedError

    def get_labler(self):
        return NotImplementedError

    def get_labler(self):
        return NotImplementedError

    def get_metadata_lablers(self):
        return {}

    def get_metadata_types(self):
        return []


class TaskTypeMixin:
    def __init__(self):
        self.is_regression = False
        self.is_classification = False
        self.is_autoencoder = False
        self.is_double_encoding = False


class ClassificationMixin(TaskTypeMixin):
    def __init__(self):
        TaskTypeMixin.__init__(self)
        self.is_classification = True

    def evaluate_results(self, logits, labels, threshold=0.5):
        return classification_eval(logits, labels, threshold=threshold)

    def get_huggingface_metrics(self, threshold=0.5):
        def evaluator(pred):
            labels = pred.label_ids
            logits = pred.predictions
            return self.evaluate_results(logits, labels, threshold=threshold)

        return evaluator


class RegressionMixin(TaskTypeMixin):
    def __init__(self):
        TaskTypeMixin.__init__(self)
        self.is_regression = True

    def evaluate_results(self, preds, labels):
        return regression_eval(preds, labels)

    def get_huggingface_metrics(self):
        def evaluator(predictions):
            labels = predictions.label_ids
            preds = predictions.predictions
            return self.evaluate_results(preds, labels)

        return evaluator


class AutoencodeMixin(RegressionMixin):
    def __init__(self):
        self.is_autoencoder = True
        super(AutoencodeMixin, self).__init__()


def verify_backend(backend,
                   limit_train_frac,
                   data_location,
                   datareader_ray_obj_ref,
                   activity_level):
    if backend == "petastorm":
        if activity_level == "day":
            raise NotImplementedError("Day-level data is not yet supported with petastorm")

        if limit_train_frac:
            raise NotImplementedError("Petastorm relies on pre-processed data, so limit_train_frac can't be used yet")

        if data_location:
            raise NotImplementedError("With Petastorm please use --train_path and --val_path")

        if datareader_ray_obj_ref:
            raise NotImplementedError("Petastorm backend does not support ray references")


class SubsamplingReaderWrapper:
    def __init__(self, reader, prob):
        self._reader = reader
        self.fraction = prob
        prob_as_100xdecimal = round(self.fraction * 100)
        print("Resolution of subsampling is 0.01")
        assert 1 <= prob_as_100xdecimal <= 99
        # probability has to be a decimal with a single digit. It is represented by an integer (between 1 and 9)
        self._prob = prob_as_100xdecimal
        self.read = 0
        self.returned = 0

    @property
    def batched_output(self):
        return self._reader.batched_output

    def stop(self):
        return self._reader.stop()

    def join(self):
        return self._reader.join()

    def reset(self):
        print(self.returned, self.read, f"{self.returned/self.read:.4f}", f"{self._prob}%")
        self.read = 0
        self.returned = 0
        return self._reader.reset()

    def __iter__(self):
        return self

    def __next__(self):
        x = self._reader.next()
        h = hash(x.id)
        self.read += 1
        while h % 100 >= self._prob:
            x = self._reader.next()
            h = hash(x.id)
            self.read += 1
        self.returned += 1
        return x


class ActivityTask(Task):
    # """Base class for tasks in this project"""
    # def __init__(self,train_path: Optional[str] = None):
    #     self.train_path = train_path

    def __init__(self, fields: List[str] = None,
                 train_path: Optional[str] = None,
                 val_path: Optional[str] = None,
                 test_path: Optional[str] = None,
                 *,
                 downsample_negative_frac: Optional[float] = None,
                 shape: Optional[Tuple[int, ...]] = None,
                 normalize_numerical: bool = True,
                 fix_step_outliers: bool = False,
                 append_daily_features: bool = False,
                 daily_features_path: Optional[str] = None,
                 backend: str = "petastorm",
                 batch_size: int = 800,
                 activity_level: str = "minute",
                 row_transform: Optional[Callable] = None,
                 combine_rows: Optional[int] = None,
                 readable_name=None,
                 subsample_training_data=None):

        # TODO does not currently support day level data
        super(ActivityTask, self).__init__()
        self.fields = fields
        self.batch_size = batch_size
        self.backend = backend
        self.normalize_numerical = normalize_numerical
        self.fix_step_outliers = fix_step_outliers
        self.combine_rows = combine_rows
        self.readable_name = readable_name
        self.subsample_training_data = subsample_training_data

        self.daily_features_appended = append_daily_features
        if self.daily_features_appended:
            self.daily_features_labler = DailyFeaturesLabler(data_location=daily_features_path, window_size=1)
        else:
            self.daily_features_labler = None

        ### Newer backend relies on petastorm and is faster, but requires more pre-processing:
        if self.backend == "petastorm":
            """
            Set the necessary attributes and adjust the time window of data  
            """
            # TODO make sure labler gets label for right day
            # TODO ensure labler is serialized properly

            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path

            self.train_url = url_from_path(train_path)
            self.val_url = url_from_path(val_path)
            self.test_url = url_from_path(test_path)

            labler = self.get_labler()

            if downsample_negative_frac:
                if not hasattr(labler, "get_positive_keys"):
                    raise ValueError(f"Tried to downsample negatives but {type(labler)}"
                                     " does not support `get_positive_keys`")
                positive_keys = labler.get_positive_keys()
                has_positive_predicate = peta_pred.in_lambda(["participant_id", "end"],
                                                             lambda x, y: (x, pd.to_datetime(y)) in positive_keys)
                in_subset_predicate = peta_pred.in_pseudorandom_split(
                    [downsample_negative_frac, 1 - downsample_negative_frac],
                    0, "id")
                self.predicate = peta_pred.in_reduce([has_positive_predicate, in_subset_predicate], any)
            else:
                self.predicate = None

            infer_schema_path = None
            for path in [self.train_path, self.val_path, self.test_path]:
                if path:
                    infer_schema_path = path
                    break

            if not infer_schema_path:
                raise ValueError("Must provide at least one of train_path, val_path, or test_path")

            self.schema = infer_or_load_unischema(ParquetDataset(infer_schema_path, validate_schema=False))
            numerical_fields = [k for k, v in self.schema.fields.items() if np.issubdtype(v.numpy_dtype, np.number)]

            # Try to infer the shape of the data 
            # TODO: Really, really don't like how many guesses we make here. There
            # are two issues:
            #   1) We allow the user to provide field names, but then entirely ignore
            #      them if they're missing from the schema, which is confusing. 
            #      I think that rather than providing all field names, we should ask
            #      for schema fields that are to be used as keys for the labler,
            #      and fields that should be ignored (e.g. "id")
            #   2) Input length feels sloppy. We should be able to infer this from the schema
            lengths = set()
            missing_fields = [x for x in self.fields if not x in self.schema.fields.keys()]

            if not missing_fields:
                for k in self.fields:
                    lengths.add(getattr(self.schema, k).shape[-1])

            else:
                logger.warning(f"""Missing fields {missing_fields} in schema {self.schema.fields.keys()}
                                   Will attempt to infer data shape from numerical fields""")
                self.fields = [x for x in numerical_fields if not x in ["id", "__index_level_0__"]]
                for k in self.fields:
                    shape = getattr(self.schema, k).shape[-1]
                    if shape:
                        lengths.add(shape)

            if len(lengths) > 1:
                raise ValueError("Provided fields have mismatched feature sizes")
            if len(lengths) == 0:
                logger.warning(f"Could not infer data shape from schema, assuming ({len(numerical_fields)},)")
            else:
                data_length = lengths.pop()

            self.data_shape = (int(data_length), len(self.fields))

        elif backend == "dynamic":
            self.data_shape = shape

        self.save_hyperparameters()

    @property
    def name(self):
        if self.readable_name is not None:
            return self.readable_name
        return self.__class__.__name__

    def get_description(self):
        return self.__doc__

    def get_transform_spec(self, *, test=False):
        """To get 'real' results (e.g., for testing), pass test=True
         which should disable any modifications from RowTransforms"""
        try:
            row_transform = self.trainer.model.row_transform
        except AttributeError:
            row_transform = DefaultFixerTransformRow(self,
                                                     fix_step_outliers=self.fix_step_outliers,
                                                     normalize_numerical=self.normalize_numerical,
                                                     test=test)

        removed_fields = row_transform.get_removed_fields()
        new_fields = row_transform.get_new_fields()
        return TransformSpec(row_transform, removed_fields=removed_fields,
                             edit_fields=new_fields)

    def train_dataloader(self):
        if self.train_url:
            reader = make_reader(self.train_url, transform_spec=self.get_transform_spec(),
                                 predicate=self.predicate)
            if self.subsample_training_data is not None:
                reader = SubsamplingReaderWrapper(reader, self.subsample_training_data)
                print(f"Subsampling training data to {reader.fraction}")
            return PetastormDataLoader(reader,
                                       collate_fn=self.collate_fn,
                                       batch_size=self.batch_size)

    def val_dataloader(self):
        if self.val_url:
            return PetastormDataLoader(make_reader(self.val_url, transform_spec=self.get_transform_spec(),
                                                   predicate=self.predicate),
                                       collate_fn=self.collate_fn,
                                       batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test_url:
            return PetastormDataLoader(make_reader(self.test_url, transform_spec=self.get_transform_spec(test=True),
                                                   predicate=self.predicate),
                                       collate_fn=self.collate_fn,
                                       batch_size=self.batch_size)

    def do_combine_rows(self, rows):
        combined_row = {}
        if len(rows) != 2:
            raise NotImplementedError
        row1, row2 = rows
        for key in row1.keys():
            if key not in ("inputs_embeds", "label"):
                combined_row[f"{key}_L"] = row1[key]
                combined_row[f"{key}_R"] = row2[key]
        # TODO embed along extra dimension
        combined_row["inputs_embeds"] = np.hstack([row1["inputs_embeds"], row2["inputs_embeds"]])
        combined_row["label"] = self.get_labler()(embedded_row1=row1, embedded_row2=row2)
        return combined_row

    def collate_fn(self, batch):
        if self.combine_rows:  # is not None and != 0
            assert self.combine_rows >= 1 and isinstance(self.combine_rows, int)
            try:
                assert len(batch) % self.combine_rows == 0, f"Batch can not be partitioned by {self.combine_rows}"
            except AssertionError as ae:
                print(f"Dropping final {len(batch) % self.combine_rows} row(s):", ae)
                batch = batch[:-(len(batch) % self.combine_rows)]
            new_batch = []
            for i in range(0, len(batch), self.combine_rows):
                combined_row = self.do_combine_rows(batch[i:i + self.combine_rows])
                new_batch.append(combined_row)
            batch = new_batch
        return decimal_friendly_collate(batch)  # default collate fn for PetaStorm

    def get_train_dataset(self):
        if self.train_dataset is None:
            # we only process the full training dataset once if this method is called
            self.train_dataset = self.format_dataset(self.train_path, self.get_labler())

        return self.train_dataset

    def get_val_dataset(self):
        if self.val_dataset is None:
            # we only process the full validation dataset once if this method is called
            self.val_dataset = self.format_dataset(self.val_path, self.get_labler())

        return self.val_dataset

    def get_test_dataset(self):
        if self.test_dataset is None:
            # we only process the full testing dataset once if this method is called
            self.test_dataset = self.format_dataset(self.test_path, self.get_labler())

        return self.test_dataset

    def format_dataset(self, data_path, labler):
        dataset = read_parquet_to_pandas(data_path)
        x = np.array(dataset[self.fields].values.tolist()).reshape(len(dataset), -1)
        y = dataset.apply(lambda x: labler(x["participant_id"], x["start"], x["end"]), axis=1)
        return (dataset["participant_id"], dataset["start"], x, y)

    # def add_task_specific_args(parent_parser):
    #         parser = parent_parser.add_argument_group("Task")
    #         return parent_parser
