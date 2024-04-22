################################################
########### TASKS IMPLEMENTATIONS ##############
################################################
import os
from typing import List, Optional

from src.data.utils import load_processed_table
from src.tasks.lablers import DailyFeaturesLabler, FluPosLabler, CovidSignalsLabler, FluPosWeakLabler, DayOfWeekLabler, \
    CovidLabler, ClauseLabler, AudereObeseLabler
from src.tasks.tasks import ActivityTask, RegressionMixin, DEFAULT_FIELDS, ClassificationMixin


class PredictDailyFeatures(ActivityTask, RegressionMixin):
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self, fields: List[str] = DEFAULT_FIELDS,
                 activity_level: str = "minute",
                 window_size: int = 7,
                 **kwargs):
        assert not kwargs['train_path'].endswith("day"), "rolling window type dataset is unexpected"
        self.labler = DailyFeaturesLabler(window_size=window_size)
        self.fields = fields

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        RegressionMixin.__init__(self)

    def get_labler(self):
        return self.labler


class PredictFluPos(ActivityTask):
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""
    is_classification = True

    def __init__(self, fields: List[str] = DEFAULT_FIELDS, activity_level: str = "minute",
                 window_onset_max: int = 0, window_onset_min: int = 0,
                 **kwargs):
        self.is_classification = True
        self.labler = FluPosLabler(window_onset_max=window_onset_max,
                                   window_onset_min=window_onset_min)

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        # ClassificationMixin.__init__(self)

    def get_labler(self):
        return self.labler


class PredictCovidSignalsPositivity(ActivityTask):
    is_classification = True

    def __init__(self, fields: List[str] = DEFAULT_FIELDS,
                 activity_level: str = "minute",
                 window_onset_min: int = 0,
                 window_onset_max: int = 0,
                 **kwargs):

        self.is_classification = True

        self.window_onset_min = window_onset_min
        self.window_onset_max = window_onset_max

        self.labler = CovidSignalsLabler(window_onset_min=self.window_onset_min,
                                         window_onset_max=self.window_onset_max)
        if fields:
            self.keys = fields
        else:
            self.keys = ['heart_rate',
                         'missing_heart_rate',
                         'missing_steps',
                         'sleep_classic_0',
                         'sleep_classic_1',
                         'sleep_classic_2',
                         'sleep_classic_3',
                         'steps']

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        # ClassificationMixin.__init__(self)

    @property
    def name(self):
        if self.readable_name is not None:
            return self.readable_name
        return f"{self.__class__.__name__}-{self.window_onset_min}-{self.window_onset_max}"

    def get_labler(self):
        return self.labler


class PredictFluPos(ActivityTask):
    """ Predict whether a participant was positive
        given a rolling window of minute level activity data.

        Note that this class should be deprecated in favor of the
        PredictPositivity task.
    """
    is_classification = True

    def __init__(self, fields: List[str] = DEFAULT_FIELDS, activity_level: str = "minute",
                 window_onset_max: int = 0, window_onset_min: int = 0,
                 **kwargs):
        self.is_classification = True
        self.labler = FluPosLabler(window_onset_max=window_onset_max,
                                   window_onset_min=window_onset_min)

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        # ClassificationMixin.__init__(self)

    def get_labler(self):
        return self.labler


class PredictFluPosWeak(ActivityTask):
    """ Predict whether a participant was positive
        given a rolling window of minute level activity data.
        Note that this class should be deprecated in favor of the
        PredictPositivity task.
    """
    is_classification = True

    def __init__(self, fields: List[str] = DEFAULT_FIELDS, activity_level: str = "minute",
                 window_onset_max: int = 0, window_onset_min: int = 0, survey_path: Optional[str] = None,
                 **kwargs):
        self.is_classification = True
        self.survey_responses = load_processed_table("daily_surveys_onehot", path=survey_path).set_index(
            "participant_id")

        self.labler = FluPosLabler(window_onset_max=window_onset_max, window_onset_min=window_onset_min)

        self.weak_labler = FluPosWeakLabler(survey_responses=self.survey_responses,
                                            window_onset_max=window_onset_max, window_onset_min=window_onset_min)

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)

    def get_labler(self):
        return self.labler

    def get_metadata_lablers(self):
        return {"weak_label": self.weak_labler}

    def get_metadata_types(self):
        return [float]


class PredictWeekend(ActivityTask, ClassificationMixin):
    """Predict whether the associated data belongs to a
       weekend"""

    def __init__(self, fields: List[str] = DEFAULT_FIELDS,
                 activity_level: str = "minute",
                 **kwargs):
        self.labler = DayOfWeekLabler([5, 6])
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)

    def get_labler(self):
        return self.labler


class PredictCovidSmall(ActivityTask, ClassificationMixin):
    """Predict whether a participant was diagnosed with
    covid on the final day of the window

    This was designed for data from Mirsha et al.,
    and uses the processed results from
    /projects/bdata/datasets/covid-fitbit/processed/covid_dates.csv
    """

    def __init__(self, dates_path: str,
                 fields: List[str] = DEFAULT_FIELDS,
                 activity_level: str = "minute",
                 **kwargs):
        self.dates_path = dates_path
        self.filename = os.path.basename(dates_path)

        self.labler = CovidLabler(dates_path)
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)

    def get_labler(self):
        return self.labler


class PredictSurveyClause(ActivityTask, ClassificationMixin):
    """Predict whether a clause in the onehot
       encoded surveys is true for a given day.

       For a sense of what kind of logical clauses are
       supported, check out:

       https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html"""

    def __init__(self, clause: str,
                 activity_level: str = "minute",
                 fields: List[str] = DEFAULT_FIELDS,
                 survey_path: Optional[str] = None,
                 **kwargs):
        self.clause = clause
        self.survey_responses = load_processed_table("daily_surveys_onehot", path=survey_path).set_index(
            "participant_id")
        self.labler = ClauseLabler(self.survey_responses, self.clause)
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)

    def get_labler(self):
        return self.labler

    @property
    def name(self):
        if self.readable_name is not None:
            return self.readable_name
        return f"{self.__class__.__name__}-{self.clause}"

    def get_description(self):
        return self.__doc__


class ClassifyObese(ActivityTask, ClassificationMixin):
    def __init__(self, activity_level: str = "minute",
                 fields: List[str] = DEFAULT_FIELDS,
                 **kwargs):
        self.labler = AudereObeseLabler()
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)

    def get_labler(self):
        return self.labler
