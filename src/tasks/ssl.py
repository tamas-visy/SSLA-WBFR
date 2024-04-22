################################################
############ SSL IMPLEMENTATIONS ###############
################################################
# from https://github.com/behavioral-data/FluStudy/blob/master/src/models/tasks.py#L543

from typing import List

from src.tasks import PredictDailyFeatures
from src.tasks.lablers import SameParticipantLabler, ConstantLabler
from src.tasks.tasks import ActivityTask, DEFAULT_FIELDS, ClassificationMixin, AutoencodeMixin


class Autoencode(AutoencodeMixin, ActivityTask):
    """Autoencode minute level data"""

    def __init__(self, clause: str = None,
                 activity_level: str = "minute",
                 fields: List[str] = DEFAULT_FIELDS,
                 **kwargs):
        self.clause = clause

        # ActivityTask.__init__(self,  td.AutoencodeDataset, dataset_args=dataset_args, **kwargs)
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)
        AutoencodeMixin.__init__(self)
        self.is_autoencoder = True

    def get_description(self):
        return self.__doc__

    def get_name(self):
        return self.__class__.__name__


class PredictSameParticipant(ActivityTask, ClassificationMixin):
    def __init__(self, activity_level="minute", fields=DEFAULT_FIELDS, **kwargs):
        self.labler = SameParticipantLabler()
        ActivityTask.__init__(self, fields=fields,
                              activity_level=activity_level, combine_rows=2,
                              **kwargs)

        ClassificationMixin.__init__(self)
        self.is_double_encoding = True

    def get_labler(self):
        return self.labler


class Triplet(ActivityTask):
    def __init__(self, activity_level="minute", fields=DEFAULT_FIELDS, **kwargs):
        self.labler = ConstantLabler()
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, combine_rows=2, **kwargs)

    def get_labler(self):
        return self.labler


class Contrastive(ActivityTask):
    def __init__(self, activity_level="minute", fields=DEFAULT_FIELDS, **kwargs):
        self.labler = ConstantLabler()
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level, **kwargs)

    def get_labler(self):
        return self.labler


class MultiHeadTriplet(Triplet):
    pass


class MultiHeadDailyFeatures(PredictDailyFeatures):
    pass
