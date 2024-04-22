from typing import Tuple, List, Dict

from torch import nn, Tensor


class Masker(nn.Module):
    def forward(self, *, mask, x):
        """Takes the mask and the input, then returns a modified copy of the mask."""
        raise NotImplementedError


class MaskColumns(Masker):
    """
    Masks out certain channels (columns) of the data for loss calculation.
    """

    def __init__(self, targets: List[int]):
        super().__init__()
        self.targets = targets

    def forward(self, *, mask, x):
        """Takes the mask and the input, then returns a modified copy of the mask."""
        mask_copy = mask.clone()
        mask_copy[:, :, self.targets] = 0
        return mask_copy

    def __str__(self):
        return f'{self.__class__.__name__}(targets={self.targets})'


class MaskWithInput(Masker):
    """Masks out values in the target channel based on values of the source channels"""
    # TODO triple check how this and Dropout interact

    def __init__(self, actions: List[Dict[str, int]]):
        super().__init__()
        self.actions = [(action['source'], action['target']) for action in actions]

    def forward(self, *, mask, x):
        """Takes the mask and the input, then returns a modified copy of the mask."""
        # The input may have been modified by augments before passing to this method
        # but that is fine, if we
        mask_copy = mask.clone()
        for source, target in self.actions:
            # mask out values in target column where source column of input evaluates to True
            mask_copy[:, :, target][x[:, :, source].bool()] = 0
        return mask_copy

    def __str__(self):
        return f'{self.__class__.__name__}(actions={self.actions})'
