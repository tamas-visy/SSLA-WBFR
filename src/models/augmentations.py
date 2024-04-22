from typing import Tuple, List

import torch
from torch import nn, Tensor

import random

try:
    from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs
except ModuleNotFoundError:
    def error(*args, **kwargs):
        raise NotImplementedError


    NaturalCubicSpline = error
    natural_cubic_spline_coeffs = error


# see tasks.py with DEFAULT_FIELDS


class ExactRelatedDropout1D(nn.Module):
    """Drops out exactly one set of "related" channels specified in targets."""

    # call with [[0, 1], [2, 7], [3, 4, 5, 6]] - see DEFAULT_FIELDS
    def __init__(self, targets: List[List[int]], modify_mask=False):
        super().__init__()
        self.targets = targets
        self.modify_mask = modify_mask

    def forward(self, x, *, mask=None):
        """Tensor, and optionally mask as Tensor; if given, mask is modified in place"""
        x, y = x, mask
        if not self.modify_mask:
            y = None  # set y (or mask) back to None
        masks = []
        for cols in self.targets:
            mask = torch.zeros(x.size(2), dtype=torch.bool)
            mask[cols] = True
            masks.append(mask)

        mask_tensor = torch.stack(masks)
        # Random mask for batch elements
        random_indices = torch.randint(0, len(self.targets), (x.size(0),))
        selected_masks = mask_tensor[random_indices]
        # Broadcasting into correct shape
        mask = selected_masks.unsqueeze(1).expand(-1, x.size(1), -1)

        x_copy = x.clone()
        x_copy[mask] = 0  # for each element in the batch, drop a random (list of columns from targets)
        if y is not None:
            y[mask] = 0
        return x_copy

    def __str__(self):
        return f'{self.__class__.__name__}(targets={self.targets}, modify_mask={self.modify_mask})'


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        gaussian = torch.randn(x.size(), device=x.device) * self.std + self.mean
        return x + gaussian

    def __str__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class MaskedGaussianNoise(GaussianNoise):
    """Adds Gaussian noise to selected channels"""

    def __init__(self, targets: List[int], **kwargs):
        super().__init__(**kwargs)
        self.targets = targets

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        gaussian = torch.zeros_like(x)
        gaussian[:, :, self.targets] = torch.randn(gaussian[:, :, self.targets].size(),
                                                   device=x.device) * self.std + self.mean
        return x + gaussian

    def __str__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, targets={self.targets})'


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return torch.nn.functional.dropout(input=x, p=self.p)

    def __str__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class TimeWarp(nn.Module):
    """
    Is very slow.

    Refer to https://github.com/uchidalab/time_series_augmentation.
    Also pip install git+https://github.com/patrick-kidger/torchcubicspline.git
    """

    def __init__(self, sigma=0.2, knot=4):
        super().__init__()
        self.sigma = sigma
        self.knot = knot

    @staticmethod
    def _linear_interp(x, y, x_new):
        # Is buggy
        x_new = x_new.clamp(min=x.min(), max=x.max())

        idx = torch.searchsorted(x, x_new)
        idx = torch.clamp(idx, 1, len(x) - 1)

        x0 = x[idx - 1]
        x1 = x[idx]
        y0 = y[idx - 1]
        y1 = y[idx]

        slope = (y1 - y0) / (x1 - x0)

        return y0 + slope * (x_new - x0)

    def forward(self, x: Tensor, *, mask: Tensor) -> Tensor:
        if mask is not None:
            raise NotImplementedError

        if x.dim() == 2:
            x = x.unsqueeze(0)

        orig_steps = torch.linspace(0, 1, steps=x.size(1), device=x.device)
        random_warps = torch.normal(mean=1.0, std=self.sigma, size=(x.size(0), self.knot + 2, x.size(2)),
                                    device=x.device)
        warp_steps = torch.linspace(0, 1, steps=self.knot + 2,
                                    device=x.device).repeat(x.size(2), 1).transpose(0, 1)

        ret = torch.zeros_like(x)
        for i in range(x.size(0)):
            for dim in range(x.size(2)):
                y = warp_steps[:, dim] * random_warps[i, :, dim]
                coeffs = natural_cubic_spline_coeffs(warp_steps[:, dim], y.unsqueeze(1))
                spline = NaturalCubicSpline(coeffs)
                time_warp_ = spline.evaluate(orig_steps).squeeze(1)

                scale = (x.size(1) - 1) / time_warp_[-1]
                scaled_time_warp = torch.clamp(scale * time_warp_, 0, x.size(1) - 1)

                ret[i, :, dim] = self._linear_interp(orig_steps, x[i, :, dim], scaled_time_warp)
        return ret.squeeze(0)


class PairReorder(nn.Module):
    # Does permutation

    def __init__(self, split_into=7):
        super().__init__()
        self.split_into = split_into

    def forward(self, x: Tensor, mask=None, **kwargs) -> Tensor:
        # TODO handle mask
        # if mask is not None:
        #     raise NotImplementedError
        # assert x.shape[1] == 7 * 24 * 60, "Data is not one week long"

        xs = list(torch.split(x, [x.shape[1] // self.split_into] * self.split_into, dim=1))

        flip_index = torch.randint(0, self.split_into - 1, size=(1,)).item()
        xs[flip_index], xs[flip_index + 1] = xs[flip_index + 1], xs[flip_index]

        x = torch.cat(xs, dim=1)
        return x


class RandomReorder(nn.Module):
    # Does permutation

    def __init__(self, split_into=7):
        super().__init__()
        self.split_into = split_into

    def forward(self, x: Tensor, mask=None, **kwargs) -> Tensor:
        if mask is not None:
            raise NotImplementedError
        assert x.shape[1] == 7 * 24 * 60, "Data is not one week long"

        xs = torch.split(x, [x.shape[1] // self.split_into] * self.split_into, dim=1)
        shuffled_indices = torch.randperm(self.split_into)

        x = torch.cat([xs[i] for i in shuffled_indices], dim=1)
        return x


class Identity(nn.Identity):
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return super().forward(input=x)


class Random(nn.Module):
    def __init__(self, options: List[nn.Module]):
        super().__init__()
        self.options = options

    def forward(self, x: Tensor, *, mask: Tensor) -> Tensor:
        aug = random.choice(self.options)
        return aug.forward(x, mask=mask)

    def __str__(self):
        return f'{self.__class__.__name__}(options={str(self.options)})'


class ExampleAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x: Tensor, *, mask: Tensor) -> Tensor:
        ...
