from typing import Optional, List

import torch

from src.models.eval import TorchMetricClassification, TorchMetricRegression
from src.models.losses import build_loss_fn
from src.models.models import modules
from src.models.models.bases import SensingModel


class MyCNNToTransformerClassifier(SensingModel):
    """This model and related configs might assume DEFAULT_FIELDS
    for column names see DEFAULT_FIELDS in tasks.py"""

    def __init__(self, *, num_attention_heads: int = 4, num_hidden_layers: int = 4, num_labels=2,
                 kernel_sizes=None, out_channels=None, stride_sizes=None, dropout_rate=0.4,
                 positional_encoding=False, pretrained_ckpt_path: Optional[str] = None,
                 loss_fn="CrossEntropyLoss", task_type="classification",
                 multihead_mode: Optional[str] = None,
                 combine_rows: Optional[int] = None,
                 pos_class_weight=1, neg_class_weight=1,
                 augmentations: Optional[List[torch.nn.Module]] = None,
                 maskers: Optional[List[torch.nn.Module]] = None,
                 batch_norm: Optional[bool] = False,
                 denoising: Optional[bool] = False,
                 **kwargs) -> None:
        # ModelTypeMixin
        self.is_regressor = False
        self.is_classifier = False
        self.is_autoencoder = False
        self.combine_rows = combine_rows
        if self.combine_rows is not None and self.combine_rows != 2:
            raise NotImplementedError

        self.is_classifier = task_type == "classification"
        self.is_autoencoder = task_type == "autoencoder"
        self.is_regressor = task_type == "regression"
        self.is_triplet = task_type == "triplet"
        self.is_contrastive = task_type == "contrastive"
        self.is_multihead = task_type == "multihead"
        self.multihead_mode = multihead_mode
        metric_class = TorchMetricClassification if self.is_classifier else TorchMetricRegression
        super().__init__(metric_class=metric_class, **kwargs)

        self.denoising = denoising

        if stride_sizes is None:
            stride_sizes = [2, 2, 2]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 1]
        if out_channels is None:
            out_channels = [256, 128, 64]
        if num_hidden_layers == 0:
            self.name = "MyCNNClassifier"
        else:
            self.name = "MyCNNToTransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")

        self.augmentations = augmentations if augmentations is not None else []
        self.maskers = maskers if maskers is not None else []

        self.criterion = build_loss_fn(loss_fn=loss_fn, task_type=task_type if not self.is_multihead else "autoencoder",
                                       pos_class_weight=pos_class_weight,
                                       neg_class_weight=neg_class_weight)
        self._second_criterion = None
        if self.is_multihead:
            assert self.multihead_mode is not None
            self._second_criterion = build_loss_fn(loss_fn=loss_fn, task_type=self.multihead_mode,
                                                   pos_class_weight=pos_class_weight,
                                                   neg_class_weight=neg_class_weight)

        print(f"{self.__class__.__name__}(out_channels={out_channels}, ...) with {self.criterion.__class__.__name__}"
              f"\n\trunning on {task_type}"
              f"\n\t{' + ' + ', '.join([str(x) for x in self.augmentations]) if self.augmentations is not None else ''}"
              f"\n\t{' + ' + ', '.join([str(x) for x in self.maskers]) if self.maskers is not None else ''}")

        self.encoder = modules.CNNToTransformerEncoder(input_features, num_attention_heads, num_hidden_layers,
                                                       n_timesteps, kernel_sizes=kernel_sizes,
                                                       out_channels=out_channels,
                                                       stride_sizes=stride_sizes, dropout_rate=dropout_rate,
                                                       num_labels=num_labels,
                                                       positional_encoding=positional_encoding)
        self._second_head = None
        if self.is_classifier:
            # TODO two layer?
            self.head = modules.ClassificationModule(self.encoder.d_model * (2 if self.combine_rows else 1),
                                                     self.encoder.final_length, num_labels)
        elif self.is_autoencoder:
            self.head = modules.CNNDecoder.from_inverse_of_encoder(self.encoder.input_embedding)
        elif self.is_regressor:
            self.head = modules.ClassificationModule(self.encoder.d_model * (2 if self.combine_rows else 1),
                                                     self.encoder.final_length, num_labels)
        elif self.is_contrastive:
            self.head = modules.ClassificationModule(self.encoder.d_model * (2 if self.combine_rows else 1),
                                                     self.encoder.final_length, num_labels)
        elif self.is_triplet:
            assert self.combine_rows is not None
            self.head = modules.ClassificationModule(self.encoder.d_model,
                                                     self.encoder.final_length, num_labels)
        elif self.is_multihead:
            # copy of self.is_autoencoder
            self.head = modules.CNNDecoder.from_inverse_of_encoder(self.encoder.input_embedding)
            if self.multihead_mode == 'triplet':
                # copy of self.is_triplet
                assert self.combine_rows is not None
                self._second_head = modules.ClassificationModule(self.encoder.d_model,
                                                                 self.encoder.final_length, num_labels)
            elif self.multihead_mode == 'regression':
                # copy of self.is_regressor
                # TODO two layer?
                self._second_head = modules.ClassificationModule(self.encoder.d_model * (2 if self.combine_rows else 1),
                                                                 self.encoder.final_length, num_labels)

        else:
            raise ValueError

        self.norm = None
        if batch_norm:
            self._norm = torch.nn.BatchNorm1d(num_features=input_features)
            self.norm = lambda x: self._norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        if pretrained_ckpt_path:
            self.load_ckpt_path(pretrained_ckpt_path)

        self.save_hyperparameters()

    def _get_encoding(self, x, mask=None, do_norm=True):
        if self.norm is not None and do_norm:
            x = self.norm(x)
        if self.training:
            for augmentation in self.augmentations:
                x = augmentation(x, mask=mask)
        return self.encoder.encode(x)

    def _forward_autoencoder(self, inputs_embeds):
        mask = None
        if self.is_autoencoder:
            mask = torch.ones_like(inputs_embeds, dtype=torch.bool)
        x = inputs_embeds
        if self.norm is not None:
            x = self.norm(x)
        x_ = x.clone()
        if self.training:
            for augmentation in self.augmentations:
                x = augmentation(x, mask=mask)
        encoding = self.encoder.encode(x)
        preds = self.head(encoding)
        # for autoencoders, usually
        for masker in self.maskers:
            mask = masker(mask=mask, x=inputs_embeds)
        if not self.denoising:
            x = x_
        loss = self.criterion(preds[mask], x[mask])
        return loss, preds

    def _forward_normal(self, inputs_embeds, labels):
        encoding = self._get_encoding(inputs_embeds)
        preds = self.head(encoding)
        loss = self.criterion(preds, labels)
        return loss, preds

    def _forward_combined_classifier(self, inputs_embeds, labels):
        assert self.combine_rows == 2
        inputs_embeds_L = inputs_embeds[:, :, :self.encoder.input_dim[-1]]
        inputs_embeds_R = inputs_embeds[:, :, self.encoder.input_dim[-1]:]

        # same_user

        encoding_L = self._get_encoding(inputs_embeds_L)
        encoding_R = self._get_encoding(inputs_embeds_R)

        encoding = torch.cat([encoding_L, encoding_R], dim=2)
        preds = self.head(encoding)
        loss = self.criterion(preds, labels)
        return loss, preds

    def _forward_triplet(self, inputs_embeds):
        assert self.combine_rows == 2
        inputs_embeds_L = inputs_embeds[:, :, :self.encoder.input_dim[-1]]
        inputs_embeds_R = inputs_embeds[:, :, self.encoder.input_dim[-1]:]

        # triplet
        inputs_embeds_L1 = inputs_embeds_L
        inputs_embeds_L2 = inputs_embeds_L

        encoding_L1 = self._get_encoding(inputs_embeds_L1)
        encoding_L2 = self._get_encoding(inputs_embeds_L2)
        encoding_R = self._get_encoding(inputs_embeds_R)

        anchor = self.head(encoding_L1)
        positive = self.head(encoding_L2)
        negative = self.head(encoding_R)
        loss = self.criterion(anchor, positive, negative)
        return loss, anchor

    def _forward_contrastive(self, inputs_embeds):
        # get encodings, which differ if a random DA is applied
        encoding_1 = self._get_encoding(inputs_embeds)
        encoding_2 = self._get_encoding(inputs_embeds)

        query = self.head(encoding_1)
        positive = self.head(encoding_2)
        loss = self.criterion(query, positive)
        return loss, query

    def _handle_single(self, inputs_embeds, labels, task=None):
        if self.combine_rows:
            if self.is_triplet or task == 'contrastive':
                return self._forward_triplet(inputs_embeds)
            if self.is_classifier:  # same_user
                return self._forward_combined_classifier(inputs_embeds, labels)

        if self.is_autoencoder:
            return self._forward_autoencoder(inputs_embeds)
        if self.is_classifier:
            return self._forward_normal(inputs_embeds, labels)
        if self.is_contrastive:
            return self._forward_contrastive(inputs_embeds)
        if self.is_regressor or task == 'regression':
            return self._forward_normal(inputs_embeds, labels)
        raise ValueError

    def forward(self, inputs_embeds, labels):
        if self.is_multihead:
            x = inputs_embeds
            if self.combine_rows:
                assert self.combine_rows == 2
                x = x[:, :, :self.encoder.input_dim[-1]]

            loss_a, preds_a = self._forward_autoencoder(x)

            # Swap heads and loss, then undo
            self.head, self._second_head = self._second_head, self.head
            self.criterion, self._second_criterion = self._second_criterion, self.criterion
            loss_b, preds_b = self._handle_single(inputs_embeds, labels, task=self.multihead_mode)
            self.head, self._second_head = self._second_head, self.head
            self.criterion, self._second_criterion = self._second_criterion, self.criterion

            return loss_a + loss_b, preds_b

        return self._handle_single(inputs_embeds, labels)

    def load_ckpt_path(self, pretrained_ckpt_path):
        print(f"Received ckpt_path {pretrained_ckpt_path}")

        ckpt = torch.load(pretrained_ckpt_path)
        try:
            self.load_state_dict(ckpt['state_dict'])

        except RuntimeError as re:
            print(f"{re} loading state_dict from ckpt, now doing 'Nasty hack for reverse compatibility'")
            new_state_dict = {}
            for k, v in ckpt["state_dict"].items():
                if "encoder" not in k:
                    new_state_dict["encoder." + k] = v
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict, strict=False)
            print("Loaded new_state_dict with strict=False")
