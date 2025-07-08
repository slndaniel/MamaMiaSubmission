import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F

def positive_loss_function(prediction, label):

    B, C, D, H, W = prediction.shape

    positive_mask = label.squeeze(1) == 1  # (B, D, H, W)

    prediction_reshaped = prediction.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
    positive_features = prediction_reshaped[positive_mask]  # (N_pos, C)

    if positive_features.shape[0] > 0:
        standard_positive_feature = positive_features.mean(dim=0)  # (C,)
    else:
        standard_positive_feature = torch.zeros(C, device=prediction.device)

    if positive_features.shape[0] > 0:
        positive_similarities = F.cosine_similarity(
            positive_features, standard_positive_feature.unsqueeze(0), dim=1
        )
        positive_loss = torch.mean(1 - positive_similarities)
    else:
        positive_loss = 0.0
    return positive_loss



def misclassified_loss_function(prediction, label):

    positive_mask = label.squeeze(1) == 1  # (B, D, H, W)

    prediction_reshaped = prediction.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
    positive_features = prediction_reshaped[positive_mask]  # (N_pos, C)

    if positive_features.shape[0] > 0:
        standard_positive_feature = positive_features.mean(dim=0)  #(C,)
    else:
        standard_positive_feature = torch.zeros(C, device=prediction.device)



    kernel = torch.ones((1, 1, 3, 3, 3), device=prediction.device)
    padding = 1

    positive_mask_float = positive_mask.float().unsqueeze(1)  # ??:(B, 1, D, H, W)

    dilated_mask = positive_mask_float.clone()
    for _ in range(10):
        dilated_mask = F.conv3d(dilated_mask, kernel, padding=padding)
        dilated_mask = (dilated_mask > 0).float()

    dilated_mask = dilated_mask.squeeze(1)  # ??:(B, D, H, W)
    easily_misclassified_mask = (dilated_mask == 1) & (
        positive_mask == 0
    )  # (B, D, H, W)

    misclassified_features = prediction_reshaped[
        easily_misclassified_mask
    ]  # (N_mis, C)

    if misclassified_features.shape[0] > 0:
        # misclassified_features_norm = F.normalize(misclassified_features, p=2, dim=1)
        misclassified_similarities = F.cosine_similarity(
            misclassified_features,
            standard_positive_feature.unsqueeze(0),
            dim=1,
        )
        misclassified_loss = torch.mean(F.relu(misclassified_similarities))
    else:
        misclassified_loss = 0.0

    return misclassified_loss


def FRLoss(prediction, label):
   
    positive_mask = label.squeeze(1) == 1  # (B, D, H, W)

    prediction_reshaped = prediction.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)

    positive_features = prediction_reshaped[positive_mask]  # (N_pos, C)

    if positive_features.shape[0] > 0:
        standard_positive_feature = positive_features.mean(dim=0)  # (C,)
    else:
        standard_positive_feature = torch.zeros(C, device=prediction.device)

    standard_positive_feature_norm = F.normalize(standard_positive_feature, p=2, dim=0)

    if positive_features.shape[0] > 0:
        positive_features_norm = F.normalize(positive_features, p=2, dim=1)
        positive_similarities = F.cosine_similarity(
            positive_features_norm, standard_positive_feature_norm.unsqueeze(0), dim=1
        )
        positive_loss = torch.mean(1 - positive_similarities)
    else:
        positive_loss = 0.0


    kernel = torch.ones((1, 1, 3, 3, 3), device=prediction.device)
    padding = 1

    positive_mask_float = positive_mask.float().unsqueeze(1)  # (B, 1, D, H, W)

    dilated_mask = positive_mask_float.clone()
    for _ in range(10):
        dilated_mask = F.conv3d(dilated_mask, kernel, padding=padding)
        dilated_mask = (dilated_mask > 0).float()

    dilated_mask = dilated_mask.squeeze(1)  # (B, D, H, W)
    easily_misclassified_mask = (dilated_mask == 1) & (
        ~positive_mask
    )  # (B, D, H, W)

    misclassified_features = prediction_reshaped[
        easily_misclassified_mask
    ]  # (N_mis, C)

    if misclassified_features.shape[0] > 0:
        misclassified_features_norm = F.normalize(misclassified_features, p=2, dim=1)
        misclassified_similarities = F.cosine_similarity(
            misclassified_features_norm,
            standard_positive_feature_norm.unsqueeze(0),
            dim=1,
        )
        misclassified_loss = torch.mean(F.relu(misclassified_similarities))
    else:
        misclassified_loss = 0.0


    negative_mask = label.squeeze(1) == 0  #(B, D, H, W)

    prediction_reshaped_norm = F.normalize(
        prediction_reshaped, p=2, dim=-1
    )  # (B, D, H, W, C)

    cosine_similarity_map = torch.einsum(
        "bdhwc,c->bdhw", prediction_reshaped_norm, standard_positive_feature_norm
    )
    # (B, D, H, W)

    negative_similarities = cosine_similarity_map[negative_mask]  # ??:(N_neg,)

    if negative_similarities.numel() > 0:
        top_N = 250  
        if negative_similarities.numel() < top_N:
            top_N = negative_similarities.numel()

        sorted_similarities, indices = torch.topk(
            negative_similarities, top_N, largest=True
        )

        negative_positions = negative_mask.nonzero(as_tuple=False)  # (N_neg, 4)
        top_positions = negative_positions[indices]  # (top_N, 4)

        high_similarity_mask = torch.zeros_like(negative_mask, dtype=torch.float)
        high_similarity_mask[
            top_positions[:, 0],
            top_positions[:, 1],
            top_positions[:, 2],
            top_positions[:, 3],
        ] = 1.0

        dilated_high_similarity_mask = high_similarity_mask.unsqueeze(
            1
        )  # (B, 1, D, H, W)
        for _ in range(10):
            dilated_high_similarity_mask = F.conv3d(
                dilated_high_similarity_mask, kernel, padding=padding
            )
            dilated_high_similarity_mask = (dilated_high_similarity_mask > 0).float()
        dilated_high_similarity_mask = dilated_high_similarity_mask.squeeze(
            1
        )  #(B, D, H, W)

        final_negative_regions = (dilated_high_similarity_mask == 1) & (~positive_mask)

        negative_dilated_features = prediction_reshaped_norm[final_negative_regions]

        if negative_dilated_features.shape[0] > 0:
            negative_dilated_similarities = F.cosine_similarity(
                negative_dilated_features,
                standard_positive_feature_norm.unsqueeze(0),
                dim=1,
            )
            negative_dilated_loss = torch.mean(F.relu(negative_dilated_similarities))
        else:
            negative_dilated_loss = 0.0
    else:
        negative_dilated_loss = 0.0

    total_loss = positive_loss + misclassified_loss + negative_dilated_loss

    return total_loss
    
class DC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=SoftDiceLoss,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(
        self, feature: torch.Tensor, net_output: torch.Tensor, target: torch.Tensor
    ):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, (
                "ignore label is not implemented for one hot encoded target variables "
                "(DC_and_CE_loss)"
            )
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask)
            if self.weight_dice != 0
            else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )
        # positive_loss=positive_loss_function(feature,target)
        # pw=5
        # result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + pw * positive_loss
        # misclassified_loss=misclassified_loss_function(feature,target)
        # mw=5
        # result = self.weight_ce * ce_loss + self.weight_dice * dc_loss +  mw * misclassified_loss
        # result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + pw * positive_loss + mw * misclassified_loss
        self_loss = FRLoss(feature, target)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + 5 * self_loss
        # result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
  
"""     
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        
        #Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        #:param soft_dice_kwargs:
        #:param ce_kwargs:
        #:param aggregate:
        #:param square_dice:
        #:param weight_ce:
        #:param weight_dice:
        
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        
        #target must be b, c, x, y(, z) with c=1
        #:param net_output:
        #:param target:
        #:return:
        
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
"""
        


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
