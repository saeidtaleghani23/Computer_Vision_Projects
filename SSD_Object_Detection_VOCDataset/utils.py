import math
import torch
from typing import List


def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    :param boxes1: (Tensor of shape N x 4)
    :param boxes2: (Tensor of shape M x 4)
    :return: IOU matrix of shape N x M
    """

    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * \
        (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * \
        (boxes2[:, 3] - boxes2[:, 1])  # (M,)

    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)

    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)

    intersection_area = ((x_right - x_left).clamp(min=0) *
                         (y_bottom - y_top).clamp(min=0))  # (N, M)
    union = area1[:, None] + area2 - intersection_area  # (N, M)
    iou = intersection_area / union  # (N, M)
    return iou


def generate_default_boxes(feature_maps: List[torch.Tensor], aspect_ratios: List[List[float]], scales: List[float]) -> List[torch.Tensor]:
    """
    Generates default (anchor) boxes for each feature map layer in the SSD model.
    It is used in the initialization phase to set up the default boxes that will be matched with ground truth boxes. 
    Each feature map (e.g., from different layers of the network) contributes default boxes of different sizes, allowing SSD to detect objects of various scales.

    Args:
        feature_maps (List[torch.Tensor]): A list of tensors representing the six feature map layers of the model.
            Each tensor has a shape (B, C, Feat_H, Feat_W), where:
                - B is the batch size
                - C is the number of channels
                - Feat_H and Feat_W are the height and width of the feature map.
        aspect_ratios (List[List[float]]): A list of lists where each sublist contains aspect ratios
            for the corresponding feature map layer.
        scales (List[float]): A list of scale values, one for each feature map layer, used to set the size
            of the default boxes relative to the input image.

    Returns:
        List[torch.Tensor]: A list of tensors, each tensor representing the default boxes for a feature map layer.
            Each tensor has a shape (N, 4), where N is the number of boxes and 4 represents the coordinates (x1, y1, x2, y2).
    """
    # List to store default boxes for all feature maps
    default_boxes = []
    """ simple method to calculate default_boxex
    #loop over each feature map
    for feature_map_idx in range(len(feature_maps)):
        h_feat_map, w_feat_map = feature_maps[feature_map_idx].shape[-2:]
        #loop over each cell to shift the center of boxes
        for i in range(w_feat_map):
            for j in range(h_feat_map):
                cx =  (j + 0.5) / w_feat_map
                cy =  (j + 0.5) / h_feat_map
                for ratio in aspect_ratios[feature_map_idx]:
                    default_boxes.append([cx, cy, scales[feature_map_idx] * math.sqrt(ratio), scales[feature_map_idx] / math.sqrt(ratio)])
                    # For an aspect ratio of 1, use an additional prior
                    if ratio ==1.:
                        try: 
                            additional_scale = math.sqrt(scales[feature_map_idx] * scales[feature_map_idx+1])
                            # For the last feature map, there is no "next" feature map
                        except IndexError:
                            additional_scale = 1.
                        default_boxes.append([cx, cy, additional_scale, additional_scale])
    default_boxes = torch.FloatTensor(default_boxes).to(self.device)  # (8732, 4)
    default_boxes.clamp_(0, 1)  # (8732, 4)
    """
    for feature_map_idx in range(len(feature_maps)):
        # first of all, add the aspect ratio 1 and scale (sqrt(scale[k])*sqrt(scale[k+1])
        additional_scale = math.sqrt(
            scales[feature_map_idx] * scales[feature_map_idx+1])
        wh_pairs = [[additional_scale, additional_scale]]
        # add all possible w and h of bounding boxes of a feature map[feature_map_idx] according to its aspect ratio[feature_map_idx]
        for aspect_ratio in aspect_ratios[feature_map_idx]:
            w = scales[feature_map_idx] * math.sqrt(aspect_ratio)
            h = scales[feature_map_idx] / math.sqrt(aspect_ratio)
            wh_pairs.extend([[w, h]])
        # spatial size of the feature map[feature_map_idx]
        h_feat_map, w_feat_map = feature_maps[feature_map_idx].shape[-2:]
        # These shifts will be the centre of each of the default boxes
        shifts_x = ((torch.arange(0, w_feat_map) + 0.5) /
                    w_feat_map).to(torch.float32)

        shifts_y = ((torch.arange(0, h_feat_map) + 0.5) /
                    h_feat_map).to(torch.float32)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        # Duplicate these shifts for as
        # many boxes(aspect ratios)
        # per position we have
        shifts = torch.stack((shift_x, shift_y) *
                             len(wh_pairs), dim=-1).reshape(-1, 2)
        # shifts for first feature map will be (5776 x 2)

        wh_pairs = torch.as_tensor(wh_pairs)

        # Repeat the wh pairs for all positions in feature map
        wh_pairs = wh_pairs.repeat((h_feat_map * w_feat_map), 1)
        # wh_pairs for first feature map will be (5776 x 2)

        # Concat the shifts(cx cy) and wh values for all positions
        default_box = torch.cat((shifts, wh_pairs), dim=1)
        # default box for feat_1 -> (5776, 4)
        # default box for feat_2 -> (2166, 4)
        # default box for feat_3 -> (600, 4)
        # default box for feat_4 -> (150, 4)
        # default box for feat_5 -> (36, 4)
        # default box for feat_6 -> (4, 4)
        default_boxes.append(default_box)

    default_boxes = torch.cat(default_boxes, dim=0)
    # default_boxes -> (8732, 4)

    # dboxes = []
    # for _ in range(feature_maps[0].size(0)):
    #     dboxes_in_image = default_boxes
    #     # x1 = cx - 0.5 * width
    #     # y1 = cy - 0.5 * height
    #     # x2 = cx + 0.5 * width
    #     # y2 = cy + 0.5 * height
    #     dboxes_in_image = torch.cat(
    #         [
    #             (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]),
    #             (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]),
    #         ],
    #         -1,
    #     )
    #     dboxes.append(dboxes_in_image.to(feature_maps[0].device))

    return default_boxes


def xy_to_cxcy(xy):
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    Args:
        xy (torch.Tensor): Tensor of shape (N, 4), where each row contains bounding box 
            coordinates in (x1, y1, x2, y2) format.

   Returns:
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: 
            - center_x (torch.Tensor): Tensor of shape (N,) containing center x-coordinates.
            - center_y (torch.Tensor): Tensor of shape (N,) containing center y-coordinates.
            - widths (torch.Tensor): Tensor of shape (N,) containing widths of bounding boxes.
            - heights (torch.Tensor): Tensor of shape (N,) containing heights of bounding boxes.
    """
    # Calculate the center coordinates, width, and height
    center_x = (xy[:, 2] + xy[:, 0]) / 2  # (x2 + x1) / 2
    center_y = (xy[:, 3] + xy[:, 1]) / 2  # (y2 + y1) / 2
    widths = xy[:, 2] - xy[:, 0]          # x2 - x1
    heights = xy[:, 3] - xy[:, 1]         # y2 - y1
    return center_x, center_y, widths, heights


def cxcy_to_xy(cxcy):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        cxcy (torch.Tensor): Tensor of shape (N, 4), where each row contains bounding box
            coordinates in (cx, cy, w, h) format.

    Returns:
        torch.Tensor: Tensor of shape (N, 4), where each row contains bounding box coordinates
            in (x1, y1, x2, y2) format, where:
            - x1, y1 are the coordinates of the top-left corner
            - x2, y2 are the coordinates of the bottom-right corner.
    """
    dboxes_in_image = torch.cat(
        [
            (cxcy[:, :2] - 0.5 * cxcy[:, 2:]),
            (cxcy[:, :2] + 0.5 * cxcy[:, 2:]),
        ],
        -1,
    )
    return dboxes_in_image
    # return torch.cat([cxcy[:, :2] - 0.5 * (cxcy[:, 2:]),  # x_min, y_min
    #                   cxcy[:, :2] + 0.5 * (cxcy[:, 2:])], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """ Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    Args:
        cxcy (torch.Tensor): Tensor of shape (N, 4), where each row contains bounding box
            coordinates in (cx, cy, w, h) format.
        priors_cxcy (torch.Tensor): Tensor of shape (N, 4), where each row contains prior box
            coordinates in (cx, cy, w, h) format used for encoding.

    Returns:
        torch.Tensor: Tensor of shape (N, 4), where each row contains encoded bounding box
            coordinates (g_cx, g_cy, g_w, g_h), where:
            - g_cx and g_cy are the offsets of the bounding box center relative to the prior box center, scaled by priors' width and height
            - g_w and g_h are the log-scaled ratios of bounding box width and height to prior box width and height.
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

   Args:
        gcxgcy (torch.Tensor): Tensor of shape (N, 4), where each row contains encoded bounding box
            coordinates in (g_cx, g_cy, g_w, g_h) format.
        priors_cxcy (torch.Tensor): Tensor of shape (N, 4), where each row contains prior box
            coordinates in (cx, cy, w, h) format used for decoding.

    Returns:
        torch.Tensor: Tensor of shape (N, 4), where each row contains bounding box coordinates
            in (cx, cy, w, h) format, where:
            - cx and cy are the center coordinates of the bounding box
            - w and h are the width and height of the bounding box, respectively.
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def boxes_to_transformation_targets(ground_truth_boxes,
                                    default_boxes,
                                    weights=(10., 10., 5., 5.)):
    """
    Method to compute targets for each default_boxes. This function is essential to train SSD model
    Assumes boxes are in x1y1x2y2 format.
    We first convert  (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    then, compute targets based on following formulation
    target_dx = (gt_cx - default_boxes_cx) / default_boxes_w
    target_dy = (gt_cy - default_boxes_cy) / default_boxes_h
    target_dw = log(gt_w / default_boxes_w)
    target_dh = log(gt_h / default_boxes_h)

    Args:
            ground_truth_boxes: (Tensor of shape N x 4)
            default_boxes: (Tensor of shape N x 4)
            weights: Tuple[float] -> (wx, wy, ww, wh)

    Return:
            regression_targets: (Tensor of shape N x 4)
    """
    # Get center_x,center_y,w,h from x1,y1,x2,y2 for default_boxes
    center_x, center_y, widths, heights = xy_to_cxcy(default_boxes)
    # Get center_x,center_y,w,h from x1,y1,x2,y2 for gt boxes
    gt_center_x, gt_center_y, gt_widths, gt_heights = xy_to_cxcy(default_boxes)

    # Use formulation to compute all targets
    targets_dx = weights[0] * (gt_center_x - center_x) / widths
    targets_dy = weights[1] * (gt_center_y - center_y) / heights
    targets_dw = weights[2] * torch.log(gt_widths / widths)
    targets_dh = weights[3] * torch.log(gt_heights / heights)
    regression_targets = torch.stack((targets_dx,
                                      targets_dy,
                                      targets_dw,
                                      targets_dh), dim=1)
    return regression_targets


def apply_regression_pred_to_default_boxes(box_transform_pred,
                                           default_boxes,
                                           weights=(10., 10., 5., 5.)):
    r"""
    Method to transform default_boxes based on transformation parameter
    prediction. This is function is used in the inference/ prediction phase of the SSD model.
    Assumes boxes are in x1y1x2y2 format
    :param box_transform_pred: (Tensor of shape N x 4)
    :param default_boxes: (Tensor of shape N x 4)
    :param weights: Tuple[float] -> (wx, wy, ww, wh)
    :return: pred_boxes: (Tensor of shape N x 4)
    """

    # Get cx, cy, w, h from x1,y1,x2,y2
    center_x, center_y, w, h = xy_to_cxcy(default_boxes)
    # [..., 0] is a form of indexing that selects the first element of the last dimension. It gives you a 1D tensor
    dx = box_transform_pred[..., 0] / weights[0]
    dy = box_transform_pred[..., 1] / weights[1]
    dw = box_transform_pred[..., 2] / weights[2]
    dh = box_transform_pred[..., 3] / weights[3]
    # dh -> (num_default_boxes)

    pred_center_x = dx * w + center_x
    pred_center_y = dy * h + center_y
    pred_w = torch.exp(dw) * w
    pred_h = torch.exp(dh) * h
    # pred_center_x -> (num_default_boxes, 4)

    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    pred_boxes = torch.stack((
        pred_box_x1,
        pred_box_y1,
        pred_box_x2,
        pred_box_y2),
        dim=-1)
    return pred_boxes


def compute_loss(
        self,
        targets,
        cls_logits,
        bbox_regression,
        default_boxes,
        matched_idxs,
):
    # Counting all the foreground default_boxes for computing N in loss equation
    num_foreground = 0
    # BBox losses for all batch images(for foreground default_boxes)
    bbox_loss = []
    # classification targets for all batch images(for ALL default_boxes)
    cls_targets = []
    for (
        targets_per_image,
        bbox_regression_per_image,
        cls_logits_per_image,
        default_boxes_per_image,
        matched_idxs_per_image,
    ) in zip(targets, bbox_regression, cls_logits, default_boxes, matched_idxs):
        # Foreground default_boxes -> matched_idx >=0
        # Background default_boxes -> matched_idx = -1
        fg_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        foreground_matched_idxs_per_image = matched_idxs_per_image[
            fg_idxs_per_image
        ]
        num_foreground += foreground_matched_idxs_per_image.numel()

        # Get foreground default_boxes and their transformation predictions
        matched_gt_boxes_per_image = targets_per_image["boxes"][
            foreground_matched_idxs_per_image
        ]
        bbox_regression_per_image = bbox_regression_per_image[fg_idxs_per_image, :]
        default_boxes_per_image = default_boxes_per_image[fg_idxs_per_image, :]
        target_regression = boxes_to_transformation_targets(
            matched_gt_boxes_per_image,
            default_boxes_per_image)

        bbox_loss.append(
            torch.nn.functional.smooth_l1_loss(bbox_regression_per_image,
                                               target_regression,
                                               reduction='sum')
        )

        # Get classification target for ALL default_boxes
        # For all default_boxes set it as 0 first
        # Then set foreground default_boxes target as label
        # of assigned gt box
        gt_classes_target = torch.zeros(
            (cls_logits_per_image.size(0),),
            dtype=targets_per_image["labels"].dtype,
            device=targets_per_image["labels"].device,
        )
        gt_classes_target[fg_idxs_per_image] = targets_per_image["labels"][
            foreground_matched_idxs_per_image
        ]
        cls_targets.append(gt_classes_target)

    # Aggregated bbox loss and classification targets
    # for all batch images
    bbox_loss = torch.stack(bbox_loss)
    cls_targets = torch.stack(cls_targets)  # (B, 8732)

    # Calculate classification loss for ALL default_boxes
    num_classes = cls_logits.size(-1)
    cls_loss = torch.nn.functional.cross_entropy(cls_logits.view(-1, num_classes),
                                                 cls_targets.view(-1),
                                                 reduction="none").view(
        cls_targets.size()
    )

    # Hard Negative Mining
    foreground_idxs = cls_targets > 0
    # We will sample total of 3 x (number of fg default_boxes)
    # background default_boxes
    num_negative = self.neg_pos_ratio * foreground_idxs.sum(1, keepdim=True)

    # As of now cls_loss is for ALL default_boxes
    negative_loss = cls_loss.clone()
    # We want to ensure that after sorting based on loss value,
    # foreground default_boxes are never picked when choosing topK
    # highest loss indexes
    negative_loss[foreground_idxs] = -float("inf")
    values, idx = negative_loss.sort(1, descending=True)
    # Fetch those indexes which have in topK(K=num_negative) losses
    background_idxs = idx.sort(1)[1] < num_negative
    N = max(1, num_foreground)
    return {
        "bbox_regression": bbox_loss.sum() / N,
        "classification": (cls_loss[foreground_idxs].sum() +
                           cls_loss[background_idxs].sum()) / N,
    }
