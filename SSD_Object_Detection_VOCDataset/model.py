# %%
import torch.nn as nn
import torch
import torchvision
import torchvision.models as models
import yaml
from utils import *

# %%


class SSD(nn.Module):
    def __init__(self, config):
        super(SSD, self).__init__()
        # get hyper parameters from config file
        self.aspect_ratios = config['model_params']['aspect_ratios']
        self.scales = config['model_params']['scales']
        self.scales.append(1.0)
        self.no_boxes = config['model_params']['n_boxes']
        self.channels = config['model_params']['channels']
        self.num_classes = config['dataset_params']['num_classes']

        # Load the pretrained VGG-16 model
        base_model = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        # Extract all maxpool layers' index
        max_pool_pos = [idx for idx, layer in enumerate(
            list(base_model.features)) if isinstance(layer, nn.MaxPool2d)]
        # force the output of conv4_3 to have shape 38x38 instead of 37x37
        base_model.features[max_pool_pos[-3]].ceil_mode = True
        # extract the model at conv4_3
        self.conv4_3_features = nn.Sequential(
            *base_model.features[:max_pool_pos[-2]])  # output: N x 512 x 38 x 38

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        # there are 512 channels in conv4_3_feats
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        ###################################
        # Conv5_3 = nn.Sequential(*base_model.features[max_pool_pos[-2]:-1]) : 512x19x19
        # Conv for fc6 and fc 7 #
        ###################################
        convs_instead_fcs = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1,
                         padding=1),  # N x 512 x 19 x 19
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                      padding=6, dilation=6),  # Conv6 (FC6) : N x 1024 x 19 x 19
            nn.ReLU(inplace=True),
            # Conv7 (FC7) : N x 1024 x 19 x 19
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # initialize weights for convs_instead_fcs
        for layer in convs_instead_fcs:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.0)
        # attach convs_instead_fcs and layers after conv4_3_features
        self.conv7_features = nn.Sequential(  # inout is the output of the  self.conv4_3_features with the shape of  N x 512 x 38 x 38
            *base_model.features[max_pool_pos[-2]:-1],
            convs_instead_fcs,
        )  # output:  N x 1024 x 19 x 19

        ###################################
        # additional convolutions on top of the VGG base
        ###################################
        # Modules to take from 19x19 to 10x10
        self.conv8_2_features = nn.Sequential(
            # stride = 1, by default
            nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # dim. reduction because stride > 1
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Modules to take from 10x10 to 5x5
        self.conv9_2_features = nn.Sequential(
            # stride = 1, by default
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # dim. reduction because stride > 1
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Modules to take from 5x5 to 3x3
        self.conv10_2_features = nn.Sequential(
            # stride = 1, by default
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # dim. reduction because stride > 1
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Modules to take from 3x3 to 1x1
        self.conv11_2_features = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        # Initialize weights for additional convolutions
        for conv_layer in [self.conv8_2_features, self.conv9_2_features, self.conv10_2_features, self.conv11_2_features]:
            for layer in conv_layer.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0.0)
        ###################################
        #  Prediction layers
        ###################################
        # classification head
        self.classification_head = nn.ModuleList()
        # localization head
        self.bounding_box_head = nn.ModuleList()

        for input_channels, no_boxes in zip(self.channels, self.no_boxes):
            self.classification_head.append(nn.Conv2d(
                input_channels,  no_boxes * self.num_classes, kernel_size=3, padding=1))
            self.bounding_box_head.append(
                nn.Conv2d(input_channels, 4 * no_boxes, kernel_size=3, padding=1))

            # initialize conv weights
            for module in self.classification_head:
                torch.nn. init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            for module in self.bounding_box_head:
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def prediction(self, outputs):
        # Classification and bbox regression for all feature maps
        cls_logits = []
        bbox_reg_deltas = []
        for i, features in enumerate(outputs):
            # Predict classes in localization boxes
            cls_feat_i = self.classification_head[i](features)
            # classification_head output from (batch_size, no_boxes * num_classes, H, W) to (N, no_boxes *HW, num_classes).
            N, _, H, W = cls_feat_i.shape
            # (batch_size, no_boxes, num_classes, H, W)
            cls_feat_i = cls_feat_i.view(N, -1, self.num_classes, H, W)
            # (batch_size, H, W, no_boxes, num_classes)
            cls_feat_i = cls_feat_i.permute(0, 3, 4, 1, 2)
            # (batch_size, no_boxes *HW, num_classes)
            cls_feat_i = cls_feat_i.reshape(N, -1, self.num_classes)
            cls_logits.append(cls_feat_i)

            # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
            bbox_reg_feat_i = self.bounding_box_head[i](features)
            # Permute bbox reg output from (batch_size, no_boxes * 4, H, W) to (batch_size, no_boxes*HW, 4).
            N, _, H, W = bbox_reg_feat_i.shape
            bbox_reg_feat_i = bbox_reg_feat_i.view(
                N, -1, 4, H, W)  # (batch_size, no_boxes, 4, H, W)
            bbox_reg_feat_i = bbox_reg_feat_i.permute(
                0, 3, 4, 1, 2)  # (batch_size, H, W, no_boxes, 4)
            bbox_reg_feat_i = bbox_reg_feat_i.reshape(
                N, -1, 4)  # Size=(batch_size, no_boxes*HW, 4)
            bbox_reg_deltas.append(bbox_reg_feat_i)

        # Concat cls logits and bbox regression predictions for all feature maps
        # (batch_size, 8732, num_classes)
        classes_scores = torch.cat(cls_logits, dim=1)
        locs = torch.cat(
            bbox_reg_deltas, dim=1)  # (batch_size, 8732, 4)
        return locs, classes_scores

    def forward(self, image, targets=None):
        """
        Forward propagation.

        Args:
            image (Tensor): a tensor of dimensions (N, 3, 300, 300)

        Returns:
            _type_: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_features_out = self.conv4_3_features(image)  # (N, 512, 38, 38)
        # Rescale conv4_3 after L2 norm
        norm = conv4_3_features_out.pow(2).sum(
            dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_features_out_out_scaled = conv4_3_features_out / \
            norm  # (N, 512, 38, 38)
        conv4_3_features_out_scaled = conv4_3_features_out_out_scaled * \
            self.rescale_factors  # (N, 512, 38, 38)

        conv7_features_out = self.conv7_features(
            conv4_3_features_out)  # ( N, 1024, 19, 19)

        conv8_2_features_out = self.conv8_2_features(conv7_features_out)

        conv9_2_features_out = self.conv9_2_features(conv8_2_features_out)

        conv10_2_features_out = self.conv10_2_features(conv9_2_features_out)

        conv11_2_features_out = self.conv11_2_features(conv10_2_features_out)

        # Feature maps for predictions
        outputs = [
            conv4_3_features_out_scaled,  # 38 x 38
            conv7_features_out,  # 19 x 19
            conv8_2_features_out,  # 10 x 10
            conv9_2_features_out,  # 5 x 5
            conv10_2_features_out,  # 3 x 3
            conv11_2_features_out,   # 1 x 1
        ]

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        bbox_reg_deltas, classes_logits = self.prediction(outputs)
        # bbox_reg_deltas: (N, 8732, 4),   classes_logits :  (N, 8732, n_classes)

        # Generate default_boxes for all feature maps
        default_boxes = generate_default_boxes(
            outputs, self.aspect_ratios, self.scales)
        # default_boxes are in  cx,cy,w,h format
        # we need to convert their format into x1,y1,x2,y2
        dboxes = cxcy_to_xy(default_boxes)
        losses = {}
        detections = []
        if self.training:
            # List to hold for each image, which default box
            # is assigned to with gt box if any
            # or unassigned(background)
            matched_idxs = []
            for default_boxes_per_image, targets_per_image in zip(default_boxes,
                                                                  targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (default_boxes_per_image.size(0),), -1,
                            dtype=torch.int64,
                            device=default_boxes_per_image.device
                        )
                    )
                    continue
                iou_matrix = get_iou(targets_per_image["boxes"],
                                     default_boxes_per_image)
                # For each default box find best ground truth box
                matched_vals, matches = iou_matrix.max(dim=0)
                # matches -> [8732]

                # Update index of match for all default_boxes which
                # have maximum iou with a gt box < low threshold
                # as -1
                # This allows selecting foreground boxes as match index >= 0
                below_low_threshold = matched_vals < self.iou_threshold
                matches[below_low_threshold] = -1

                # We want to also assign the best default box for every gt
                # as foreground
                # So first find the best default box for every gt
                _, highest_quality_pred_foreach_gt = iou_matrix.max(dim=1)
                # Update the best matching gt index for these best default_boxes
                # as 0, 1, 2, ...., len(gt)-1
                matches[highest_quality_pred_foreach_gt] = torch.arange(
                    highest_quality_pred_foreach_gt.size(0), dtype=torch.int64,
                    device=highest_quality_pred_foreach_gt.device
                )
                matched_idxs.append(matches)
            losses = self.compute_loss(targets, classes_logits, bbox_reg_deltas,
                                       default_boxes, matched_idxs)
        else:
            # For test time we do the following:
            # 1. Convert default_boxes to boxes using predicted bbox regression deltas
            # 2. Low score filtering
            # 3. Pre-NMS TopK filtering
            # 4. NMS
            # 5. Post NMS TopK Filtering
            cls_scores = torch.nn.functional.softmax(classes_logits, dim=-1)
            num_classes = cls_scores.size(-1)

            for bbox_deltas_i, cls_scores_i, default_boxes_i in zip(bbox_reg_deltas,
                                                                    cls_scores,
                                                                    default_boxes):
                boxes = apply_regression_pred_to_default_boxes(bbox_deltas_i,
                                                               default_boxes_i)
                # Ensure all values are between 0-1
                boxes.clamp_(min=0., max=1.)

                pred_boxes = []
                pred_scores = []
                pred_labels = []
                # Class wise filtering
                for label in range(1, num_classes):
                    score = cls_scores_i[:, label]

                    # Remove low scoring boxes of this class
                    keep_idxs = score > self.low_score_threshold
                    score = score[keep_idxs]
                    box = boxes[keep_idxs]

                    # keep only topk scoring predictions of this class
                    score, top_k_idxs = score.topk(
                        min(self.pre_nms_topK, len(score)))
                    box = box[top_k_idxs]

                    pred_boxes.append(box)
                    pred_scores.append(score)
                    pred_labels.append(torch.full_like(score, fill_value=label,
                                                       dtype=torch.int64,
                                                       device=cls_scores.device))

                pred_boxes = torch.cat(pred_boxes, dim=0)
                pred_scores = torch.cat(pred_scores, dim=0)
                pred_labels = torch.cat(pred_labels, dim=0)

                # Class wise NMS
                keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
                for class_id in torch.unique(pred_labels):
                    curr_indices = torch.where(pred_labels == class_id)[0]
                    curr_keep_idxs = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                               pred_scores[curr_indices],
                                                               self.nms_threshold)
                    keep_mask[curr_indices[curr_keep_idxs]] = True
                keep_indices = torch.where(keep_mask)[0]
                post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(
                    descending=True)[1]]
                keep = post_nms_keep_indices[:self.detections_per_img]
                pred_boxes, pred_scores, pred_labels = (pred_boxes[keep],
                                                        pred_scores[keep],
                                                        pred_labels[keep])

                detections.append(
                    {
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                    }
                )
        return losses, detections
