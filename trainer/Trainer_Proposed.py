# trainer/Trainer_Proposed.py

import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Import necessary modules from the original project structure
from utils.loss import loss_calc, dice_loss
from utils.lr_adjust import adjust_learning_rate #, adjust_learning_rate_custom # Use adjust_learning_rate directly
from utils.utils_ import update_class_center_iter # Keep for source centroid update if desired
from utils import timer
import config

# Import base Trainer class and potentially data generators
from trainer.Trainer_Advent import Trainer_Advent # Inherit from Advent to leverage its discriminator setup if needed

# Import or define new utility functions/losses for your proposed method
from utils.proposed_utils import (
    calculate_self_adaptive_thresholds,
    calculate_prototype_loss,
    augment_for_consistency, # New augmentation function
    prob_to_entropy # Already exists in utils.utils_
)
from utils.proposed_losses import (
    consistency_loss_mse, # New consistency loss
    prototype_contrastive_loss # New prototype alignment loss
)


class Trainer_Proposed(Trainer_Advent): # Inherit from Trainer_Advent
    def __init__(self):
        super().__init__()
        # Initialize any new components or override existing ones
        # Example: Teacher model for Mean Teacher (optional, but good for consistency regularization)
        # self.teacher_segmentor = None # Initialized in prepare_model

        # Dictionary to store historical confidences for adaptive thresholding
        self.all_pixel_confidences = {c: [] for c in range(self.args.num_classes)}
        self.adaptive_thresholds = {c: self.args.confidence_threshold_initial for c in range(self.args.num_classes)}

    def add_additional_arguments(self):
        super().add_additional_arguments()
        # Add new hyperparameters specific to your proposed method
        self.parser.add_argument('-lambda_consistency', type=float, default=0.5,
                                 help='Weight for consistency regularization loss.')
        self.parser.add_argument('-lambda_proto', type=float, default=0.1,
                                 help='Weight for prototype-based feature alignment loss.')
        self.parser.add_argument('-alpha_ema', type=float, default=0.99,
                                 help='EMA decay rate for teacher model (if Mean Teacher is used).')
        self.parser.add_argument('-confidence_threshold_initial', type=float, default=0.8,
                                 help='Initial confidence threshold for pseudo-label filtering.')
        self.parser.add_argument('-history_len_conf', type=int, default=10000,
                                 help='Length of confidence history for adaptive thresholding.')
        self.parser.add_argument('-no_mean_teacher', action='store_true',
                                 help='Do not use Mean Teacher for pseudo-label generation and consistency.')
        # You can add more args for augmentation types, prototype loss variants, etc.

    def get_arguments_apdx(self):
        # Generate appendix for logging and checkpointing
        super().get_basic_arguments_apdx(name='Proposed') # Use a new unique name

        self.apdx += f".lcns{self.args.lambda_consistency}.lprt{self.args.lambda_proto}"
        if not self.args.no_mean_teacher:
            self.apdx += f".ema{self.args.alpha_ema}"
        self.apdx += f".cti{self.args.confidence_threshold_initial}"
        # Add other relevant arguments to the appendix for unique identification
        # self.apdx += f".bs{self.args.bs}" # Already added by basic_arguments_apdx

    def prepare_model(self):
        super().prepare_model() # This initializes self.segmentor

        if not self.args.no_mean_teacher:
            # Initialize teacher model for Mean Teacher setup
            # Make sure it's the same architecture as the student segmentor
            print("Initializing Mean Teacher model...")
            if self.args.backbone == 'unet':
                from model.unet_model import UNet
                self.teacher_segmentor = UNet(n_channels=3, n_classes=self.args.num_classes).to(self.device)
            elif self.args.backbone == 'drunet':
                from model.DRUNet import Segmentation_model as DR_UNet
                self.teacher_segmentor = DR_UNet(filters=self.args.filters, n_block=self.args.nb,
                                                 bottleneck_depth=self.args.bd,
                                                 n_class=self.args.num_classes, multilvl=self.args.multilvl,
                                                 args=self.args).to(self.device)
            elif self.args.backbone == 'deeplabv2':
                from model.deeplabv2 import get_deeplab_v2
                self.teacher_segmentor = get_deeplab_v2(num_classes=self.args.num_classes, multi_level=self.args.multilvl,
                                                        input_size=224).to(self.device)
            elif 'resnet' in self.args.backbone or 'efficientnet' in self.args.backbone or \
                 'mobilenet' in self.args.backbone or 'densenet' in self.args.backbone or \
                 'ception' in self.args.backbone or 'se_resnet' in self.args.backbone or 'skresnext' in self.args.backbone:
                from model.segmentation_models import segmentation_models
                self.teacher_segmentor = segmentation_models(name=self.args.backbone, pretrained=False,
                                                             decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                             classes=self.args.num_classes, multilvl=self.args.multilvl,
                                                             args=self.args).to(self.device)
            else:
                raise NotImplementedError(f"Unsupported backbone for teacher model: {self.args.backbone}")

            self.teacher_segmentor.load_state_dict(self.segmentor.state_dict())
            for param in self.teacher_segmentor.parameters():
                param.detach_() # Teacher is not updated by backprop
            self.teacher_segmentor.eval() # Teacher typically runs in eval mode

        # Override prepare_losses to set up new loss functions
        self.prepare_losses()

    def prepare_losses(self):
        # Override parent's prepare_losses
        self.seg_criterion = loss_calc # CrossEntropyLoss + Jaccard Loss
        # Consistency Loss: MSE between probabilities (or KL-divergence)
        # Assuming consistency_loss_mse is defined in utils.proposed_losses
        self.consistency_criterion = consistency_loss_mse

        # Prototype-Based Feature Alignment Loss
        # Assuming prototype_contrastive_loss is defined in utils.proposed_losses
        self.prototype_alignment_criterion = prototype_contrastive_loss

        # MSE Loss for CNR, if you still want to use it or adapt it
        self.mse_loss = torch.nn.MSELoss()

        # Initialize source centroid (if you still need a single source centroid)
        # This will need to be initialized based on the source data once, similar to Trainer_MPSCL
        try:
            source_modality = "bssfp" if "mscmrseg" in self.args.data_dir else "ct"
            centroid_filename = f'class_center_{source_modality}_f{self.args.fold}.npy'
            init_centroid_path = Path(centroid_filename)
            if not init_centroid_path.is_file():
                 project_dir_in_working = Path("/kaggle/working/-SLCL-Boundary/")
                 init_centroid_path = project_dir_in_working / centroid_filename
            print(f"Attempting to load initial source centroids from: {init_centroid_path.resolve()}")
            self.centroid_s = np.load(init_centroid_path)
            self.centroid_s = torch.from_numpy(self.centroid_s).float().to(self.device)
            print(f"Initial source centroids loaded successfully, shape: {self.centroid_s.shape}")
        except FileNotFoundError as e:
            print(f"WARNING: Initial source centroid file not found ({e}). Initializing with zeros.")
            # Fallback: Initialize with zeros if file not found. This might be problematic for early training.
            # You might need to determine feature dimension dynamically here (e.g., by running a dummy forward pass)
            # For now, let's assume a known feature dimension, e.g., 32 as seen in utils/utils_.py for DR_UNet
            dummy_feature_dim = 32 # This needs to match your model's feature extractor output
            self.centroid_s = torch.zeros(self.args.num_classes, dummy_feature_dim).float().to(self.device)
        except Exception as e:
            print(f"Error loading initial source centroids: {e}. Initializing with zeros.")
            dummy_feature_dim = 32
            self.centroid_s = torch.zeros(self.args.num_classes, dummy_feature_dim).float().to(self.device)


    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        self.segmentor.train()
        if not self.args.no_mean_teacher:
            self.teacher_segmentor.eval() # Teacher should be in eval mode during forward pass for consistency
        # If you keep discriminators, enable their training here
        # self.d_main.train()
        # if self.args.multilvl: self.d_aux.train()

        results = {}
        # source_domain_label = 1 # Not directly used for proposed method's core losses
        # target_domain_label = 0 # Not directly used for proposed method's core losses

        loss_seg_list = []
        loss_consistency_list = []
        loss_proto_list = []
        # if you retain discriminators for some reason, keep their loss lists
        # loss_adv_list, loss_dis_list = [], []

        # Iterate over both source and target loaders
        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt.zero_grad()
            # If you keep discriminators, zero their grads and set requires_grad to False
            # self.opt_d.zero_grad()
            # for param in self.d_main.parameters(): param.requires_grad = False

            # --- Source Domain Training (Supervised) ---
            img_s, labels_s, names_s = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                               labels_s.to(self.device, non_blocking=self.args.pin_memory)

            # Get segmentation logits and potentially features from the student model
            # Assuming segmentor returns (logits, bottleneck_features, decoder_features)
            out_s = self.segmentor(img_s, features_out=True)
            pred_s_logits, bottleneck_ft_s, dcdr_ft_s = out_s
            
            # Segmentation Loss
            loss_seg = self.seg_criterion(pred_s_logits, labels_s, self.device, jaccard=True)
            loss_seg_list.append(loss_seg.item())

            # Update source centroids (if still using for prototype-based alignment reference)
            # This is typically done with supervised labels
            self.centroid_s = update_class_center_iter(dcdr_ft_s, labels_s, self.centroid_s,
                                                       m=self.args.class_center_m, num_class=self.args.num_classes)

            # --- Target Domain Training (Unsupervised) ---
            img_t, labels_t_dummy, names_t = batch_style # labels_t_dummy are not used here as they are unlabeled
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            # 1. Self-Adaptive Thresholding for Pseudo-Label Refinement
            with torch.no_grad():
                # Get pseudo-labels from teacher or student
                if not self.args.no_mean_teacher:
                    # Teacher model also needs to return features for prototype generation
                    # This implies modifying segmentation_models.py (or DRUNet.py etc.) to always return features
                    # Or defining a feature_extractor method on your model
                    teacher_out = self.teacher_segmentor(img_t, features_out=True) # Assuming it returns features
                    pseudo_logits_t, teacher_bottleneck_ft_t, teacher_dcdr_ft_t = teacher_out
                else:
                    student_out_no_grad = self.segmentor(img_t, features_out=True)
                    pseudo_logits_t, teacher_bottleneck_ft_t, teacher_dcdr_ft_t = student_out_no_grad # Re-use names

                pseudo_probabilities_t = F.softmax(pseudo_logits_t, dim=1)
                max_confidences_t, pseudo_labels_hard_t = torch.max(pseudo_probabilities_t, dim=1)

                # Flatten for processing for adaptive thresholding
                flat_max_confidences = max_confidences_t.view(-1)
                flat_pseudo_labels_hard = pseudo_labels_hard_t.view(-1)

                # Update historical confidences
                for c in range(self.args.num_classes):
                    class_confidences = flat_max_confidences[flat_pseudo_labels_hard == c].cpu().numpy()
                    self.all_pixel_confidences[c].extend(class_confidences)
                    # Keep history size manageable
                    self.all_pixel_confidences[c] = self.all_pixel_confidences[c][-self.args.history_len_conf:]

                # Calculate adaptive thresholds
                self.adaptive_thresholds = calculate_self_adaptive_thresholds(
                    self.all_pixel_confidences,
                    self.args.confidence_threshold_initial,
                    self.args.num_classes
                )
                
                # Create a mask for high-confidence pseudo-labels
                reliable_mask = torch.zeros_like(flat_max_confidences, dtype=torch.bool, device=self.device)
                for c in range(self.args.num_classes):
                    threshold = self.adaptive_thresholds[c]
                    reliable_mask |= (flat_pseudo_labels_hard == c) & (flat_max_confidences >= threshold)

                # Filter pseudo-labels and corresponding features
                # The filtered_pseudo_labels are now pseudo-labels of high quality for target domain
                filtered_pseudo_labels_t = flat_pseudo_labels_hard[reliable_mask]
                filtered_dcdr_ft_t = teacher_dcdr_ft_t.permute(0, 2, 3, 1).reshape(-1, teacher_dcdr_ft_t.shape[1])[reliable_mask]


            # 2. Consistency Regularization for Robust Feature Learning
            # Generate two augmented views of target_images (student branch)
            aug_target_images_1 = augment_for_consistency(img_t, self.args.crop) # Need to implement this function
            aug_target_images_2 = augment_for_consistency(img_t, self.args.crop)

            # Get predictions/features from student model for consistency
            logits_aug1_student, _, _ = self.segmentor(aug_target_images_1, features_out=True)
            logits_aug2_student, _, _ = self.segmentor(aug_target_images_2, features_out=True)

            # Consistency loss uses teacher's prediction for one branch for robustness
            with torch.no_grad():
                # The teacher model should also be able to produce augmented views
                # Re-applying aug_target_images_1 to teacher for consistency target
                teacher_logits_aug1, _, _ = self.teacher_segmentor(aug_target_images_1, features_out=True)
                teacher_probs_aug1 = F.softmax(teacher_logits_aug1, dim=1)

            loss_consistency = self.consistency_criterion(F.softmax(logits_aug2_student, dim=1), teacher_probs_aug1)
            loss_consistency_list.append(loss_consistency.item())

            # 3. Prototype-Based Feature Alignment
            loss_proto = torch.tensor(0.0, device=self.device)
            if filtered_pseudo_labels_t.numel() > 0:
                # Calculate prototype loss using filtered target features and their high-confidence pseudo-labels
                # You might need to decide if you use source centroids (self.centroid_s) here,
                # or dynamically compute target centroids from filtered_dcdr_ft_t and filtered_pseudo_labels_t
                # The pseudocode implies calculating target prototypes from filtered_student_features
                # Let's adapt `calculate_prototype_loss` to compute its own prototypes.
                loss_proto = self.prototype_alignment_criterion(
                    filtered_dcdr_ft_t,
                    filtered_pseudo_labels_t,
                    self.args.num_classes
                )
            loss_proto_list.append(loss_proto.item())

            # --- Total Loss Calculation and Backpropagation ---
            total_unsupervised_loss = self.args.lambda_consistency * loss_consistency + \
                                      self.args.lambda_proto * loss_proto
            
            # Combine supervised and unsupervised losses
            total_loss = loss_seg + total_unsupervised_loss

            total_loss.backward()
            self.opt.step()

            # Update teacher model using EMA (if Mean Teacher is used)
            if not self.args.no_mean_teacher:
                self.update_teacher_model_ema()

            # If you keep discriminators, train them here.
            # This part is omitted for now, assuming your proposal replaces adversarial learning with consistency.
            # If you wish to keep domain adversarial on outputs, you'd integrate it here
            # similar to Trainer_Advent.

        # Aggregate results for logging
        results['seg_s'] = np.mean(loss_seg_list) if loss_seg_list else 0
        results['consistency_loss'] = np.mean(loss_consistency_list) if loss_consistency_list else 0
        results['proto_loss'] = np.mean(loss_proto_list) if loss_proto_list else 0
        # Add discriminator losses if they are kept
        # results['loss_dis'] = np.mean(loss_dis_list)

        return results

    def update_teacher_model_ema(self):
        """Updates teacher model parameters using EMA of student model."""
        if not self.args.no_mean_teacher and self.teacher_segmentor:
            for teacher_param, student_param in zip(self.teacher_segmentor.parameters(), self.segmentor.parameters()):
                teacher_param.data.mul_(self.args.alpha_ema).add_(student_param.data, alpha=1 - self.args.alpha_ema)

    def train(self):
        # The main training loop from Trainer_Advent is largely reusable.
        # You'll need to modify the summary writing part to reflect your new losses.
        super().train() # Calls the parent's train method, which calls train_epoch

        # Customize logging in the parent's train method's summary writing block
        # You'll need to open Trainer_Advent.py and modify its `train` method's `self.writer.add_scalars` calls
        # to include your new losses.