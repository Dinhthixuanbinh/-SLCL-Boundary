# %%writefile /kaggle/working/-SLCL-Boundary/trainer/Trainer_Proposed.py
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
from utils.lr_adjust import adjust_learning_rate
from utils.utils_ import update_class_center_iter, prob_2_entropy # prob_2_entropy imported from here
from utils import timer
import config # Import config to access new constants

# Import base Trainer class and potentially data generators
from trainer.Trainer_Advent import Trainer_Advent

# Import new utility functions/losses for your proposed method
from utils.proposed_utils import (
    calculate_self_adaptive_thresholds,
    calculate_prototype_loss_from_features, # Ensure this is imported for prototype loss
    augmentation_weak, # New weak augmentation function
    augmentation_strong, # New strong augmentation function
    sharpen_probabilities # New sharpening function
)
from utils.proposed_losses import (
    consistency_loss_mse,
    prototype_contrastive_loss
)


class Trainer_Proposed(Trainer_Advent):
    def __init__(self):
        super().__init__()
        self.all_pixel_confidences = {c: [] for c in range(self.args.num_classes)}
        self.adaptive_thresholds = {c: self.args.confidence_threshold_initial for c in range(self.args.num_classes)}

    def add_additional_arguments(self):
        super().add_additional_arguments()
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
        
        # New arguments for CR specific augmentations
        self.parser.add_argument('-aug_weak_mode', type=str, default=config.AUG_WEAK_MODE,
                                 help='Mode for weak augmentation in consistency regularization.')
        self.parser.add_argument('-aug_strong_mode', type=str, default=config.AUG_STRONG_MODE,
                                 help='Mode for strong augmentation in consistency regularization.')
        self.parser.add_argument('-sharpen_temp', type=float, default=config.SHARPENING_TEMPERATURE,
                                 help='Temperature for sharpening pseudo-labels for consistency.')

        # Existing MPSCL-specific arguments (keep them for compatibility as they are parsed)
        self.parser.add_argument('-src_temp', type=float, default=0.1, help='Source temperature for MPCL.')
        self.parser.add_argument('-src_base_temp', type=float, default=1, help='Source base temperature for MPCL.')
        self.parser.add_argument('-trg_temp', type=float, default=0.1, help='Target temperature for MPCL.')
        self.parser.add_argument('-trg_base_temp', type=float, default=1, help='Target base temperature for MPCL.')
        self.parser.add_argument('-src_margin', type=float, default=.4, help='Source margin for MPCL.')
        self.parser.add_argument('-trg_margin', type=float, default=.2, help='Target margin for MPCL.')
        self.parser.add_argument('-class_center_m', type=float, default=0.9, help='Momentum for updating class centers.')
        self.parser.add_argument('-pixel_sel_th', type=float, default=.25, help='Pixel selection threshold for pseudo-labeling.')
        self.parser.add_argument('-w_mpcl_s', type=float, default=1.0, help='Weight for source MPCL.')
        self.parser.add_argument('-w_mpcl_t', type=float, default=1.0, help='Weight for target MPCL.')
        self.parser.add_argument('-dis_type', type=str, default='origin', help='Discriminator type (if still used).')
        self.parser.add_argument('-part', type=int, default=1, help='number of partitions for rMC (set to 2 via cmd/hardcode).')
        self.parser.add_argument('-CNR_w', type=float, default=0.0, help='Weight for CNR loss (set via cmd/hardcode).')
    def prepare_model(self):
        super().prepare_model() # This initializes self.segmentor

        # Initialize teacher_segmentor to None first, so the attribute always exists
        self.teacher_segmentor = None

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
        # The parent's prepare_losses (in Trainer_Advent) doesn't define seg_criterion,
        # so you need to explicitly define it here.
        # super().prepare_losses() # You can call this if parent has relevant init, but for seg_criterion, you define it.

        self.seg_criterion = loss_calc # CrossEntropyLoss + Jaccard Loss 
        # Consistency Loss: MSE between probabilities (or KL-divergence)
        self.consistency_criterion = consistency_loss_mse

        # Prototype-Based Feature Alignment Loss
        self.prototype_alignment_criterion = prototype_contrastive_loss

        # MSE Loss for CNR, if you still want to use it or adapt it
        self.mse_loss = torch.nn.MSELoss()

        # Initialize source centroid (if you still need a single source centroid)
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



    def get_arguments_apdx(self):
        super().get_basic_arguments_apdx(name='Proposed_CR') # New unique name for logging

        self.apdx += f".lcns{self.args.lambda_consistency}.lprt{self.args.lambda_proto}"
        if not self.args.no_mean_teacher:
            self.apdx += f".ema{self.args.alpha_ema}"
        self.apdx += f".cti{self.args.confidence_threshold_initial}"
        
        # Add new CR specific aug modes to appendix
        self.apdx += f".weakaug{self.args.aug_weak_mode}.strongaug{self.args.aug_strong_mode}"
        self.apdx += f".sharpT{self.args.sharpen_temp}"
        # self.apdx += f".bs{self.args.bs}" # Already added by basic_arguments_apdx


    # prepare_model, prepare_losses remain largely the same as previous steps,
    # just ensure they use the newly added arguments/functions.
    # The `prepare_losses` method is already defined correctly to set up consistency criterion.

    def train_epoch(self, epoch):
        print(f'start to train epoch: {epoch}')
        self.segmentor.train()
        if not self.args.no_mean_teacher:
            self.teacher_segmentor.eval()

        results = {}
        loss_seg_list = []
        loss_consistency_list = []
        loss_proto_list = []

        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt.zero_grad()

            # --- Source Domain Training (Supervised) ---
            img_s, labels_s, names_s = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                               labels_s.to(self.device, non_blocking=self.args.pin_memory)

            out_s = self.segmentor(img_s, features_out=True)
            pred_s_logits, bottleneck_ft_s, dcdr_ft_s = out_s
            
            loss_seg = self.seg_criterion(pred_s_logits, labels_s, self.device, jaccard=True)
            loss_seg_list.append(loss_seg.item())

            self.centroid_s = update_class_center_iter(dcdr_ft_s, labels_s, self.centroid_s,
                                                       m=self.args.class_center_m, num_class=self.args.num_classes)

            # --- Target Domain Training (Unsupervised) ---
            img_t, labels_t_dummy, names_t = batch_style # labels_t_dummy are not used here as they are unlabeled
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            # 1. Consistency Regularization (CR) - Pseudocode Implementation
            # Apply different augmentations to the same unlabeled image
            image_weak_aug = augmentation_weak(img_t, self.args.crop, self.args.aug_weak_mode)
            image_strong_aug = augmentation_strong(img_t, self.args.crop, self.args.aug_strong_mode)

            # Get predictions/features for consistency
            # It's common to use the Teacher for the 'pseudo_label_for_consistency' (weak branch)
            # and the Student for the 'prediction_strong' (strong branch).
            with torch.no_grad():
                # Get pseudo_label_for_consistency from weak augmentation using Teacher (for stability)
                # Ensure teacher_segmentor returns logits
                pseudo_logits_weak_teacher, _, _ = self.teacher_segmentor(image_weak_aug, features_out=True)
                pseudo_probabilities_weak_teacher = F.softmax(pseudo_logits_weak_teacher, dim=1)
                
                # Optional: Sharpen the teacher's weak-augmented prediction to make it a better pseudo-label
                pseudo_label_for_consistency = sharpen_probabilities(
                    pseudo_probabilities_weak_teacher, 
                    self.args.sharpen_temp
                )
                
                # (Self-Adaptive Thresholding part from previous step is still here, but can be integrated
                # for *masking* consistency loss if needed, not just for prototype loss)
                # Here, for SAT, we use the teacher's prediction from the *original* img_t for reliable pseudo-labels for *prototype alignment*.
                # This SAT part is *not* directly for consistency regularization's pseudo-label, which usually relies on weak aug + sharpening.
                # However, you could use a mask derived from SAT to mask the consistency loss.
                
                # Original SAT for Prototype Alignment (retained here):
                pseudo_logits_t_original, _, teacher_dcdr_ft_t = self.teacher_segmentor(img_t, features_out=True)
                pseudo_probabilities_t_original = F.softmax(pseudo_logits_t_original, dim=1)
                max_confidences_t_original, pseudo_labels_hard_t_original = torch.max(pseudo_probabilities_t_original, dim=1)

                flat_max_confidences = max_confidences_t_original.view(-1)
                flat_pseudo_labels_hard = pseudo_labels_hard_t_original.view(-1)

                for c in range(self.args.num_classes):
                    class_confidences = flat_max_confidences[flat_pseudo_labels_hard == c].cpu().numpy()
                    self.all_pixel_confidences[c].extend(class_confidences)
                    self.all_pixel_confidences[c] = self.all_pixel_confidences[c][-self.args.history_len_conf:]

                self.adaptive_thresholds = calculate_self_adaptive_thresholds(
                    self.all_pixel_confidences,
                    self.args.confidence_threshold_initial,
                    self.args.num_classes
                )
                
                reliable_mask_for_proto = torch.zeros_like(flat_max_confidences, dtype=torch.bool, device=self.device)
                for c in range(self.args.num_classes):
                    threshold = self.adaptive_thresholds[c]
                    reliable_mask_for_proto |= (flat_pseudo_labels_hard == c) & (flat_max_confidences >= threshold)

                filtered_pseudo_labels_t = flat_pseudo_labels_hard[reliable_mask_for_proto]
                filtered_dcdr_ft_t = teacher_dcdr_ft_t.permute(0, 2, 3, 1).reshape(-1, teacher_dcdr_ft_t.shape[1])[reliable_mask_for_proto]

            # Get prediction from student model for strong augmented view
            prediction_strong_logits, _, _ = self.segmentor(image_strong_aug, features_out=True)

            # Calculate the consistency loss
            # You might want to mask this loss based on confidence from pseudo_label_for_consistency
            # e.g., if pseudo_label_for_consistency_mask = (max_confidences_weak_teacher >= X)
            # then consistency_loss = F.mse_loss(..., reduction='none')[pseudo_label_for_consistency_mask].mean()
            loss_consistency = self.consistency_criterion(
                F.softmax(prediction_strong_logits, dim=1), # Student's strong aug prediction (probabilities)
                pseudo_label_for_consistency.detach() # Teacher's weak aug sharpened prediction (probabilities, detached)
            )
            loss_consistency_list.append(loss_consistency.item())

            # 2. Prototype-Based Feature Alignment (Retained from previous step)
            loss_proto = torch.tensor(0.0, device=self.device)
            if filtered_pseudo_labels_t.numel() > 0:
                loss_proto = self.prototype_alignment_criterion(
                    filtered_dcdr_ft_t,
                    filtered_pseudo_labels_t,
                    self.args.num_classes
                )
            loss_proto_list.append(loss_proto.item())

            # --- Total Loss Calculation and Backpropagation ---
            total_unsupervised_loss = self.args.lambda_consistency * loss_consistency + \
                                      self.args.lambda_proto * loss_proto
            
            total_loss = loss_seg + total_unsupervised_loss

            total_loss.backward()
            self.opt.step()

            # Update teacher model using EMA (if Mean Teacher is used)
            if not self.args.no_mean_teacher:
                self.update_teacher_model_ema()

        # Aggregate results for logging
        results['seg_s'] = np.mean(loss_seg_list) if loss_seg_list else 0
        results['consistency_loss'] = np.mean(loss_consistency_list) if loss_consistency_list else 0
        results['proto_loss'] = np.mean(loss_proto_list) if loss_proto_list else 0

        return results

    # The train method remains the same as in the previous step, handling logging and checkpoints.
    def train(self):
        """
        Main training loop, customized for the proposed method.
        """
        # (This part is copied and modified from Trainer_Advent.train)

        print('start to train')
        print("Evaluator created.")

        """mkdir for the stylized images (if still relevant)"""
        # This might be specific to previous methods, keep if needed.
        if not os.path.exists(self.args.style_dir):
            os.makedirs(self.args.style_dir)

        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training Epochs"):
            epoch_start = datetime.now()
            """adjust learning rate with polynomial decay"""
            self.adjust_lr(epoch) # This adjusts segmentor and discriminator LRs if discriminators are still present

            train_results = self.train_epoch(epoch) # Your custom train_epoch

            # --- Evaluation ---
            # Call eval for validation
            results_valid = self.eval(modality='target', phase='valid', toprint=False)
            # Ensure proper indexing for Dice scores for your num_classes.
            # Assuming Dice for MYO, LV, RV are relevant (indices 0, 2, 4 in results['dc'] for 3 classes + background)
            # Adjust if your num_classes is different or if order changes.
            if self.args.num_classes == 4: # Assuming BG, MYO, LV, RV
                valid_dice_scores = [results_valid['dc'][0], results_valid['dc'][2], results_valid['dc'][4]] # MYO, LV, RV
            else: # Fallback or adjust based on your specific classes
                 valid_dice_scores = [results_valid['dc'][i] for i in range(self.args.num_classes -1)] # Assuming 1st is BG, rest are foreground
            
            # Filter out potential -1 values (for classes not found or HD/ASD not calculated)
            valid_dice_scores = [score for score in valid_dice_scores if score != -1]
            lge_dice = np.nanmean(valid_dice_scores) if valid_dice_scores else 0.0
            lge_dice = np.round(lge_dice, 4)

            # Optional: Test evaluation
            test_dice_str = ""
            if self.args.evalT:
                results_test = self.eval(modality='target', phase='test', toprint=False)
                if self.args.num_classes == 4:
                    test_dice_scores = [results_test['dc'][0], results_test['dc'][2], results_test['dc'][4]]
                else:
                    test_dice_scores = [results_test['dc'][i] for i in range(self.args.num_classes -1)]

                test_dice_scores = [score for score in test_dice_scores if score != -1]
                lge_dice_test = np.nanmean(test_dice_scores) if test_dice_scores else 0.0
                lge_dice_test = np.round(lge_dice_test, 4)
                test_dice_str = f", Test Dice: {lge_dice_test:.4f}"

            # --- Logging (TensorBoard and Print) ---
            print("\nWriting summary...")
            if self.args.evalT:
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)

            # Log losses specific to your proposed method
            self.writer.add_scalar('Loss/Seg_Source', train_results['seg_s'], epoch + 1)
            self.writer.add_scalar('Loss/Consistency', train_results['consistency_loss'], epoch + 1)
            self.writer.add_scalar('Loss/Prototype_Alignment', train_results['proto_loss'], epoch + 1)
            
            # Log Learning Rates
            self.writer.add_scalars('LR', {'Segmentor': self.opt.param_groups[0]['lr']}, epoch + 1)
            # If you still have discriminator optimizers and adjust their LR, add them:
            # if hasattr(self, 'opt_d') and self.opt_d:
            #     self.writer.add_scalars('LR', {'Discriminator': self.opt_d.param_groups[0]['lr']}, epoch + 1)
            # if hasattr(self, 'opt_d_aux') and self.opt_d_aux:
            #     self.writer.add_scalars('LR', {'Discriminator_Aux': self.opt_d_aux.param_groups[0]['lr']}, epoch + 1)


            # Print epoch summary
            message = (f'\nEpoch = {epoch + 1:4d}/{self.args.epochs:4d} | '
                       f'LR={self.opt.param_groups[0]["lr"]:.2e} | '
                       f'Seg S={train_results["seg_s"]:.4f} | '
                       f'Cons={train_results["consistency_loss"]:.4f} | '
                       f'Proto={train_results["proto_loss"]:.4f} | '
                       f'Val Dice={lge_dice:.4f}{test_dice_str}')
            print(message)

            # --- Checkpointing and Early Stopping ---
            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt, tobreak=tobreak)
            
            # If you have discriminators and want to checkpoint them:
            # if hasattr(self, 'modelcheckpoint_d') and self.modelcheckpoint_d:
            #     self.modelcheckpoint_d.step(monitor=lge_dice, model=self.d_main, epoch=epoch + 1,
            #                                 optimizer=self.opt_d, tobreak=tobreak)
            # if hasattr(self, 'modelcheckpoint_d_aux') and self.modelcheckpoint_d_aux:
            #     self.modelcheckpoint_d_aux.step(monitor=lge_dice, model=self.d_aux, epoch=epoch + 1,
            #                                     optimizer=self.opt_d_aux, tobreak=tobreak)

            if tobreak:
                print(f"Stopping training at epoch {epoch+1} due to early stopping or time limit.")
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result

        # --- MODIFIED LOG RENAMING ---
        if hasattr(self, 'log_dir') and isinstance(self.log_dir, str):
            current_log_path = Path(self.log_dir)
            if current_log_path.exists():
                try:
                    log_dir_new_name = '{}.e{}.Scr{:.4f}'.format(self.apdx, best_epoch, best_score)
                    log_dir_new = current_log_path.parent / log_dir_new_name
                    os.rename(current_log_path, log_dir_new)
                    print(f"Renamed log directory to: {log_dir_new}")
                except OSError as e:
                    print(f"Error renaming log directory from {current_log_path} to {log_dir_new_name}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred during log renaming: {e}")
            else:
                print(f"Log directory path does not exist: {current_log_path}")
        elif hasattr(self, 'log_dir'):
            print(f"Log directory ('self.log_dir') is not a string. Type: {type(self.log_dir)}. Skipping rename.")
        else:
            print("Log directory ('self.log_dir') attribute not found, skipping rename.")

        # --- Final Evaluation ---
        model_name = self.mcp_segmentor.best_model_save_dir
        print(f"\nLoading best model from: {model_name}")
        if model_name and os.path.exists(model_name):
            try:
                checkpoint = torch.load(model_name)
                if 'model_state_dict' in checkpoint:
                    self.segmentor.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.segmentor.load_state_dict(checkpoint)
                print("Best model loaded successfully for final evaluation.")
                print("\n--- Final Test Evaluation (Target Domain) ---")
                self.eval(modality='target', phase='test', toprint=True)
                print("\n--- Final Test Evaluation (Source Domain) ---")
                self.eval(modality='source', phase='test', toprint=True)
            except Exception as e:
                print(f"Error loading best model weights for final evaluation: {e}")
        else:
            print("Best model checkpoint not found or not saved. Skipping final evaluation.")

        return