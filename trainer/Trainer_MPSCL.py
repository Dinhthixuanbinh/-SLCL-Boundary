import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
from pathlib import Path

"""torch import"""
import torch
import torch.nn.functional as F
import torch.nn as nn

"""utils import"""
from utils.loss import loss_calc, MPCL, dice_loss, mpcl_loss_calc, jaccard_loss, ContrastiveLoss
from utils.utils_ import update_class_center_iter, generate_pseudo_label, prob_2_entropy, cal_centroid, calculate_juq_reliability_map
from utils import timer 
"""evaluator import"""
from evaluator import Evaluator

"""trainer import"""
from trainer.Trainer_Advent import Trainer_Advent
import config

class Trainer_MPSCL(Trainer_Advent):
    def __init__(self):
        super().__init__()
        self.teacher_segmentor = None
        self.teacher_optimizer = None
        self.ema_momentum = 0.99

    def add_additional_arguments(self):
        super(Trainer_MPSCL, self).add_additional_arguments()
        self.parser.add_argument('-adjust_lr', action='store_true')

        self.parser.add_argument('-src_temp', type=float, default=0.1)
        self.parser.add_argument('-src_base_temp', type=float, default=1)
        self.parser.add_argument('-trg_temp', type=float, default=0.07)
        self.parser.add_argument('-trg_base_temp', type=float, default=1)
        self.parser.add_argument('-src_margin', type=float, default=.4)
        self.parser.add_argument('-trg_margin', type=float, default=.2)

        self.parser.add_argument('-class_center_m', type=float, default=0.9)
        self.parser.add_argument('-pixel_sel_th', type=float, default=.25)

        self.parser.add_argument('-w_mpcl_s', type=float, default=1.0)
        self.parser.add_argument('-w_mpcl_t', type=float, default=1.0)

        self.parser.add_argument('-dis_type', type=str, default='origin')

        self.parser.add_argument('-w_cl', type=float, default=0.6, help='Weight for Contrastive Loss (L_CL)')
        self.parser.add_argument('-w_cnr', type=float, default=0.05, help='Weight for Centroid Norm Regularizer Loss (L_CNR)')
        self.parser.add_argument('-juq_temp', type=float, default=1.0, help='Temperature for JUQ entropy calculation')
        self.parser.add_argument('-juq_thd', type=float, default=0.8, help='Threshold for JUQ reliability map (e.g., 0.8 for 80% certainty)')
        self.parser.add_argument('-eta_1', type=float, default=0.6, help='Weight for entropy uncertainty in JUQ')
        self.parser.add_argument('-eta_2', type=float, default=0.4, help='Weight for distributional uncertainty in JUQ')
        self.parser.add_argument('-momentum_teacher', type=float, default=0.99, help='EMA momentum for teacher network update')

        if '-part' not in self.parser._option_string_actions:
             self.parser.add_argument('-part', type=int, default=1,
                                      help='number of partitions for rMC (set to 2 via cmd/hardcode)')
        if '-CNR_w' not in self.parser._option_string_actions:
             self.parser.add_argument('-CNR_w', type=float, default=0.0,
                                      help='Weight for CNR loss (set via cmd/hardcode)')

        self.parser.add_argument('-min_reliability_for_rmc', type=float, default=0.5, help='Minimum reliability for rMC partitioning')

    def get_arguments_apdx(self):
        super(Trainer_MPSCL, self).get_basic_arguments_apdx(name='JUQ_UDA')
        self.apdx += f".bs{self.args.bs}"
        self.apdx += f".lr_dis{self.args.lr_dis}.w_dis{self.args.w_dis}"
        if self.args.multilvl:
            self.apdx += f'.w_d_aux{self.args.w_dis_aux}'
        self.apdx += f'.w_mpscl_s{self.args.w_mpcl_s}.t{self.args.w_mpcl_t}'
        
        self.apdx += f'.w_cl{self.args.w_cl}.w_cnr{self.args.w_cnr}'
        self.apdx += f'.juq_t{self.args.juq_temp}.juq_th{self.args.juq_thd}'
        self.apdx += f'.eta1{self.args.eta_1}.eta2{self.args.eta_2}'
        self.apdx += f'.ema{self.args.momentum_teacher}'

        part_num = getattr(self.args, 'part', 1)
        cnr_weight = getattr(self.args, 'CNR_w', 0.0)
        if part_num > 1:
             self.apdx += f'.p{part_num}'
        if cnr_weight > 0:
             self.apdx += f'.cnr{cnr_weight}'
             

    @timer.timeit
    def prepare_model(self):
        super(Trainer_MPSCL, self).prepare_model()
        from model.DRUNet import Segmentation_model as DR_UNet
        if self.args.backbone == 'drunet':
            self.teacher_segmentor = DR_UNet(filters=self.args.filters, n_block=self.args.nb, 
                                            bottleneck_depth=self.args.bd, n_class=self.args.num_classes, 
                                            multilvl=self.args.multilvl, args=self.args).to(self.device)
        elif 'resnet' in self.args.backbone or 'efficientnet' in self.args.backbone or \
                'mobilenet' in self.args.backbone or 'densenet' in self.args.backbone or 'ception' in self.args.backbone or \
                'se_resnet' in self.args.backbone or 'skresnext' in self.args.backbone:
            from model.segmentation_models import segmentation_models
            self.teacher_segmentor = segmentation_models(name=self.args.backbone, pretrained=False,
                                                 decoder_channels=(512, 256, 128, 64, 32), in_channel=3,
                                                 classes=4, multilvl=self.args.multilvl, args=self.args).to(self.device)
        else:
            raise NotImplementedError("Teacher network backbone not implemented.")
            
        self.teacher_segmentor.load_state_dict(self.segmentor.state_dict())
        for param in self.teacher_segmentor.parameters():
            param.requires_grad = False

    def prepare_losses(self):
        super().prepare_losses()
        self.criterion_cl = ContrastiveLoss(tau=self.args.trg_temp, n_class=self.args.num_classes, norm=True)
        self.mse_loss = nn.MSELoss()

    def _update_teacher_network(self):
        for teacher_param, student_param in zip(self.teacher_segmentor.parameters(), self.segmentor.parameters()):
            teacher_param.data.mul_(self.args.momentum_teacher).add_(student_param.data, alpha=1 - self.args.momentum_teacher)

    def train_epoch(self, epoch):
        self.segmentor.train()
        self.teacher_segmentor.eval()
        self.d_main.train()
        if self.args.multilvl:
             self.d_aux.train()

        results = {}
        source_domain_label = 1
        target_domain_label = 0
        
        loss_seg_list, loss_seg_aux_list = [], []
        loss_adv_list, loss_adv_aux_list, loss_dis_list, loss_dis_aux_list = [], [], [], []
        loss_cl_list, loss_cnr_list = [], []
        
        d_acc_s, d_acc_t = [], []
        d_aux_acc_s, d_aux_acc_t = [], []

        partition_num = getattr(self.args, 'part', 1)
        cnr_weight = getattr(self.args, 'CNR_w', 0.0)

        min_reliability_for_rmc = self.args.min_reliability_for_rmc

        for batch_content, batch_style in zip(self.content_loader, self.style_loader):
            self.opt_d.zero_grad()
            self.opt.zero_grad()
            if self.args.multilvl:
                self.opt_d_aux.zero_grad()

            for param in self.d_main.parameters():
                param.requires_grad = False
            if self.args.multilvl:
                for param in self.d_aux.parameters():
                    param.requires_grad = False

            img_s, labels_s, names = batch_content
            img_s, labels_s = img_s.to(self.device, non_blocking=self.args.pin_memory), \
                              labels_s.to(self.device, non_blocking=self.args.pin_memory)
            img_t, _, namet = batch_style
            img_t = img_t.to(self.device, non_blocking=self.args.pin_memory)

            pred_s_main, pred_s_aux, dcdr_ft_s = self.segmentor(img_s, features_out=True)
            
            loss_seg = loss_calc(pred_s_main, labels_s, self.device, jaccard=True)
            loss_seg_list.append(loss_seg.item())
            
            if self.args.multilvl:
                loss_seg_aux = loss_calc(pred_s_aux, labels_s, self.device, jaccard=True)
                loss_seg_aux_list.append(loss_seg_aux.item())
                loss_seg += self.args.w_seg_aux * loss_seg_aux

            with torch.no_grad():
                teacher_pred_t_main, teacher_pred_t_aux, teacher_dcdr_ft_t = self.teacher_segmentor(img_t, features_out=True)
                teacher_pred_t_softmax_main = F.softmax(teacher_pred_t_main, dim=1)

                reliability_map_continuous = calculate_juq_reliability_map(
                    teacher_pred_t_softmax_main, self.args.eta_1, self.args.eta_2
                )

                reliable_pseudo_labels_soft = teacher_pred_t_softmax_main * reliability_map_continuous

                student_pred_t_main, student_pred_t_aux, student_dcdr_ft_t = self.segmentor(img_t, features_out=True)

            self.centroid_s, _, _ = cal_centroid(dcdr_ft_s.detach(), labels_s, previous_centroid=self.centroid_s, momentum=self.args.class_center_m, n_class=self.args.num_classes, reliability_map=None)

            reliable_pixel_mask = (reliability_map_continuous.squeeze(1) >= min_reliability_for_rmc)
            
            reliable_indices = torch.nonzero(reliable_pixel_mask, as_tuple=True)
            
            num_reliable_pixels = reliable_indices[0].shape[0]
            target_sub_centroid_1 = torch.zeros(self.args.num_classes, student_dcdr_ft_t.shape[1], device=self.device)
            target_sub_centroid_2 = torch.zeros(self.args.num_classes, student_dcdr_ft_t.shape[1], device=self.device)
            
            if num_reliable_pixels > 0:
                permuted_indices = torch.randperm(num_reliable_pixels, device=self.device)
                split_point = num_reliable_pixels // 2
                
                partition1_indices_flat = permuted_indices[:split_point]
                partition2_indices_flat = permuted_indices[split_point:]

                partition_mask1 = torch.zeros_like(reliable_pixel_mask, dtype=torch.bool)
                partition_mask2 = torch.zeros_like(reliable_pixel_mask, dtype=torch.bool)

                for i_dim in range(len(reliable_indices)):
                    if i_dim == 0:
                        partition_mask1[reliable_indices[i_dim][partition1_indices_flat]] = True
                        partition_mask2[reliable_indices[i_dim][partition2_indices_flat]] = True
                    else:
                        partition_mask1[:, reliable_indices[i_dim][partition1_indices_flat]] = True
                        partition_mask2[:, reliable_indices[i_dim][partition2_indices_flat]] = True

                flattened_student_ft = student_dcdr_ft_t.permute(0, 2, 3, 1).reshape(-1, student_dcdr_ft_t.shape[1])
                flattened_reliable_pl = reliable_pseudo_labels_soft.permute(0, 2, 3, 1).reshape(-1, self.args.num_classes)
                flattened_reliability_mask = reliability_map_continuous.flatten()
                
                filtered_student_ft = flattened_student_ft[flattened_reliability_mask.bool()]
                filtered_reliable_pl = flattened_reliable_pl[flattened_reliability_mask.bool()]
                
                num_filtered_pixels = filtered_student_ft.shape[0]
                if num_filtered_pixels > 0:
                    permuted_indices_filtered = torch.randperm(num_filtered_pixels, device=self.device)
                    split_point_filtered = num_filtered_pixels // 2
                    
                    partition1_ft = filtered_student_ft[permuted_indices_filtered[:split_point_filtered]]
                    partition1_pl = filtered_reliable_pl[permuted_indices_filtered[:split_point_filtered]]

                    partition2_ft = filtered_student_ft[permuted_indices_filtered[split_point_filtered:]]
                    partition2_pl = filtered_reliable_pl[permuted_indices_filtered[split_point_filtered:]]

                    def calculate_partition_centroid(features, pseudo_labels_softmax, num_classes):
                        centroids = []
                        for k in range(num_classes):
                            weighted_features_k = features * pseudo_labels_softmax[:, k].unsqueeze(1)
                            sum_weights_k = torch.sum(pseudo_labels_softmax[:, k])
                            
                            if sum_weights_k > 0:
                                centroid_k = torch.sum(weighted_features_k, dim=0) / sum_weights_k
                            else:
                                centroid_k = torch.zeros(features.shape[1], device=features.device)
                            centroids.append(centroid_k)
                        return torch.stack(centroids, dim=0)
                    
                    target_sub_centroid_1 = calculate_partition_centroid(partition1_ft, partition1_pl, self.args.num_classes)
                    target_sub_centroid_2 = calculate_partition_centroid(partition2_ft, partition2_pl, self.args.num_classes)
                else:
                    target_sub_centroid_1 = torch.zeros(self.args.num_classes, student_dcdr_ft_t.shape[1], device=self.device)
                    target_sub_centroid_2 = torch.zeros(self.args.num_classes, student_dcdr_ft_t.shape[1], device=self.device)
                    
            loss_cl_1 = self.criterion_cl(self.centroid_s[1:], target_sub_centroid_1[1:], bg=False)
            loss_cl_2 = self.criterion_cl(self.centroid_s[1:], target_sub_centroid_2[1:], bg=False)
            total_loss_cl = (loss_cl_1 + loss_cl_2) / 2.0
            total_loss_cl = loss_cl_1 + loss_cl_2
            loss_cl_list.append(total_loss_cl.item())

            centroid_s_norm = torch.norm(self.centroid_s[1:], p=2, dim=1)
            centroid_t1_norm = torch.norm(target_sub_centroid_1[1:], p=2, dim=1)
            centroid_t2_norm = torch.norm(target_sub_centroid_2[1:], p=2, dim=1)

            loss_cnr_1 = self.mse_loss(centroid_t1_norm, centroid_s_norm.detach())
            loss_cnr_2 = self.mse_loss(centroid_t2_norm, centroid_s_norm.detach())
            total_loss_cnr = (loss_cnr_1 + loss_cnr_2) / 2.0
            total_loss_cnr = loss_cnr_1 + loss_cnr_2
            loss_cnr_list.append(total_loss_cnr.item())

            pred_t_softmax_student = F.softmax(student_pred_t_main, dim=1)
            uncertainty_map_student_T = prob_2_entropy(pred_t_softmax_student)
            D_out_main_student = self.d_main(uncertainty_map_student_T)
            loss_adv_main = F.binary_cross_entropy_with_logits(D_out_main_student, torch.FloatTensor(
                D_out_main_student.data.size()).fill_(source_domain_label).to(self.device))
            loss_adv_list.append(loss_adv_main.item())

            total_generator_loss = (loss_seg 
                                   + self.args.w_cl * total_loss_cl 
                                   + self.args.w_cnr * total_loss_cnr
                                   + self.args.w_dis * loss_adv_main
                                   )

            if self.args.multilvl:
                pred_t_softmax_aux_student = F.softmax(student_pred_t_aux, dim=1)
                uncertainty_map_aux_student_T = prob_2_entropy(pred_t_softmax_aux_student)
                D_out_aux_student = self.d_aux(uncertainty_map_aux_student_T)
                loss_adv_aux = F.binary_cross_entropy_with_logits(D_out_aux_student, torch.FloatTensor(
                    D_out_aux_student.data.size()).fill_(source_domain_label).to(self.device))
                loss_adv_aux_list.append(loss_adv_aux.item())
                total_generator_loss += self.args.w_dis_aux * loss_adv_aux

            total_generator_loss.backward()
            self.opt.step()

            for param in self.d_main.parameters():
                param.requires_grad = True
            if self.args.multilvl:
                for param in self.d_aux.parameters():
                    param.requires_grad = True

            pred_s_main_d = pred_s_main.detach()
            d_out_main_s = self.d_main(prob_2_entropy(F.softmax(pred_s_main_d, dim=1)))
            loss_d_main_s = F.binary_cross_entropy_with_logits(d_out_main_s, torch.FloatTensor(
                d_out_main_s.data.size()).fill_(source_domain_label).to(self.device))
            loss_d_main_s = loss_d_main_s / 2.0
            loss_d_main_s.backward()
            D_out_s_main = torch.sigmoid(d_out_main_s.detach()).cpu().numpy()
            D_out_s_main = np.where(D_out_s_main >= .5, 1, 0)
            d_acc_s.append(np.mean(D_out_s_main))

            pred_t_main_d = student_pred_t_main.detach()
            d_out_main_t = self.d_main(prob_2_entropy(F.softmax(pred_t_main_d, dim=1)))
            loss_d_main_t = F.binary_cross_entropy_with_logits(d_out_main_t, torch.FloatTensor(
                d_out_main_t.data.size()).fill_(target_domain_label).to(self.device))
            loss_d_main_t = loss_d_main_t / 2.0
            loss_d_main_t.backward()
            loss_dis_list.append(loss_d_main_s.item() + loss_d_main_t.item())
            D_out_t_main = torch.sigmoid(d_out_main_t.detach()).cpu().numpy()
            D_out_t_main = np.where(D_out_t_main >= .5, 1, 0)
            d_acc_t.append(1 - np.mean(D_out_t_main))
            self.opt_d.step()

            if self.args.multilvl:
                pred_s_aux_d = pred_s_aux.detach()
                d_out_aux_s = self.d_aux(prob_2_entropy(F.softmax(pred_s_aux_d, dim=1)))
                loss_d_aux_s = F.binary_cross_entropy_with_logits(d_out_aux_s, torch.FloatTensor(
                    d_out_aux_s.data.size()).fill_(source_domain_label).to(self.device))
                loss_d_aux_s = loss_d_aux_s / 2.0
                loss_d_aux_s.backward()
                D_out_s_aux = torch.sigmoid(d_out_aux_s.detach()).cpu().numpy()
                D_out_s_aux = np.where(D_out_s_aux >= .5, 1, 0)
                d_aux_acc_s.append(np.mean(D_out_s_aux))

                pred_t_aux_d = student_pred_t_aux.detach()
                d_out_aux_t = self.d_aux(prob_2_entropy(F.softmax(pred_t_aux_d, dim=1)))
                loss_d_aux_t = F.binary_cross_entropy_with_logits(d_out_aux_t, torch.FloatTensor(
                    d_out_aux_t.data.size()).fill_(target_domain_label).to(self.device))
                loss_d_aux_t = loss_d_aux_t / 2.0
                loss_d_aux_t.backward()
                loss_dis_aux_list.append(loss_d_aux_s.item() + loss_d_aux_t.item())
                D_out_t_aux = torch.sigmoid(d_out_aux_t.detach()).cpu().numpy()
                D_out_t_aux = np.where(D_out_t_aux >= .5, 1, 0)
                d_aux_acc_t.append(1 - np.mean(D_out_t_aux))
                self.opt_d_aux.step()

            self._update_teacher_network()

        results['seg_s'] = np.mean(loss_seg_list) if loss_seg_list else 0
        results['seg_s_aux'] = np.mean(loss_seg_aux_list) if loss_seg_aux_list else 0
        results['loss_cl'] = np.mean(loss_cl_list) if loss_cl_list else 0
        results['loss_cnr'] = np.mean(loss_cnr_list) if loss_cnr_list else 0
        
        results['dis_acc_s'] = np.mean(d_acc_s) if d_acc_s else 0
        results['dis_acc_t'] = np.mean(d_acc_t) if d_acc_t else 0
        results['loss_adv'] = np.mean(loss_adv_list) if loss_adv_list else 0
        results['loss_dis'] = np.mean(loss_dis_list) if loss_dis_list else 0
        
        results['dis_aux_acc_s'] = np.mean(d_aux_acc_s) if d_aux_acc_s else 0
        results['dis_aux_acc_t'] = np.mean(d_aux_acc_t) if d_aux_acc_t else 0
        results['loss_adv_aux'] = np.mean(loss_adv_aux_list) if loss_adv_aux_list else 0
        results['loss_dis_aux'] = np.mean(loss_dis_aux_list) if loss_dis_aux_list else 0
        
        return results

    def train(self):
        try:
            source_modality = "bssfp" if "mscmrseg" in self.args.data_dir else "ct"
            centroid_filename = f'class_center_{source_modality}_f{self.args.fold}.npy'

            init_centroid_path = Path(centroid_filename)
            if not init_centroid_path.is_file():
                 project_dir_in_working = Path("./")
                 init_centroid_path = project_dir_in_working / centroid_filename

            if not init_centroid_path.is_file():
                kaggle_input_path = Path("/kaggle/input/slcl-boundary-precomputed-centroids/")
                init_centroid_path = kaggle_input_path / centroid_filename

            print(f"Attempting to load initial centroids from: {init_centroid_path.resolve()}")
            if not init_centroid_path.is_file():
                raise FileNotFoundError(f"Centroid file not found at expected locations: {Path(centroid_filename).resolve()}, {Path('./') / centroid_filename}, or {init_centroid_path.resolve()}")

            self.centroid_s = np.load(init_centroid_path)
            self.centroid_s = torch.from_numpy(self.centroid_s).float().to(self.device)
            print(f"Initial source centroids loaded successfully, shape: {self.centroid_s.shape}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Please ensure the initial source centroid file exists and is accessible.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading initial source centroids: {e}")
            sys.exit(1)

        self.prepare_model()
        self._update_teacher_network()

        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training Epochs"):
            epoch_start = datetime.now()
            self.adjust_lr(epoch=epoch) 

            train_results = self.train_epoch(epoch)

            results_valid = self.eval(modality='target', phase='valid', toprint=False)
            valid_dice_scores = [results_valid['dc'][k] for k in range(1, self.args.num_classes)]
            lge_dice = np.nanmean(valid_dice_scores) if valid_dice_scores else 0.0
            lge_dice = np.round(lge_dice, 4)

            test_dice_str = ""
            if self.args.evalT:
                 results_test = self.eval(modality='target', phase='test', toprint=False)
                 test_dice_scores = [results_test['dc'][k] for k in range(1, self.args.num_classes)]
                 lge_dice_test = np.nanmean(test_dice_scores) if test_dice_scores else 0.0
                 lge_dice_test = np.round(lge_dice_test, 4)
                 test_dice_str = f", Test Dice: {lge_dice_test:.4f}"

            print("\nWriting summary...")
            if self.args.evalT:
                self.writer.add_scalars('Dice/LGE', {'Valid': lge_dice, 'Test': lge_dice_test}, epoch + 1)
            else:
                self.writer.add_scalar('Dice/LGE_valid', lge_dice, epoch + 1)
            
            self.writer.add_scalar('Loss/Seg_Source', train_results['seg_s'], epoch + 1)
            self.writer.add_scalar('Loss/Seg_Source_Aux', train_results['seg_s_aux'], epoch + 1)
            self.writer.add_scalar('Loss/Contrastive_CL', train_results['loss_cl'], epoch + 1)
            self.writer.add_scalar('Loss/CNR', train_results['loss_cnr'], epoch + 1)
            
            self.writer.add_scalars('Loss/Adv', {'main': train_results['loss_adv'],
                                                 'aux': train_results['loss_adv_aux']}, epoch + 1)
            self.writer.add_scalars('Loss/Dis', {'main': train_results['loss_dis'],
                                                 'aux': train_results['loss_dis_aux']}, epoch + 1)
            self.writer.add_scalars('Acc/Dis_main', {'source': train_results['dis_acc_s'],
                                                  'target': train_results['dis_acc_t']}, epoch + 1)
            self.writer.add_scalars('Acc/Dis_aux', {'source': train_results['dis_aux_acc_s'],
                                                 'target': train_results['dis_aux_acc_t']}, epoch + 1)
            self.writer.add_scalars('LR', {'Segmentor': self.opt.param_groups[0]['lr'],
                                          'Discriminator': self.opt_d.param_groups[0]['lr']}, epoch + 1)
            
            message = (f'\nEpoch = {epoch + 1:4d}/{self.args.epochs:4d} | '
                       f'LR={self.opt.param_groups[0]["lr"]:.2e} | '
                       f'Seg S={train_results["seg_s"]:.4f} | '
                       f'L_CL={train_results["loss_cl"]:.4f} | '
                       f'L_CNR={train_results["loss_cnr"]:.4f} | '
                       f'Adv={train_results["loss_adv"]:.4f} | '
                       f'Dis={train_results["loss_dis"]:.4f} | '
                       f'Val Dice={lge_dice:.4f}{test_dice_str}')
            print(message)

            tobreak = self.stop_training(epoch, epoch_start, lge_dice)

            self.mcp_segmentor.step(monitor=lge_dice, model=self.segmentor, epoch=epoch + 1,
                                    optimizer=self.opt, tobreak=tobreak)
            self.modelcheckpoint_d.step(monitor=lge_dice, model=self.d_main, epoch=epoch + 1,
                                        optimizer=self.opt_d, tobreak=tobreak)
            if self.args.multilvl:
                self.modelcheckpoint_d_aux.step(monitor=lge_dice, model=self.d_aux, epoch=epoch + 1,
                                                optimizer=self.opt_d_aux, tobreak=tobreak)
            if tobreak:
                print(f"Stopping training at epoch {epoch+1} due to early stopping or time limit.")
                break

        self.writer.close()
        best_epoch = self.mcp_segmentor.epoch
        best_score = self.mcp_segmentor.best_result

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