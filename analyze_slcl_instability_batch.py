import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm # For progress bar

# --- IMPORTANT: Adjust sys.path to include your project root ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import necessary modules from your project
import config
from utils.utils_ import get_device
from dataset.data_generator_mscmrseg import ImageProcessor # Assuming this is accessible
from model.DRUNet import Segmentation_model as DR_UNet
from model.segmentation_models import segmentation_models
from dataset.data_generator_mscmrseg import prepare_dataset as prepare_dataset_mscmrseg
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw


def analyze_pseudo_label_consistency_batch(model, target_loader, num_classes, num_images_to_test=10, output_dir="slcl_instability_plots"):
    """
    Analyzes pseudo-label consistency for multiple target images under minor perturbations,
    saving plots and aggregating disagreement statistics.
    
    Args:
        model (torch.nn.Module): The trained UDA segmentation model.
        target_loader (torch.utils.data.DataLoader): DataLoader for the target domain test set.
        num_classes (int): Number of segmentation classes.
        num_images_to_test (int): The total number of images to process and visualize.
        output_dir (str): Directory to save the generated plots.
    """
    model.eval()
    device = next(model.parameters()).device
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving instability plots to: {Path(output_dir).resolve()}")

    all_disagreement_percentages =
    images_processed_count = 0

    # Iterate through the DataLoader to get multiple images
    for batch_idx, batch_data in enumerate(tqdm(target_loader, desc="Processing images for instability analysis")):
        if images_processed_count >= num_images_to_test:
            break

        img_t_batch, _, names_batch = batch_data # Extract images and names from the batch

        for sample_in_batch_idx in range(img_t_batch.shape):
            if images_processed_count >= num_images_to_test:
                break

            img_t_single_tensor = img_t_batch[sample_in_batch_idx:sample_in_batch_idx+1].to(device)
            original_image_name = names_batch[sample_in_batch_idx]

            # Convert to NumPy for ImageProcessor.simple_aug (expects HWC, uint8)
            # Assuming input tensor is (1, C, H, W) and normalized  or [-1,1]
            img_t_single_np = img_t_single_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            if img_t_single_np.dtype!= np.uint8:
                # Scale to 0-255 and convert to uint8
                img_t_single_np = (img_t_single_np * 255).astype(np.uint8)
            
            # Generate two augmented views using your project's simple_aug
            aug_img_1_np, _ = ImageProcessor.simple_aug(image=img_t_single_np, mask=None)
            aug_img_2_np, _ = ImageProcessor.simple_aug(image=img_t_single_np, mask=None)

            # Convert back to CHW and then to Tensor for model input, re-normalize
            aug_img_1_tensor = torch.from_numpy(np.moveaxis(aug_img_1_np, -1, 0)).float().unsqueeze(0).to(device) / 255.0
            aug_img_2_tensor = torch.from_numpy(np.moveaxis(aug_img_2_np, -1, 0)).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                # Get pseudo-labels from both augmented views
                logits_1_output = model(aug_img_1_tensor, features_out=True)
                logits_2_output = model(aug_img_2_tensor, features_out=True)

                # Ensure logits are extracted correctly (first element of tuple)
                logits_1 = logits_1_output if isinstance(logits_1_output, tuple) else logits_1_output
                logits_2 = logits_2_output if isinstance(logits_2_output, tuple) else logits_2_output

                pseudo_labels_1 = torch.argmax(torch.softmax(logits_1, dim=1), dim=1)
                pseudo_labels_2 = torch.argmax(torch.softmax(logits_2, dim=1), dim=1)

                # Calculate disagreement map
                disagreement_map = (pseudo_labels_1!= pseudo_labels_2).float()
                mean_disagreement = disagreement_map.mean().item() * 100 # Percentage

            all_disagreement_percentages.append(mean_disagreement)

            # Move to CPU for plotting
            pseudo_labels_1_np = pseudo_labels_1.cpu().squeeze().numpy()
            pseudo_labels_2_np = pseudo_labels_2.cpu().squeeze().numpy()
            disagreement_map_np = disagreement_map.cpu().squeeze().numpy()
            
            # For displaying original image (ensure it's HWC for imshow)
            original_img_display = img_t_single_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            if original_img_display.shape[1] == 1: # If grayscale, remove channel dim
                original_img_display = original_img_display.squeeze(2)

            # Plotting
            fig, axes = plt.subplots(1, 4, figsize=(20, 6))

            axes.imshow(original_img_display, cmap='gray')
            axes.set_title(f'Original Image\n({original_image_name})')
            axes.axis('off')

            axes.[2]imshow(pseudo_labels_1_np, cmap='tab10', vmin=0, vmax=num_classes-1)
            axes.[2]set_title('Pseudo-Label View 1')
            axes.[2]axis('off')

            axes.[1]imshow(pseudo_labels_2_np, cmap='tab10', vmin=0, vmax=num_classes-1)
            axes.[1]set_title('Pseudo-Label View 2')
            axes.[1]axis('off')

            axes.[3]imshow(disagreement_map_np, cmap='binary') # Binary colormap for disagreement (0=agree, 1=disagree)
            axes.[3]set_title(f'Disagreement Map\n({mean_disagreement:.2f}% pixels differ)')
            axes.[3]axis('off')

            plt.tight_layout()
            
            # Save the plot
            plot_filename = os.path.join(output_dir, f"instability_{original_image_name}_batch{batch_idx}_sample{sample_in_batch_idx}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig) # Close the figure to free memory

            images_processed_count += 1
    
    # Print summary statistics
    print("\n--- Pseudo-Label Instability Analysis Summary ---")
    if all_disagreement_percentages:
        print(f"Processed {len(all_disagreement_percentages)} images.")
        print(f"Average pixel disagreement: {np.mean(all_disagreement_percentages):.2f}%")
        print(f"Min pixel disagreement: {np.min(all_disagreement_percentages):.2f}%")
        print(f"Max pixel disagreement: {np.max(all_disagreement_percentages):.2f}%")
        print(f"Std dev of pixel disagreement: {np.std(all_disagreement_percentages):.2f}%")
    else:
        print("No images were processed.")

# --- Main Execution Block ---
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.data_dir = config.DATA_DIRECTORY
            self.raw_data_dir = config.RAW_DATA_DIRECTORY
            self.num_classes = config.NUM_CLASSES
            self.crop = config.INPUT_SIZE
            self.bs = config.BATCH_SIZE # Use batch size for loading, but process images individually
            self.eval_bs = config.EVAL_BS
            self.seed = config.RANDOM_SEED
            self.normalization = 'minmax'
            self.clahe = False
            self.raw = True
            self.rev = False
            self.fold = 0
            self.split = 0
            self.val_num = 0
            self.noM3AS = True
            self.pin_memory = True
            self.num_workers = 0
            self.spacing = 1.0
            self.percent = 100

            self.backbone = 'drunet'
            self.filters = 32
            self.nb = 4
            self.bd = 4
            self.multilvl = True

            # --- IMPORTANT: REPLACE THIS PATH WITH YOUR ACTUAL TRAINED MODEL CHECKPOINT ---
            self.restore_from = 'path/to/your/trained_model.pt' 
            
            # Other args (keep as default for loading)
            self.apdx = ''
            self.evalT = False
            self.aug_s = False
            self.aug_t = False
            self.aug_mode = 'simple'
            self.save_data = False
            self.pretrained = False
            self.optim = 'adam'
            self.lr_decay_method = None
            self.lr = config.LEARNING_RATE
            self.lr_decay = config.LEARNING_RATE_DECAY
            self.lr_end = 0.0
            self.momentum = config.MOMENTUM
            self.power = config.POWER
            self.weight_decay = config.WEIGHT_DECAY
            self.epochs = config.EPOCHS
            self.vgg = '/kaggle/input/vgg-normalised/vgg_normalised.pth'
            self.style_dir = './style_track'
            self.save_every_epochs = config.SAVE_PRED_EVERY
            self.train_with_s = True
            self.train_with_t = True
            self.toggle_klc = True
            self.hd95 = False
            self.estop = False
            self.stop_epoch = 200
            self.adjust_lr = True
            self.src_temp = 0.1
            self.src_base_temp = 1
            self.trg_temp = 0.1
            self.trg_base_temp = 1
            self.src_margin = 0.4
            self.trg_margin = 0.2
            self.class_center_m = 0.9
            self.pixel_sel_th = 0.25
            self.w_mpcl_s = 1.0
            self.w_mpcl_t = 1.0
            self.dis_type = 'origin'
            self.part = 1
            self.CNR_w = 0.0
            self.lr_dis = 1e-4
            self.w_dis = 1e-3
            self.mmt1 = 0.9
            self.mmt = 0.99
            self.restore_d = None
            self.restore_d_aux = None
            self.w_seg_aux = 0.1
            self.w_dis_aux = 2e-4
            self.scratch = False

    args = Args()

    if not os.path.exists(args.restore_from):
        print(f"Error: Model checkpoint not found at '{args.restore_from}'")
        print("Please update 'args.restore_from' in the script to point to your trained model's.pt file.")
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    try:
        print("Initializing data loader for target domain test data...")
        target_loader = None
        if 'mscmrseg' in args.data_dir:
            data_loaders = prepare_dataset_mscmrseg(args)
            target_loader = data_loaders['test_t']
        elif 'mmwhs' in args.data_dir:
            if args.raw:
                _, _, _, target_loader = prepare_dataset_mmwhs_raw(args)
            else:
                _, _, _, target_loader = prepare_dataset_mmwhs(args)
        else:
            raise NotImplementedError(f"Dataset '{args.data_dir}' not supported for direct loading.")

        if target_loader is None or len(target_loader) == 0:
            print("Error: Target data loader could not be initialized or is empty. Check data paths and configuration.")
            sys.exit(1)

        print(f"Initializing model with backbone: {args.backbone}...")
        model = None
        if args.backbone == 'drunet':
            model = DR_UNet(filters=args.filters, n_block=args.nb, bottleneck_depth=args.bd,
                            n_class=args.num_classes, multilvl=args.multilvl, args=args)
        elif 'resnet' in args.backbone or 'efficientnet' in args.backbone or \
             'mobilenet' in args.backbone or 'densenet' in args.backbone or \
             'ception' in args.backbone or 'se_resnet' in args.backbone or \
             'skresnext' in args.backbone:
            model = segmentation_models(name=args.backbone, pretrained=False,
                                        decoder_channels=(512, 256, 128, 64, 32),
                                        in_channel=3, classes=args.num_classes,
                                        multilvl=args.multilvl, args=args)
        else:
            raise NotImplementedError(f"Model backbone '{args.backbone}' not supported for direct loading.")

        if model is None:
            print("Error: Model could not be initialized. Check backbone configuration.")
            sys.exit(1)

        print(f"Loading model weights from: {args.restore_from}")
        checkpoint = torch.load(args.restore_from, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print('Model state_dict loaded strictly from checkpoint.')
        except KeyError:
            model.load_state_dict(checkpoint, strict=True)
            print('Model state_dict loaded directly (assuming checkpoint is state_dict).')
        except RuntimeError as e:
            print(f"Strict loading failed: {e}. Attempting non-strict loading...")
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print('Model state_dict loaded non-strictly.')
            except KeyError:
                model.load_state_dict(checkpoint, strict=False)
                print('Model state_dict loaded non-strictly (assuming checkpoint is state_dict).')
            except Exception as non_strict_e:
                print(f"Non-strict loading also failed: {non_strict_e}")
                sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            sys.exit(1)

        model.to(device)
        model.eval()

        # --- Set the number of images to test ---
        NUM_IMAGES_TO_TEST = 10 # You can change this to any desired number

        print(f"\nRunning Experiment: Pseudo-Label Instability and Overconfidence at Boundaries on {NUM_IMAGES_TO_TEST} images...")
        analyze_pseudo_label_consistency_batch(
            model,
            target_loader,
            num_classes=args.num_classes,
            num_images_to_test=NUM_IMAGES_TO_TEST
        )
        print("Experiment complete. Check the 'slcl_instability_plots' directory for images.")

    except Exception as e:
        print(f"An error occurred during setup or visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)