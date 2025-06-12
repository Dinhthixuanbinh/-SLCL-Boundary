import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# --- IMPORTANT: Adjust sys.path to include your project root ---
# This assumes 'analyze_tsne.py' is placed directly in the 'dinhthixuanbinh--slcl-boundary/' directory.
# If you place it in a subdirectory (e.g., 'dinhthixuanbinh--slcl-boundary/scripts/'),
# you might need to change 'parents' to 'parents[1]' or higher to point to the project root.
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent

# Add project root to sys.path to allow imports from model/, dataset/, utils/, trainer/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) # Insert at the beginning for priority

# Import necessary modules from your project
import config
from utils.utils_ import get_device # To get the device

# Import your model classes
# The script will try to load DRUNet or segmentation_models (smp.Unet) based on args.backbone
from model.DRUNet import Segmentation_model as DR_UNet
from model.segmentation_models import segmentation_models

# Import data generators
from dataset.data_generator_mscmrseg import prepare_dataset as prepare_dataset_mscmrseg
from dataset.data_generator_mmwhs import prepare_dataset as prepare_dataset_mmwhs
from dataset.data_generator_mmwhs_raw import prepare_dataset as prepare_dataset_mmwhs_raw


# --- Visualization Function (Modified for your project's model output) ---
def visualize_soft_pseudo_labels_and_centroids(model, target_unlabeled_data_batch, num_classes, sample_size=10000, random_state=42):
    """
    Visualizes target domain features using t-SNE, colored by soft pseudo-label confidence,
    and accurately indicates the location of category-wise centroids based on soft labels
    within the same t-SNE projection.

    Args:
        model (torch.nn.Module): The trained UDA segmentation model.
        target_unlabeled_data_batch (tuple): A batch from the DataLoader (img_t, labels_t, namet).
        num_classes (int): Number of segmentation classes.
        sample_size (int): Number of pixel features to sample for t-SNE visualization.
        random_state (int): Seed for reproducibility of t-SNE and sampling.
    """
    model.eval()
    device = next(model.parameters()).device
    
    img_t, _, _ = target_unlabeled_data_batch # We only need the image for features/pseudo-labels
    img_t = img_t.to(device)

    with torch.no_grad():
        # The model's forward method (e.g., DR_UNet.Segmentation_model.forward)
        # when called with features_out=True, typically returns:
        # (final_logits, bottleneck_features, decoder_features)
        # We want the decoder_features for pixel-level representation and final_logits for pseudo-labels.
        model_output = model(img_t, features_out=True)
        
        logits = model_output # The first element is typically the final prediction logits
        features = model_output[2] # The third element is typically the decoder features (pixel-level features)

        probabilities = torch.softmax(logits, dim=1) # Soft pseudo-labels from logits

    # Reshape features and probabilities from NCHW to (N*H*W, C) for t-SNE
    flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    flat_probs = probabilities.permute(0, 2, 3, 1).reshape(-1, num_classes)

    # Calculate centroids in the original feature space based on soft pseudo-labels
    class_centroids_original_space = torch.zeros(num_classes, flat_features.shape[1], device=device)
    for i in range(num_classes):
        # Weight features by the probability of belonging to class i
        weighted_features = flat_features * flat_probs[:, i].unsqueeze(1)
        sum_probs = flat_probs[:, i].sum() + 1e-7
        # Handle cases where a class might have zero probability sum (e.g., if not present in batch)
        if sum_probs > 1e-6:
            class_centroids_original_space[i] = weighted_features.sum(dim=0) / sum_probs
        else:
            class_centroids_original_space[i] = torch.zeros(flat_features.shape[1], device=device) # Assign zero vector

    # Sample a manageable subset for t-SNE computation
    actual_sample_size = min(len(flat_features), sample_size)
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(flat_features), actual_sample_size, replace=False)

    sampled_features = flat_features[sample_indices].cpu().numpy()
    sampled_probs = flat_probs[sample_indices].cpu().numpy()

    # Concatenate sampled features with centroids for unified t-SNE projection
    centroids_numpy = class_centroids_original_space.cpu().numpy()
    features_and_centroids = np.vstack((sampled_features, centroids_numpy))

    # Compute t-SNE for the sampled features and appended centroids
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, random_state=random_state)
    tsne_results = tsne.fit_transform(features_and_centroids)

    # Separate results for pixel features and centroids
    tsne_pixel_features = tsne_results[:actual_sample_size]
    tsne_centroids = tsne_results[actual_sample_size:]

    # Visualization: Plotting features colored by confidence
    plt.figure(figsize=(12, 10))

    if num_classes == 2:
        scatter = plt.scatter(tsne_pixel_features[:, 0], tsne_pixel_features[:, 1],
                              c=sampled_probs[:, 1], # Confidence for the foreground class (assuming class 1 is foreground)
                              cmap='viridis', alpha=0.6, s=10, label='Pixel Features (Confidence for Class 1)')
        plt.colorbar(scatter, label='Soft Pseudo-Label Confidence (Class 1)')
    else:
        predicted_classes = np.argmax(sampled_probs, axis=1)
        max_confidences = np.max(sampled_probs, axis=1) # Use max confidence for alpha
        scatter = plt.scatter(tsne_pixel_features[:, 0], tsne_pixel_features[:, 1],
                              c=predicted_classes,
                              cmap='tab10', alpha=max_confidences * 0.8 + 0.2, s=10, # Scale alpha for visibility
                              label='Pixel Features (Predicted Class & Confidence)')
        plt.colorbar(scatter, label='Predicted Class Index')

    # Overlay centroids
    plt.scatter(tsne_centroids[:, 0], tsne_centroids[:, 1],
                marker='X', s=200, color='red', edgecolor='black', linewidth=2,
                label='Class Centroids (Projected)')

    plt.title('t-SNE of Target Features with Soft Pseudo-Labels and Centroids')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.legend()
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Create a dummy args object to mimic command-line arguments
    # These values should match your training setup or be adjusted as needed
    class Args:
        def __init__(self):
            # General config from config.py
            self.data_dir = config.DATA_DIRECTORY
            self.raw_data_dir = config.RAW_DATA_DIRECTORY
            self.num_classes = config.NUM_CLASSES
            self.crop = config.INPUT_SIZE
            self.bs = config.BATCH_SIZE
            self.eval_bs = config.EVAL_BS
            self.seed = config.RANDOM_SEED
            self.normalization = 'minmax' # Or 'zscore', match your training
            self.clahe = False # Match your training
            self.raw = True # Set to True if using raw NII data (e.g., for MMWHS raw)
            self.rev = False # Set to True if your target is source and source is target (e.g., MR->CT)
            self.fold = 0 # Match your training fold
            self.split = 0 # Match your training split
            self.val_num = 0
            self.noM3AS = True # Match your training
            self.pin_memory = True
            self.num_workers = 0 # Set to 0 for debugging, or config.NUM_WORKERS for faster loading
            self.spacing = 1.0 # For MMWHS, check config.py or your training setup
            self.percent = 100 # For MMWHS raw data, check config.py or your training setup

            # Model specific args
            self.backbone = 'drunet' # Or 'resnet50', 'unet', etc.
            self.filters = 32 # DRUNet specific
            self.nb = 4 # DRUNet specific
            self.bd = 4 # DRUNet specific
            self.multilvl = True # Set to True if your model was trained with multi-level outputs

            # Checkpoint path (THIS IS THE MOST IMPORTANT PART TO CONFIGURE)
            # REPLACE THIS WITH THE ACTUAL PATH TO YOUR TRAINED MODEL CHECKPOINT
            self.restore_from = 'path/to/your/trained_model.pt' 
            # Example: self.restore_from = 'weights/best_MPSCL.mscmrseg.s0.f0.v0.drunet.32.nb4.bd4.clahe.lr0.02.raw.augstHvy.bs16.trainWst.mnmx.lr_dis0.0001.w_dis0.001.w_mpscl_s1.0.t1.0.eXX.Scr0.XXX.pt'
            # Or: self.restore_from = 'weights/best_DR_UNet.fewshot.lr0.0003.cw0.002.poly.pat_10_lge.adam.e63.Scr0.674.pt'
            
            # Other args that might be expected by Trainer_MPSCL init (can be default)
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
            self.scratch = False # Set to True if your data is on /scratch

    args = Args()

    # Validate model checkpoint path
    if not os.path.exists(args.restore_from):
        print(f"Error: Model checkpoint not found at '{args.restore_from}'")
        print("Please update 'args.restore_from' in the script to point to your trained model's.pt file.")
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")

    try:
        # 1. Prepare DataLoader for the target test set
        print("Initializing data loader for target domain test data...")
        target_loader = None
        if 'mscmrseg' in args.data_dir:
            data_loaders = prepare_dataset_mscmrseg(args)
            target_loader = data_loaders['test_t'] # Target test loader
        elif 'mmwhs' in args.data_dir:
            if args.raw:
                # prepare_dataset_mmwhs_raw returns (scratch, raw_scratch, content_loader, style_loader)
                _, _, _, target_loader = prepare_dataset_mmwhs_raw(args)
            else:
                # prepare_dataset_mmwhs returns (scratch, raw_scratch, content_loader, style_loader)
                _, _, _, target_loader = prepare_dataset_mmwhs(args)
        else:
            raise NotImplementedError(f"Dataset '{args.data_dir}' not supported for direct loading.")

        if target_loader is None or len(target_loader) == 0:
            print("Error: Target data loader could not be initialized or is empty. Check data paths and configuration.")
            sys.exit(1)

        # 2. Prepare Model
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

        # Load model weights
        print(f"Loading model weights from: {args.restore_from}")
        checkpoint = torch.load(args.restore_from, map_location=device)
        try:
            # Try loading state_dict from 'model_state_dict' key (common for Trainer checkpoints)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print('Model state_dict loaded strictly from checkpoint.')
        except KeyError:
            # If 'model_state_dict' key is not found, assume the checkpoint itself is the state_dict
            model.load_state_dict(checkpoint, strict=True)
            print('Model state_dict loaded directly (assuming checkpoint is state_dict).')
        except RuntimeError as e:
            # If strict loading fails, try non-strict (e.g., if some layers are missing/renamed)
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
        model.eval() # Set model to evaluation mode

        # Get one batch of target unlabeled data
        print("Fetching a batch of target unlabeled data for visualization...")
        target_data_batch = next(iter(target_loader))

        # Call the visualization function
        print("Generating t-SNE visualization...")
        visualize_soft_pseudo_labels_and_centroids(
            model,
            target_data_batch,
            num_classes=args.num_classes,
            sample_size=5000 # Adjust sample_size based on your data and computational resources
        )
        print("Visualization complete.")

    except Exception as e:
        print(f"An error occurred during setup or visualization: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)