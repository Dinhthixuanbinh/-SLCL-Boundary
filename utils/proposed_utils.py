# utils/proposed_utils.py
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def augment_for_consistency(images, crop_size=224):
    """
    Applies random augmentations suitable for consistency regularization.
    Assumes images are NCHW (torch tensor). Converts to NHWC for imgaug and back.
    """
    # Convert torch tensor to numpy for imgaug
    # Detach from graph and move to CPU if on GPU
    np_images = images.detach().cpu().numpy()
    if np_images.ndim == 4: # NCHW
        np_images = np.transpose(np_images, (0, 2, 3, 1)) # NHWC
    elif np_images.ndim == 3: # CHW (single image)
        np_images = np.transpose(np_images, (1, 2, 0)) # HWC

    # Define augmentation sequence (can be customized)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.2), # vertical flips
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-15, 15),
            shear=(-8, 8),
            order=[0, 1] # Use nearest for masks, linear for images (if masks were also transformed)
        ),
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 1.0))),
        iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5))
    ], random_order=True)

    # Convert to uint8 as imgaug expects images typically in 0-255 range
    # Assuming images are already normalized to [0,1] or similar; scale to 0-255 for aug then back
    original_dtype = np_images.dtype
    np_images_uint8 = (np_images * 255).astype(np.uint8) # Scale to 0-255
    
    aug_images_uint8 = seq(images=np_images_uint8)

    # Convert back to original scale and torch tensor
    aug_images = aug_images_uint8.astype(original_dtype) / 255.0 # Scale back to 0-1
    
    if np_images.ndim == 4: # NCHW
        aug_images = np.transpose(aug_images, (0, 3, 1, 2)) # NCHW
    elif np_images.ndim == 3: # CHW
        aug_images = np.transpose(aug_images, (2, 0, 1)) # CHW

    return torch.from_numpy(aug_images).to(images.device).float()


def calculate_self_adaptive_thresholds(confidences_per_class, initial_threshold, num_classes, percentile_q=80):
    """
    Calculates self-adaptive thresholds based on historical confidence distributions.
    This is a simplified conceptual version. Real SAT methods are more complex (e.g., FlexMatch).
    :param confidences_per_class: A dictionary where keys are class IDs and values are lists of historical confidences.
    :param initial_threshold: Fallback threshold if no history exists for a class.
    :param num_classes: Total number of classes.
    :param percentile_q: The percentile to use for adaptive thresholding (e.g., 80 for 80th percentile).
    """
    thresholds = {}
    for c in range(num_classes):
        if c in confidences_per_class and len(confidences_per_class[c]) > 0:
            # Simple adaptive threshold: e.g., 80th percentile of recent confidences
            # Or a more sophisticated method like average of top-k confidences
            thresholds[c] = np.percentile(confidences_per_class[c], percentile_q)
        else:
            thresholds[c] = initial_threshold # Fallback
    return thresholds

def calculate_prototype_loss_from_features(features, pseudo_labels, num_classes, temperature=0.1):
    """
    Calculates prototype-based feature alignment loss (e.g., InfoNCE-like).
    Assumes features are (N*H*W, C_features) and pseudo_labels are (N*H*W) of filtered reliable pixels.
    This is a conceptual InfoNCE-like loss for clarity.
    :param features: Filtered (reliable) target features (M, Feature_Dim)
    :param pseudo_labels: Corresponding filtered pseudo-labels (M,)
    :param num_classes: Total number of classes.
    :param temperature: Temperature parameter for contrastive loss.
    """
    if features.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = 0.0
    
    # Normalize features
    features = F.normalize(features, dim=1)

    # Calculate prototypes for the current batch
    prototypes = torch.zeros(num_classes, features.shape[1], device=features.device)
    counts = torch.zeros(num_classes, device=features.device)

    for c in range(num_classes):
        class_features = features[pseudo_labels == c]
        if class_features.numel() > 0:
            prototypes[c] = torch.mean(class_features, dim=0)
            counts[c] = class_features.shape[0]

    # Normalize prototypes
    prototypes = F.normalize(prototypes, dim=1)

    # Filter out prototypes for classes that had no samples in this batch (to avoid NaNs)
    # And corresponding features.
    active_classes = torch.where(counts > 0)[0]
    if active_classes.numel() == 0:
        return torch.tensor(0.0, device=features.device) # No active classes, no loss

    active_prototypes = prototypes[active_classes]
    
    # Iterate through each sample to compute contrastive loss
    for i in range(features.shape[0]):
        feat = features[i].unsqueeze(0) # (1, Feature_Dim)
        label = pseudo_labels[i].item()

        # If the label is not an active class in this batch, skip (or handle appropriately)
        if label not in active_classes:
            continue

        # Compute similarity between current feature and all active prototypes
        # (1, Feature_Dim) @ (Feature_Dim, Num_Active_Classes) -> (1, Num_Active_Classes)
        logits = torch.matmul(feat, active_prototypes.T) / temperature

        # Identify positive sample (the prototype corresponding to its own label)
        positive_idx = (active_classes == label).nonzero(as_tuple=True)[0]
        
        # Compute log_softmax to get log probabilities for InfoNCE
        log_prob = F.log_softmax(logits, dim=1)
        
        # The loss for this sample is -log(P_positive)
        # We need to sum over the positive_idx (which should be a single element here)
        loss_i = -log_prob[0, positive_idx].mean() # Use mean in case positive_idx is not single
        loss += loss_i

    return loss / features.shape[0] if features.shape[0] > 0 else torch.tensor(0.0, device=features.device)

# Re-use prob_2_entropy from utils.utils_ for consistency
from utils.utils_ import prob_2_entropy