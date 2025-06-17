# %%writefile /kaggle/working/-SLCL-Boundary/utils/proposed_utils.py
# utils/proposed_utils.py

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

# Re-use ImageProcessor from data_generator_mscmrseg for consistent augmentation logic
from dataset.data_generator_mscmrseg import ImageProcessor # Assuming this is the core augmentation utility

# You might want to pass normalization details to augment functions
# so they can handle conversion to/from 0-255 if ImageProcessor expects it.


def augmentation_weak(images, crop_size=224, aug_mode='simple'):
    """
    Applies weak data augmentation.
    Assumes images are NCHW (torch tensor). Converts to NHWC for ImageProcessor and back.
    :param images: torch.Tensor, input images (NCHW)
    :param crop_size: int, target crop size (if ImageProcessor needs it)
    :param aug_mode: str, 'simple' or 'none' for no augmentation
    """
    np_images = images.detach().cpu().numpy()
    original_shape = np_images.shape
    original_dtype = np_images.dtype

    # Ensure HWC for ImageProcessor.simple_aug. Assuming NCHW input.
    if np_images.ndim == 4: # NCHW -> NHWC
        np_images = np.transpose(np_images, (0, 2, 3, 1))
    elif np_images.ndim == 3: # CHW -> HWC
        np_images = np.transpose(np_images, (1, 2, 0))

    # Convert to uint8 for ImageProcessor if it expects 0-255 range
    np_images_processed = (np_images * 255).astype(np.uint8) # Assuming input is [0,1] or similar

    aug_images_list = []
    for img_idx in range(np_images_processed.shape[0]):
        img_single = np_images_processed[img_idx]
        
        # Apply weak augmentation
        if aug_mode == 'simple':
            aug_img_single, _ = ImageProcessor.simple_aug(image=img_single, mask=None) # No mask here
        elif aug_mode == 'none': # Identity augmentation
            aug_img_single = img_single
        else:
            raise NotImplementedError(f"Weak augmentation mode '{aug_mode}' not implemented for consistency.")
        
        # Ensure it's 3-channel if input was 3-channel
        if aug_img_single.ndim == 2 and np_images_processed.shape[-1] == 3:
            aug_img_single = np.stack([aug_img_single, aug_img_single, aug_img_single], axis=-1)
        elif aug_img_single.ndim == 2 and np_images_processed.shape[-1] == 1:
            aug_img_single = aug_img_single[..., np.newaxis] # Add channel dim if it was lost

        # Apply crop_resize if needed to ensure output shape consistency
        aug_img_single = ImageProcessor.crop_resize(aug_img_single, target_size=(crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        aug_images_list.append(aug_img_single)

    aug_images_np = np.stack(aug_images_list, axis=0) # NHWC

    # Convert back to original scale and NCHW torch tensor
    aug_images_torch = aug_images_np.astype(original_dtype) / 255.0
    if original_shape[1] <= 3: # Assuming original input was NCHW (C<=3 for images)
        aug_images_torch = np.transpose(aug_images_torch, (0, 3, 1, 2)) # NHWC -> NCHW

    return torch.from_numpy(aug_images_torch).to(images.device).float()


def augmentation_strong(images, crop_size=224, aug_mode='heavy'):
    """
    Applies strong data augmentation.
    Assumes images are NCHW (torch tensor). Converts to NHWC for ImageProcessor and back.
    :param images: torch.Tensor, input images (NCHW)
    :param crop_size: int, target crop size (if ImageProcessor needs it)
    :param aug_mode: str, 'heavy' or 'heavy2'
    """
    np_images = images.detach().cpu().numpy()
    original_shape = np_images.shape
    original_dtype = np_images.dtype

    if np_images.ndim == 4: # NCHW -> NHWC
        np_images = np.transpose(np_images, (0, 2, 3, 1))
    elif np_images.ndim == 3: # CHW -> HWC
        np_images = np.transpose(np_images, (1, 2, 0))

    np_images_processed = (np_images * 255).astype(np.uint8) # Assuming input is [0,1] or similar

    aug_images_list = []
    for img_idx in range(np_images_processed.shape[0]):
        img_single = np_images_processed[img_idx]

        # Apply strong augmentation
        if aug_mode == 'heavy' or aug_mode == 'heavy2':
            aug_img_single, _ = ImageProcessor.heavy_aug(image=img_single, mask=None, aug_mode=aug_mode)
        else:
            raise NotImplementedError(f"Strong augmentation mode '{aug_mode}' not implemented for consistency.")

        if aug_img_single.ndim == 2 and np_images_processed.shape[-1] == 3:
            aug_img_single = np.stack([aug_img_single, aug_img_single, aug_img_single], axis=-1)
        elif aug_img_single.ndim == 2 and np_images_processed.shape[-1] == 1:
            aug_img_single = aug_img_single[..., np.newaxis] # Add channel dim if it was lost

        aug_img_single = ImageProcessor.crop_resize(aug_img_single, target_size=(crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        aug_images_list.append(aug_img_single)

    aug_images_np = np.stack(aug_images_list, axis=0)

    aug_images_torch = aug_images_np.astype(original_dtype) / 255.0
    if original_shape[1] <= 3: # Assuming original input was NCHW (C<=3 for images)
        aug_images_torch = np.transpose(aug_images_torch, (0, 3, 1, 2)) # NHWC -> NCHW

    return torch.from_numpy(aug_images_torch).to(images.device).float()


def sharpen_probabilities(probabilities, temperature):
    """
    Sharpens predicted probabilities by increasing the peak probability and decreasing others.
    :param probabilities: torch.Tensor, output probabilities (NCHW)
    :param temperature: float, sharpening temperature (lower value -> more sharpening)
    """
    if temperature == 1.0: # No sharpening
        return probabilities
    
    # Apply softmax with temperature
    # P_sharpened = softmax(logits / T)
    # Since we have probabilities, we need to convert back to "pseudo-logits"
    # log_probs = log(P)
    # pseudo_logits = log(P) / (1/T) = log(P) * T
    # Then apply softmax on sharpened pseudo-logits
    
    # Avoid log(0) if probabilities contain zeros (add small epsilon)
    pseudo_logits = torch.log(probabilities + 1e-7) / temperature
    sharpened_probs = F.softmax(pseudo_logits, dim=1)
    return sharpened_probs


# --- Existing functions (keep as is) ---
def calculate_self_adaptive_thresholds(confidences_per_class, initial_threshold, num_classes, percentile_q=80):
    """
    Calculates self-adaptive thresholds based on historical confidence distributions.
    :param confidences_per_class: A dictionary where keys are class IDs and values are lists of historical confidences.
    :param initial_threshold: Fallback threshold if no history exists for a class.
    :param num_classes: Total number of classes.
    :param percentile_q: The percentile to use for adaptive thresholding (e.g., 80 for 80th percentile).
    """
    thresholds = {}
    for c in range(num_classes):
        if c in confidences_per_class and len(confidences_per_class[c]) > 0:
            thresholds[c] = np.percentile(confidences_per_class[c], percentile_q)
        else:
            thresholds[c] = initial_threshold # Fallback
    return thresholds

def calculate_prototype_loss_from_features(features, pseudo_labels, num_classes, temperature=0.1):
    """
    Calculates prototype-based feature alignment loss (e.g., InfoNCE-like).
    Assumes features are (M, Feature_Dim) and pseudo_labels are (M,) of filtered reliable pixels.
    """
    if features.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = 0.0
    features = F.normalize(features, dim=1)

    prototypes = torch.zeros(num_classes, features.shape[1], device=features.device)
    counts = torch.zeros(num_classes, device=features.device)

    for c in range(num_classes):
        class_features = features[pseudo_labels == c]
        if class_features.numel() > 0:
            prototypes[c] = torch.mean(class_features, dim=0)
            counts[c] = class_features.shape[0]

    prototypes = F.normalize(prototypes, dim=1)
    active_classes = torch.where(counts > 0)[0]
    if active_classes.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    active_prototypes = prototypes[active_classes]
    
    for i in range(features.shape[0]):
        feat = features[i].unsqueeze(0)
        label = pseudo_labels[i].item()

        if label not in active_classes:
            continue

        logits = torch.matmul(feat, active_prototypes.T) / temperature
        positive_idx = (active_classes == label).nonzero(as_tuple=True)[0]
        log_prob = F.log_softmax(logits, dim=1)
        loss_i = -log_prob[0, positive_idx].mean()
        loss += loss_i

    return loss / features.shape[0] if features.shape[0] > 0 else torch.tensor(0.0, device=features.device)

# Re-use prob_2_entropy from utils.utils_ for consistency
from utils.utils_ import prob_2_entropy