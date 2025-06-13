# utils/proposed_losses.py
import torch
import torch.nn.functional as F

# Import the helper from proposed_utils
from utils.proposed_utils import calculate_prototype_loss_from_features
def consistency_loss_mse(predictions1, predictions2):
    """
    Calculates Mean Squared Error between two sets of probabilistic predictions.
    :param predictions1: Softmax probabilities from one augmented view (NCHW).
    :param predictions2: Softmax probabilities from another augmented view (NCHW).
    """
    return F.mse_loss(predictions1, predictions2)

def consistency_loss_kl_divergence(predictions1, predictions2, temperature=1.0):
    """
    Calculates KL-Divergence between two sets of probabilistic predictions.
    Often used in Mean Teacher where predictions1 is from teacher (detached).
    :param predictions1: Softmax probabilities (teacher's output, detached).
    :param predictions2: Logits (student's output).
    :param temperature: Softmax temperature for KL-divergence.
    """
    # Scale logits by temperature for soft probabilities
    log_probs2 = F.log_softmax(predictions2 / temperature, dim=1)
    probs1 = predictions1 # Already softmaxed probabilities
    
    # KL_div(P || Q) = sum(P * log(P / Q))
    # PyTorch's F.kl_div expects log_probs as the second argument
    # F.kl_div(input=log_probs, target=probs, reduction='batchmean')
    loss = F.kl_div(log_probs2, probs1, reduction='batchmean') * (temperature ** 2)
    return loss

def prototype_contrastive_loss(features, pseudo_labels, num_classes, temperature=0.07):
    """
    A wrapper for the InfoNCE-like prototype loss function, calling the helper.
    Assumes features are (M, Feature_Dim) and pseudo_labels are (M,).
    """
    return calculate_prototype_loss_from_features(features, pseudo_labels, num_classes, temperature)

