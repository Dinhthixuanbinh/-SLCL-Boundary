import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for path manipulation



# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your actual model if you have it.
# from model.DRUNet import Segmentation_model as DR_UNet

def visualize_soft_pseudo_labels_and_centroids(model, target_unlabeled_data, num_classes=4, sample_size=10000, random_state=42):
    """
    Visualizes target domain features using t-SNE, colored by soft pseudo-label confidence,
    and accurately indicates the location of category-wise centroids based on soft labels
    within the same t-SNE projection.

    Args:
        model (torch.nn.Module): The trained UDA segmentation model. Assumes it has
                                  'feature_extractor' and 'segmentation_head' attributes.
        target_unlabeled_data (torch.Tensor): A batch of target images.
        num_classes (int): Number of segmentation classes (e.g., 2 for foreground/background,
                            4 for cardiac structures + background, as per config.py [2]).
        sample_size (int): Number of pixel features to sample for t-SNE visualization to manage computation.
        random_state (int): Seed for reproducibility of t-SNE and sampling.
    """
    model.eval()
    with torch.no_grad():
        # Ensure target_unlabeled_data is on the correct device
        device = next(model.parameters()).device # Get model's device
        target_unlabeled_data = target_unlabeled_data.to(device)

        # Extract features and probabilities from the model
        features = model.feature_extractor(target_unlabeled_data)
        logits = model.segmentation_head(features)
        probabilities = torch.softmax(logits, dim=1) # Soft pseudo-labels

    # Reshape features and probabilities from NCHW to (N*H*W, C) for t-SNE
    flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
    flat_probs = probabilities.permute(0, 2, 3, 1).reshape(-1, num_classes)

    # Calculate centroids in the original feature space based on soft pseudo-labels
    # For each class, compute a weighted average of features, where weights are the probabilities
    class_centroids_original_space = torch.zeros(num_classes, flat_features.shape[1], device=device)
    for i in range(num_classes):
        # Weight features by the probability of belonging to class i
        # Ensure broadcasting: flat_probs[:, i] needs to be expanded to match feature dimensions
        weighted_features = flat_features * flat_probs[:, i].unsqueeze(1)
        # Sum weighted features and normalize by sum of probabilities for class i
        # Add a small epsilon to the denominator to avoid division by zero for empty classes
        sum_probs = flat_probs[:, i].sum() + 1e-7
        class_centroids_original_space[i] = weighted_features.sum(dim=0) / sum_probs

    # Sample a manageable subset for t-SNE computation
    # Ensure sample_size does not exceed available features
    actual_sample_size = min(len(flat_features), sample_size)
    np.random.seed(random_state) # Set seed for reproducibility
    sample_indices = np.random.choice(len(flat_features), actual_sample_size, replace=False)

    sampled_features = flat_features[sample_indices].cpu().numpy()
    sampled_probs = flat_probs[sample_indices].cpu().numpy()

    # Concatenate sampled features with centroids for unified t-SNE projection
    # Convert centroids to numpy and move to CPU
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
        # For multi-class, color by predicted class, and use maximum confidence for alpha (saturation)
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

# Example usage:
if __name__ == "__main__":
    # Define a dummy model for demonstration purposes.
    # In a real scenario, you would load your trained model.
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes=4, feature_channels=64, img_size=224):
            super().__init__()
            # Mimic feature_extractor and segmentation_head
            # This is a simplified convolutional network to produce features
            self.feature_extractor = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2), # Reduces spatial dimensions by 2
                torch.nn.Conv2d(32, feature_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2) # Reduces spatial dimensions by another 2
            )
            # The segmentation head takes features and outputs logits
            # The spatial dimensions of features will be img_size / 4 (e.g., 224/4 = 56)
            self.segmentation_head = torch.nn.Conv2d(feature_channels, num_classes, kernel_size=1)

        def forward(self, x):
            features = self.feature_extractor(x)
            logits = self.segmentation_head(features)
            return logits

    # Set up device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Instantiate your model.
    # Replace DummyModel with your actual Segmentation_model (e.g., DR_UNet)
    # from model.DRUNet import Segmentation_model as DR_UNet
    # model = DR_UNet(n_class=4).to(device)
    # If you have a trained model, load its weights:
    # model.load_state_dict(torch.load('path/to/your/best_model.pt')['model_state_dict'])
    # model.eval() # Set to evaluation mode

    # For demonstration, we'll use the DummyModel
    num_classes_in_model = 4 # As per config.py [2]
    model = DummyModel(num_classes=num_classes_in_model).to(device)

    # Create dummy target unlabeled data.
    # In a real scenario, this would come from your target domain DataLoader.
    # Shape: (batch_size, channels, height, width)
    dummy_target_data = torch.randn(4, 3, 224, 224) # Example: 4 images, 3 channels, 224x224 pixels

    # Call the visualization function
    print("Generating t-SNE visualization...")
    visualize_soft_pseudo_labels_and_centroids(
        model,
        dummy_target_data,
        num_classes=num_classes_in_model,
        sample_size=5000 # Adjust sample_size based on your data and computational resources
    )
    print("Visualization complete.")