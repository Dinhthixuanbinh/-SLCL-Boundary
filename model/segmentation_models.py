import segmentation_models_pytorch as smp
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np

# Assuming utils_ is importable if get_n_params is still used for prints
try:
    from utils.utils_ import get_n_params
except ImportError:
    def get_n_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


class segmentation_models(nn.Module):
    def __init__(self, name='resnet50', pretrained=False, 
                 in_channel=3, classes=4, 
                 decoder_channels=(256, 128, 64, 32, 16), # Provide default decoder_channels
                 multilvl=False, args=None): 
        super(segmentation_models, self).__init__()
        
        print(f"DEBUG: Initializing segmentation_models (ULTRA-SIMPLIFIED FOR TEST) with:")
        print(f"  name='{name}', pretrained={pretrained}, classes={classes}, in_channel={in_channel}")
        print(f"  decoder_channels for smp.Unet: {decoder_channels}")
        
        self.smp_model = smp.Unet(
            encoder_name=name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channel,
            classes=classes, 
            activation=None, # Get raw logits
            decoder_channels=decoder_channels 
        )
        print(f"DEBUG: smp.Unet model instantiated directly in segmentation_models.py.")
        # Store for compatibility with Trainer_baseline's prepare_model which might access self.encoder
        self.encoder = self.smp_model.encoder 
        print(f"Number of params (full smp.Unet model): {get_n_params(self.smp_model):,}")

        # Store args and multilvl for return signature compatibility for MCCL trainer if it uses this model
        self.args = args 
        self.multilvl = multilvl # From args, default False for baseline

    def forward(self, x, features_out=True): # features_out is now crucial
        # print(f"DEBUG: segmentation_models.forward called with input shape {x.shape}")
        
        # Standard smp.Unet forward pass steps (simplified to illustrate feature extraction)
        encoder_features = self.smp_model.encoder(x)
        # Typically, the decoder takes all encoder features.
        # The last feature map from the decoder (before the head) is what we need.
        decoder_output = self.smp_model.decoder(*encoder_features) # Unpack encoder features if decoder expects multiple

        # The segmentation_head is usually a Conv2d layer
        output_logits = self.smp_model.segmentation_head(decoder_output)
        
        # print(f"DEBUG: smp_model(x) output shape: {output_logits.shape}")

        if features_out:
            # decoder_output is likely the 'decoder_features' (dcdr_ft) you need
            # bottleneck_placeholder could be the last encoder feature or a pooled version
            # For simplicity, let's return the last encoder feature for 'bottleneck'
            # and the decoder_output for 'dcdr_ft'
            bottleneck_feature = encoder_features[-1] # Assuming encoder_features is a list/tuple
            dcdr_ft = decoder_output
            return output_logits, bottleneck_feature, dcdr_ft
        else:
            return output_logits


# --- PointNet and segmentation_model_point ---
# Kept minimal for now, as train_baseline.py uses `segmentation_models`
class PointNet(nn.Module):
    def __init__(self, **kwargs): super().__init__(); # Minimal
    def forward(self, x): return x # Pass-through

class segmentation_model_point(segmentation_models):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs) # Pass all arguments to the (now simplified) base
        encoder_out_channels = self.smp_model.encoder.out_channels[-1]
        fc_inch = kwargs.get('fc_inch', 4)
        extpn = kwargs.get('extpn', False)
        self.pointnet = PointNet(num_points=300, fc_inch=fc_inch, conv_inch=encoder_out_channels, ext=extpn)
        print(f'Model {kwargs.get("name", "N/A")} with PointNet loaded (using simplified base).')

    def forward(self, x, features_out=True):
        # For this simplified test, segmentation_model_point will also just rely on base smp_model output
        segmentation_output = self.smp_model(x)
        
        # Placeholder for point features
        point_features = torch.randn((x.shape[0], 300, 3), device=x.device) if hasattr(self, 'pointnet') else None
        output_aux = None
        
        if self.multilvl:
            return segmentation_output, output_aux, point_features
        elif features_out:
            return segmentation_output, None, point_features # pred, bottleneck_placeholder, point_features
        else:
            return segmentation_output


if __name__ == '__main__':
    from torch import rand
    class DummyArgs: phead = False; multilvl = False # Minimal args for testing base
    
    print("\nTesting ULTRA-SIMPLIFIED base segmentation_models:")
    model_base = segmentation_models(name='resnet50', pretrained=False, classes=4, args=DummyArgs())
    
    img = rand((2, 3, 224, 224))
    output_tuple = model_base(img, features_out=True) # Call as trainer would with features_out=True
    
    if isinstance(output_tuple, tuple) and len(output_tuple) == 3:
        pred_base, bn_placeholder, ft_placeholder = output_tuple
        print("Simplified Base Model - Pred Output shape:", pred_base.shape)
        if bn_placeholder is not None: print("Simplified Base Model - Bottleneck placeholder shape:", bn_placeholder.shape)
        else: print("Simplified Base Model - Bottleneck placeholder is None")
        if ft_placeholder is not None: print("Simplified Base Model - Features placeholder shape:", ft_placeholder.shape)
    else: # Single tensor output
        print("Simplified Base Model (features_out=False or not tuple) - Pred Output shape:", output_tuple.shape)