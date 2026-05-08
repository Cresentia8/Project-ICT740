# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os

# --- MODEL ---
class singlePerspectiveNet(nn.Module):
    def __init__(self):
        super(singlePerspectiveNet, self).__init__()
        self.combined_net = nn.Linear(768, 2048)
        # Instead of (8, 1) math on a reshaped 3D tensor, do one flat 2D calculation.
        self.tpu_layer = nn.Linear(2048, 256) 
        
    def forward(self, x):
        # x: [Batch, 768]
        x = self.combined_net(x)
        x = torch.clamp(x, 0, 1)
        # NO .view(-1, 256, 8) here! That is what causes Op 152.
        x = self.tpu_layer(x) 
        return x # [Batch, 256]

class ChaoticNet(nn.Module):
    def __init__(self):
        super(ChaoticNet, self).__init__()
        self.singlePerspectiveNet = singlePerspectiveNet()
        self.linear1 = nn.Linear(512, 32)
        self.linear2 = nn.Linear(64, 32)
        self.linear_out = nn.Linear(32, 1)
        self.skip_out = nn.Linear(512, 1, bias=False)
        
    def forward(self, x):
        side_a = self.singlePerspectiveNet(x[:, :768])
        side_b = self.singlePerspectiveNet(x[:, 768:])
        
        stacked_features = torch.cat((side_a, side_b), dim=1)
        stacked_features = torch.clamp(stacked_features, 0, 1)
        
        l1 = self.linear1(stacked_features)
        
        # BROKEN UP MATH: This prevents the converter from using BatchMatMul
        l1_sq = l1 * l1 
        l1_sq = l1_sq * 0.9921875 # This is 127/128
        
        # Concatenate and clamp
        combined = torch.cat((l1, l1_sq), dim=1)
        linear1_activated = torch.clamp(combined, 0, 1)
        
        linear2_out = torch.clamp(self.linear2(linear1_activated), 0, 1)
        
        # Use a simpler addition
        res = self.skip_out(stacked_features)
        final_output = res + self.linear_out(linear2_out)
        
        return final_output.view(-1) * 600

# --- EXPORT ---
def run_export():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(base_dir, "OnBorad", "model", "ChaoticNet8_epoch2.pt")
    
    if not os.path.exists(weight_path):
        # Add your absolute path here if the relative one fails
        weight_path = r"OnBorad\model\ChaoticNet8_epoch2.pt"

    print(f"Loading model and remapping weights...")
    model = ChaoticNet()

    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        # 1. Clean 'compile' prefixes
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        # 2. REMAP CONV3 TO LINEAR (The Magic Trick)
        # Weights in Conv1d are [Out, In, 1]. Linear wants [Out, In].
        if "singlePerspectiveNet.conv3.weight" in new_state_dict:
            print("Remapping conv3 weights...")
            w = new_state_dict.pop("singlePerspectiveNet.conv3.weight") # [1, 8, 1]
            b = new_state_dict.pop("singlePerspectiveNet.conv3.bias")   # [1]
            
            # Conv1d [Out, In, K] -> Linear [Out, In]
            # [1, 8, 1] -> [1, 8]
            new_state_dict["singlePerspectiveNet.tpu_friendly_conv.weight"] = w.view(1, 8)
            new_state_dict["singlePerspectiveNet.tpu_friendly_conv.bias"] = b
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print("Weights remapped and loaded successfully!")
    except Exception as e:
        print(f"Error remapping weights: {e}")
        return

    # --- Export as usual ---
    dummy_input = torch.zeros(1, 1536)
    onnx_path = os.path.join(base_dir, "chaotic_net.onnx")
    
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=11,
        input_names=['board_input'], output_names=['evaluation']
    )
    print(f"Done! ONNX ready: {onnx_path}")

if __name__ == "__main__":
    run_export()