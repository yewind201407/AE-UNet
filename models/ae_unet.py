import torch
import torch.nn as nn
import timm
from .modules import OutlookAttention, CFCM

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AE_UNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        
        # --- Encoder ---
        # Using a pre-trained Swin Transformer 'tiny' version as per the paper
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True)
        
        encoder_channels = self.encoder.feature_info.channels() # [96, 192, 384, 768]
        
        # --- Custom Modules Integration ---
        # Replace original patch merging with our CFCM
        self.encoder.patch_embed.proj = nn.Conv2d(3, encoder_channels[0], kernel_size=4, stride=4)
        self.encoder.layers[0].downsample = CFCM(encoder_channels[0], encoder_channels[1])
        self.encoder.layers[1].downsample = CFCM(encoder_channels[1], encoder_channels[2])
        self.encoder.layers[2].downsample = CFCM(encoder_channels[2], encoder_channels[3])
        
        # Integrate OAM to augment Swin blocks
        # Here we add it after each stage for feature refinement
        self.oam1 = OutlookAttention(dim=encoder_channels[0], num_heads=4, kernel_size=3)
        self.oam2 = OutlookAttention(dim=encoder_channels[1], num_heads=8, kernel_size=3)
        self.oam3 = OutlookAttention(dim=encoder_channels[2], num_heads=16, kernel_size=3)
        self.oam4 = OutlookAttention(dim=encoder_channels[3], num_heads=32, kernel_size=3)

        # --- Decoder ---
        self.decoder4 = DecoderBlock(encoder_channels[3], encoder_channels[2], 256)
        self.decoder3 = DecoderBlock(256, encoder_channels[1], 128)
        self.decoder2 = DecoderBlock(128, encoder_channels[0], 64)
        self.decoder1 = DecoderBlock(64, 0, 32)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        # Stage 0
        s0 = self.encoder.patch_embed(x)
        
        # Stage 1
        s1 = self.encoder.layers[0](s0)
        s1_oam = self.oam1(s1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Stage 2
        s2 = self.encoder.layers[1](s1)
        s2_oam = self.oam2(s2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Stage 3
        s3 = self.encoder.layers[2](s2)
        s3_oam = self.oam3(s3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Stage 4 (Bottleneck)
        s4 = self.encoder.layers[3](s3)
        s4_oam = self.oam4(s4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # --- Decoder Path ---
        d4 = self.decoder4(s4_oam, s3_oam)
        d3 = self.decoder3(d4, s2_oam)
        d2 = self.decoder2(d3, s1_oam)
        d1 = self.decoder1(d2, s0) # No skip connection for the first decoder block
        
        # Final upsampling and convolution
        out = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        
        return out

if __name__ == '__main__':
    # Test the model with a dummy input
    import torch.nn.functional as F
    model = AE_UNet(n_classes=1).cuda()
    dummy_input = torch.randn(2, 3, 224, 224).cuda()
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")