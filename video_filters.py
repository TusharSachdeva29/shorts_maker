"""
Video filtering and style transfer module
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x


class TransformerNet(nn.Module):
    """Neural Style Transfer network architecture"""
    
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = F.relu(self.in2(self.conv2(y)))
        y = F.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = F.relu(self.in4(self.deconv1(y)))
        y = F.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class VideoFilter:
    """Applies filters to video frames"""
    
    def __init__(self, use_style_transfer=False, device='cpu'):
        self.use_style_transfer = use_style_transfer
        self.device = device
        self.style_model = None
        
        if use_style_transfer:
            self.load_style_model()
    
    def load_style_model(self):
        """Load pre-trained neural style transfer model"""
        print("\nLoading Neural Style Transfer model...")
        # Note: In production, you'd download and load pre-trained weights
        self.style_model = TransformerNet()
        # self.style_model.load_state_dict(torch.load("mosaic.pth"))
        self.style_model.to(self.device)
        self.style_model.eval()
        print("✅ NST model loaded.")
    
    def apply_cinematic_filter_opencv(self, frame):
        """
        Fast OpenCV-based cinematic filter
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Filtered frame
        """
        # Increase contrast and brightness
        alpha = 1.1  # Contrast
        beta = 5     # Brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Apply a "cool" tint
        b, g, r = cv2.split(adjusted)
        b = cv2.add(b, 15)
        r = cv2.subtract(r, 10)
        final_frame = cv2.merge((b, g, r))
        
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
        return final_frame
    
    def apply_style_transfer(self, frame):
        """
        Apply neural style transfer (slow but high quality)
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Styled frame
        """
        # Convert frame to tensor
        img_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(self.device)
        img_tensor = img_tensor.mul(255)
        
        with torch.no_grad():
            styled_tensor = self.style_model(img_tensor)
        
        # Convert back to numpy
        img = styled_tensor.clone().detach().squeeze(0)
        img = img.cpu().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        return img
    
    def get_filter_function(self):
        """
        Returns the appropriate filter function based on settings
        
        Returns:
            Function that takes a frame and returns filtered frame
        """
        if self.use_style_transfer and self.style_model:
            return self.apply_style_transfer
        else:
            return self.apply_cinematic_filter_opencv
