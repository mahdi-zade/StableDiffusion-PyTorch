import torch
import torch.nn as nn
from einops import rearrange

class UNet(nn.Module):
    """
    UNet model for saliency map reconstruction.
    Simplified and tailored for image-to-image tasks.
    """

    def __init__(self, im_channels=3, saliency_channels=1, base_channels=64, num_downs=4, time_emb_dim=128):
        """
        Initialize the UNet model.
        :param im_channels: Number of input image channels (default: 3 for RGB).
        :param saliency_channels: Number of output channels (default: 1 for saliency maps).
        :param base_channels: Number of base channels in the first layer (default: 64).
        :param num_downs: Number of downsampling steps (default: 4).
        :param time_emb_dim: Dimension of time embedding (default: 128).
        """
        super(UNet, self).__init__()

        self.im_channels = im_channels
        self.saliency_channels = saliency_channels
        self.base_channels = base_channels
        self.num_downs = num_downs
        self.time_emb_dim = time_emb_dim

        # Time embedding projection
        self.t_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoding (Downsampling) layers
        self.downs = nn.ModuleList()
        in_channels = im_channels
        for i in range(num_downs):
            out_channels = base_channels * (2 ** i)
            self.downs.append(self._block(in_channels, out_channels, downsample=True))
            in_channels = out_channels

        # Bottleneck layers
        bottleneck_channels = base_channels * (2 ** num_downs)
        self.bottleneck = self._block(in_channels, bottleneck_channels, downsample=False)

        # Decoding (Upsampling) layers
        self.ups = nn.ModuleList()
        for i in reversed(range(num_downs)):
            out_channels = base_channels * (2 ** i)
            self.ups.append(self._block(in_channels, out_channels, upsample=True))
            in_channels = out_channels

        # Final convolution layer
        self.final_conv = nn.Conv2d(base_channels, saliency_channels, kernel_size=3, padding=1)

    def _block(self, in_channels, out_channels, downsample=False, upsample=False):
        """
        UNet block: two convolutional layers with optional downsampling or upsampling.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU()
        ]
        if downsample:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
        if upsample:
            layers.append(nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x, t=None):
        """
        Forward pass for the UNet.
        :param x: Input image tensor (B, C, H, W).
        :param t: Optional time step tensor (B,).
        :return: Output saliency map tensor (B, 1, H, W).
        """
        t_emb = None
        if t is not None:
            t_emb = self.t_proj(t)

        # Downsampling
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        skips = skips[::-1]
        for i, up in enumerate(self.ups):
            x = up(x)
            if i < len(skips):
                x = torch.cat([x, skips[i]], dim=1)

        # Final convolution
        x = self.final_conv(x)
        return x
