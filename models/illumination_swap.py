import torch
import torch.nn as nn


class ImageEncoding(nn.Module):
    def __init__(self, in_channels=3, enc_channels=32, out_channels=64):
        super(ImageEncoding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, enc_channels - in_channels, 7, padding=3)
        self.conv2 = nn.Conv2d(enc_channels, out_channels, 3, stride=2, padding=1)
        self.encoded_image = None
    
    def forward(self, image):
        x = self.conv1(image)
        self.encoded_image = torch.cat((x, image), dim=1)  # append image channels to the convolution result
        return self.conv2(self.encoded_image)
    
    def get_encoded_image(self):
        # should not require tensor copying (based on https://github.com/milesial/Pytorch-UNet)
        return self.encoded_image


class EncDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, channels_per_group=16):
        super(EncDoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(out_channels // channels_per_group, out_channels),
            nn.PReLU()
        )
        self.skip_connections = []

    def forward(self, x):
        self._reset_skip_connections()

        self.skip_connections.append(x)
        x = self.conv1(x)
        self.skip_connections.append(x)
        return self.conv2(x)

    def _reset_skip_connections(self):
        self.skip_connections = []

    def get_skip_connections(self):
        return self.skip_connections


class EncBottleneckConv(nn.Module):
    def __init__(self, in_channels, channels_per_group=16, envmap_H=16, envmap_W=32, depth=4):
        expected_channels = envmap_H * envmap_W
        assert in_channels == expected_channels, f'DownBottleneck input has {in_channels} channels, expected {expected_channels}'
        super(EncBottleneckConv, self).__init__()
        self.convolutions = self._build_bottleneck_convolutions(in_channels, channels_per_group, depth)
        self.skip_connections = []

    @staticmethod
    def _build_bottleneck_convolutions(in_channels, channels_per_group, depth):
        single_conv = [nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )]
        convolutions = single_conv * (depth - 1)

        # final layer before weighted pooling
        out_channels = 4 * in_channels
        convolutions.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels // channels_per_group, out_channels),
            nn.Softplus()
        ))

        return nn.ModuleList(convolutions)

    def forward(self, x):
        self._reset_skip_connections()

        for module in self.convolutions:
            self.skip_connections.append(x)
            x = module(x)

        # last layer output is not used as skip_connections:
        self.skip_connections.pop()

        # split x into environment map predictions and confidence
        channels = x.size()[1]
        split_point = 3 * (channels // 4)
        envmap_predictions, confidence = x[:, :split_point], x[:, split_point:]

        return envmap_predictions, confidence

    def _reset_skip_connections(self):
        self.skip_connections = []

    def get_skip_connections(self):
        return self.skip_connections


class WeightedPooling(nn.Module):
    def __init__(self):
        super(WeightedPooling, self).__init__()

    def forward(self, x, weights):
        # TODO: multiplication with sum can probably be implemented as convolution with groups (according to some posts)
        return (x * weights.repeat((1, 3, 1, 1))).sum(dim=(2, 3), keepdim=True)


class Tiling(nn.Module):
    def __init__(self, size=16, in_channels=1536, out_channels=512, channels_per_group=16):
        super(Tiling, self).__init__()
        self.size = size
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels // channels_per_group, out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        tiled = x.repeat((1, 1, self.size, self.size))
        return self.encode(tiled)


def channel_concat(x, y):
    return torch.cat((x, y), dim=1)


class DecBottleneckConv(nn.Module):
    def __init__(self, in_channels, channels_per_group=16, envmap_H=16, envmap_W=32, depth=4):
        expected_channels = envmap_H * envmap_W
        assert depth >= 2, f'Depth should be not smaller than 3'
        assert in_channels == expected_channels, f'UpBottleneck input has {in_channels} channels, expected {expected_channels}'
        super(DecBottleneckConv, self).__init__()
        self.depth = depth

        half_in_channels = in_channels // 2
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, half_in_channels, 3, padding=1),
            nn.GroupNorm(half_in_channels // channels_per_group, half_in_channels),
            nn.PReLU()
        )
        # TODO: why are these paddings necessary
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels + half_in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )
        self.convs = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )] * (depth - 3))
        # TODO: output_padding added to fit the spatial dimensions, but there is no reasoned justification for it
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, half_in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(half_in_channels // channels_per_group, half_in_channels),
            nn.PReLU()
        )

    def forward(self, x, skip_connections):
        # encoding convolution
        x = self.encode(x)

        # transposed convolutions with skip connections
        x = self.initial_conv(channel_concat(x, skip_connections.pop()))
        for conv in self.convs:
            x = conv(channel_concat(x, skip_connections.pop()))
        return self.out_conv(channel_concat(x, skip_connections.pop()))


class DecDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, channels_per_group=16):
        super(DecDoubleConv, self).__init__()
        # TODO: why are these paddings necessary
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(out_channels // channels_per_group, out_channels),
            nn.PReLU()
        )

    def forward(self, x, skip_connections):
        x = self.conv1(channel_concat(x, skip_connections.pop()))
        return self.conv2(channel_concat(x, skip_connections.pop()))


class Output(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(Output, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),  # should it be conv or transposed conv?
            nn.GroupNorm(1, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, encoded_img):
        return self.block(channel_concat(x, encoded_img))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.image_encoder = ImageEncoding()

        n_double_conv = 3
        self.double_convolutions = self._build_encoder_convolutions_block(n_double_conv)
        self.bottleneck = EncBottleneckConv(512)
        self.weighted_pool = WeightedPooling()

        self.skip_connections = []

    @staticmethod
    def _build_encoder_convolutions_block(n):
        return nn.ModuleList([EncDoubleConv(64 * (2 ** i), 64 * (2 ** (i + 1))) for i in range(n)])

    def forward(self, image):
        self._reset_skip_connections()

        # initial image encoding
        x = self.image_encoder(image)
        self.skip_connections.append(self.image_encoder.get_encoded_image())

        # double convolution layers
        for double_conv in self.double_convolutions:
            x = double_conv(x)
            self.skip_connections += double_conv.get_skip_connections()

        # pre-bottleneck layer
        env_map, weights = self.bottleneck(x)
        self.skip_connections += self.bottleneck.get_skip_connections()

        # predict environment map
        pred_env_map = self.weighted_pool(env_map, weights)

        return pred_env_map

    def _reset_skip_connections(self):
        self.skip_connections = []

    def get_skip_connections(self):
        return self.skip_connections


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        n_double_conv = 3
        self.tiling = Tiling()
        self.bottleneck = DecBottleneckConv(512)
        self.double_convolutions = self._build_decoder_convolutions_block(n_double_conv)

        self.output = Output()

    @staticmethod
    def _build_decoder_convolutions_block(n):
        return nn.ModuleList([DecDoubleConv(256 // (2 ** i), 256 // (2 ** (i + 1))) for i in range(n)])

    def forward(self, latent, skip_connections):
        # tiling and channel reduction
        tiled = self.tiling(latent)

        # post-bottleneck layer
        x = self.bottleneck(tiled, skip_connections)

        # double convolution layers
        for double_conv in self.double_convolutions:
            x = double_conv(x, skip_connections)

        # final layer constructing image
        relighted = self.output(x, skip_connections.pop())

        return relighted


class IlluminationSwapNet(nn.Module):
    def __init__(self):
        """
        Illumination swap network model based on "Single Image Portrait Relighting" (Sun et al., 2019).
        Autoencoder accepts two images as inputs - one to be relighted and one representing target lighting conditions.
        It learns to encode their environment maps in the latent representation. In the bottleneck the latent
        representations are swapped so that the decoder, using U-Net-like skip connections from the encoder,
        generates image with the original content but under the lighting conditions of the second input.
        """
        super(IlluminationSwapNet, self).__init__()
        self.encode = Encoder()
        self.decode = Decoder()

    def forward(self, image, target, ground_truth):
        # pass image through encoder
        image_env_map = self.encode(image)
        image_skip_connections = self.encode.get_skip_connections()

        # pass target through encoder
        target_env_map = self.encode(target)

        # pass ground-truth through encoder
        ground_truth_env_map = self.encode(ground_truth)

        # decode image with target env map
        relighted_image = self.decode(target_env_map, image_skip_connections)
        # can also add target relighting here

        # pass relighted image second time through the network to get its env map
        relighted_env_map = self.encode(relighted_image)

        return relighted_image, relighted_env_map, ground_truth_env_map
