# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from abc import ABC

# =======
# Encoder
# =======

class EncoderPart(nn.Module, ABC):
    def __init__(self):
        super(EncoderPart, self).__init__()
    
    #@abstractmethod
    def forward(self, x):
        return NotImplemented
    
    #@abstractmethod
    def get_skip_connections(self):
        return NotImplemented
    
    
class ImageEncoding(EncoderPart):
    def __init__(self, in_channels=3, enc_channels=32, out_channels=64):
        super(ImageEncoding, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, enc_channels - in_channels, 7, padding=3),
                        nn.BatchNorm2d(enc_channels - in_channels), #nn.GroupNorm(1, enc_channels - in_channels),
                        nn.PReLU()
                     )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(enc_channels, out_channels, 3, stride=2, padding=1),
                        nn.BatchNorm2d(out_channels), #nn.GroupNorm(out_channels // 16, out_channels),
                        nn.PReLU()
                     )
        self.encoded_image = None
    
    def forward(self, image):
        x = self.conv1(image)
        self.encoded_image = torch.cat((x, image), dim=1)  # append image channels to the convolution result
        return self.conv2(self.encoded_image)
    
    def get_skip_connections(self):
        # should not require tensor copying (based on https://github.com/milesial/Pytorch-UNet)
        return [self.encoded_image]
    
class EncDoubleConv(EncoderPart):
    def __init__(self, in_channels, out_channels, channels_per_group=16):
        super(EncDoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), #nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), #nn.GroupNorm(out_channels // channels_per_group, out_channels),
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


class EncBottleneckConv(EncoderPart):
    def __init__(self, in_channels, channels_per_group=16, depth=4):
        super(EncBottleneckConv, self).__init__()
        self.convolutions = self._build_bottleneck_convolutions(in_channels, channels_per_group, depth)
        self.skip_connections = []

    @staticmethod
    def _build_bottleneck_convolutions(in_channels, channels_per_group, depth):
        single_conv = [nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), #nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )]
        convolutions = single_conv * (depth - 1)

        return nn.ModuleList(convolutions)

    def forward(self, x):
        self._reset_skip_connections()

        for module in self.convolutions:
            self.skip_connections.append(x)
            x = module(x)

        return x

    def _reset_skip_connections(self):
        self.skip_connections = []

    def get_skip_connections(self):
        return self.skip_connections

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.image_encoder = ImageEncoding()

        n_double_conv = 3
        self.double_convolutions = self._build_encoder_convolutions_block(n_double_conv)
        self.bottleneck = EncBottleneckConv(512)
        self.skip_connections = []

    @staticmethod
    def _build_encoder_convolutions_block(n):
        return nn.ModuleList([EncDoubleConv(64 * (2 ** i), 64 * (2 ** (i + 1))) for i in range(n)])

    def forward(self, image):
        self._reset_skip_connections()

        # initial image encoding
        x = self.image_encoder(image)
        self.skip_connections += self.image_encoder.get_skip_connections()

        # double convolution layers
        for double_conv in self.double_convolutions:
            x = double_conv(x)
            self.skip_connections += double_conv.get_skip_connections()

        # pre-bottleneck layer
        x = self.bottleneck(x)
        self.skip_connections += self.bottleneck.get_skip_connections()

        return x

    def _reset_skip_connections(self):
        self.skip_connections = []

    def get_skip_connections(self):
        return self.skip_connections

# ===========================
# IlluminationSwapNetSplitter
# ===========================

class WeightedPooling(nn.Module):
    def __init__(self, in_channels=512, channels_per_group=16, envmap_H=16, envmap_W=32):
        expected_channels = envmap_H * envmap_W
        assert in_channels == expected_channels, f'WeightedPooling input has {in_channels} channels, expected {expected_channels}'
        
        super(WeightedPooling, self).__init__()
        
        # final conv before weighted average
        out_channels = 4 * in_channels
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels), #nn.GroupNorm(out_channels // channels_per_group, out_channels),
                        nn.Softplus()
                    )

    def forward(self, x):
        x = self.conv(x)
        
        # split x into environment map predictions and confidence
        channels = x.size()[1]
        split_point = 3 * (channels // 4)
        envmap_predictions, confidence = x[:, :split_point], x[:, split_point:]
        
        # TODO: multiplication with sum can probably be implemented as convolution with groups (according to some posts)
        return (envmap_predictions * confidence.repeat((1, 3, 1, 1))).sum(dim=(2, 3), keepdim=True)


class Tiling(nn.Module):
    def __init__(self, size=16, in_channels=1536, out_channels=512, channels_per_group=16):
        super(Tiling, self).__init__()
        self.size = size
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), #nn.GroupNorm(out_channels // channels_per_group, out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        tiled = x.repeat((1, 1, self.size, self.size))
        return self.encode(tiled)

class IlluminationSwapNetSplitter(nn.Module):
    def __init__(self):
        super(IlluminationSwapNetSplitter, self).__init__()        
        self.weighted_pool = WeightedPooling()

    def forward(self, latent):        
        pred_env_map = self.weighted_pool(latent)
        scene_latent = None
        light_latent = pred_env_map
        return scene_latent, light_latent

class IlluminationSwapNetUnsplitter(nn.Module):
    def __init__(self):
        super(IlluminationSwapNetUnsplitter, self).__init__()        
        self.tiling = Tiling()

    def forward(self, scene_latent, light_latent):        
        latent = self.tiling(light_latent)
        return latent
    
# ======================
# AnOtherSwapNetSplitter
# ======================
        
def size_splits(tensor, split_sizes, dim=0):
    #https://github.com/pytorch/pytorch/issues/3223
    """Splits the tensor according to chunks of split_sizes.
    
    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()
    
    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")
    
    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length)) 
        for start, length in zip(splits, split_sizes))

class AnOtherSwapNetSplitter(nn.Module):
    def __init__(self):
        super(AnOtherSwapNetSplitter, self).__init__()       
    def forward(self, latent):        
        light_latent, scene_latent = size_splits(latent, [3,509], dim=1)
        return scene_latent, light_latent

class AnOtherSwapNetUnsplitter(nn.Module):
    def __init__(self):
        super(AnOtherSwapNetUnsplitter, self).__init__()        

    def forward(self, scene_latent, light_latent):    
        latent = torch.cat((light_latent, scene_latent), dim = 1)
        return latent

# ======================
# Splitter512x1x1
# ======================
        
class Splitter512x1x1(nn.Module):
    def __init__(self):
        super(Splitter512x1x1, self).__init__() 
        self.to1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16),
            nn.BatchNorm2d(512), #nn.GroupNorm(512 // 16, 512),
            nn.PReLU()
        )
    def forward(self, latent):      
        latent = self.to1(latent) 
        light_latent, scene_latent = size_splits(latent, [16,512-16], dim=1)
        return scene_latent, light_latent

class Unsplitter512x1x1(nn.Module):
    def __init__(self):
        super(Unsplitter512x1x1, self).__init__()  
        self.to16 = nn.Sequential(
            nn.Upsample(size=(16, 16)),
            nn.BatchNorm2d(512), #nn.GroupNorm(512 // 16, 512),
            nn.PReLU()
        )

    def forward(self, scene_latent, light_latent):    
        latent = torch.cat((light_latent, scene_latent), dim = 1)  
        latent = self.to16(latent) 
        return latent
    
# =======
# Decoder
# =======
def channel_concat(x, y):
    return torch.cat((x, y), dim=1)


class DecoderPart(nn.Module, ABC):
    def __init__(self):
        super(DecoderPart, self).__init__()
    
    #@abstractmethod
    def forward(self, x, skip_connections):
        return NotImplemented
    


class DecBottleneckConv(DecoderPart):
    def __init__(self, in_channels, channels_per_group=16, envmap_H=16, envmap_W=32, depth=4):
        expected_channels = envmap_H * envmap_W
        assert depth >= 2, f'Depth should be not smaller than 3'
        assert in_channels == expected_channels, f'UpBottleneck input has {in_channels} channels, expected {expected_channels}'
        super(DecBottleneckConv, self).__init__()
        self.depth = depth

        half_in_channels = in_channels // 2
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, half_in_channels, 3, padding=1),
            nn.BatchNorm2d(half_in_channels), #nn.GroupNorm(half_in_channels // channels_per_group, half_in_channels),
            nn.PReLU()
        )
        # TODO: why are these paddings necessary
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels + half_in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), #nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), #nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )] * (depth - 3))
        # TODO: output_padding added to fit the spatial dimensions, but there is no reasoned justification for it
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(2 * in_channels, half_in_channels, 3, padding=1),#nn.ConvTranspose2d(2 * in_channels, half_in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(half_in_channels), #nn.GroupNorm(half_in_channels // channels_per_group, half_in_channels),
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


class DecDoubleConv(DecoderPart):
    def __init__(self, in_channels, out_channels, channels_per_group=16):
        super(DecDoubleConv, self).__init__()
        # TODO: why are these paddings necessary
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), #nn.GroupNorm(in_channels // channels_per_group, in_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(2 * in_channels, out_channels, 3, padding=1),#nn.ConvTranspose2d(2 * in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels), #nn.GroupNorm(out_channels // channels_per_group, out_channels),
            nn.PReLU()
        )

    def forward(self, x, skip_connections):
        x = self.conv1(channel_concat(x, skip_connections.pop()))
        return self.conv2(channel_concat(x, skip_connections.pop()))


class Output(DecoderPart):
    def __init__(self, in_channels=64, out_channels=3, kernel_size=3):
        super(Output, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),  # should it be conv or transposed conv?
            #nn.GroupNorm(1, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, encoded_img):
        return self.block(channel_concat(x, encoded_img))

class Decoder(nn.Module):
    def __init__(self, last_kernel_size=3):
        super(Decoder, self).__init__()

        n_double_conv = 3
        self.bottleneck = DecBottleneckConv(512)
        self.double_convolutions = self._build_decoder_convolutions_block(n_double_conv)

        self.output = Output(kernel_size=last_kernel_size)

    @staticmethod
    def _build_decoder_convolutions_block(n):
        return nn.ModuleList([DecDoubleConv(256 // (2 ** i), 256 // (2 ** (i + 1))) for i in range(n)])

    def forward(self, latent, skip_connections):

        # post-bottleneck layer
        x = self.bottleneck(latent, skip_connections)

        # double convolution layers
        for double_conv in self.double_convolutions:
            x = double_conv(x, skip_connections)

        # final layer constructing image
        relighted = self.output(x, skip_connections.pop())

        return relighted



# =======
# SwapNet
# =======
        
class SwapNet(nn.Module):
    def __init__(self, splitter, unsplitter, last_kernel_size=3):
        super(SwapNet, self).__init__()
        self.encode = Encoder()
        self.split = splitter
        self.unsplit = unsplitter
        self.decode = Decoder(last_kernel_size=last_kernel_size)

    def forward(self, image, target, groundtruth):
        # pass image through encoder
        image_latent = self.encode(image)
        image_skip_connections = self.encode.get_skip_connections()
        # pass target through encoder
        target_latent = self.encode(target)
        # pass ground-truth through encoder
        groundtruth_latent = self.encode(groundtruth)
        
        # pass image_latent through splitter
        image_scene_latent, image_light_latent = self.split(image_latent)
        # pass target_latent through splitter
        target_scene_latent, target_light_latent = self.split(target_latent)      
        # pass groundtruth_latent through splitter
        groundtruth_scene_latent, groundtruth_light_latent= self.split(groundtruth_latent)
        
        swapped_latent = self.unsplit(image_scene_latent, target_light_latent)

        # decode image with target env map
        relit_image = self.decode(swapped_latent, image_skip_connections)
        
        # decode image with its env map
        # reconstructed_image = self.decode(image_env_map, image_skip_connections)

        # pass relighted image second time through the network to get its env map
        # relighted_env_map = self.encode(relighted_image)

        return relit_image, \
               image_light_latent, target_light_latent, groundtruth_light_latent, \
               image_scene_latent, target_scene_latent, groundtruth_scene_latent
               
class IlluminationSwapNet(SwapNet):
    def __init__(self, last_kernel_size=3):
        """
        Illumination swap network model based on "Single Image Portrait Relighting" (Sun et al., 2019).
        Autoencoder accepts two images as inputs - one to be relighted and one representing target lighting conditions.
        It learns to encode their environment maps in the latent representation. In the bottleneck the latent
        representations are swapped so that the decoder, using U-Net-like skip connections from the encoder,
        generates image with the original content but under the lighting conditions of the second input.
        """
        super(IlluminationSwapNet, self).__init__(splitter = IlluminationSwapNetSplitter(),
                                                  unsplitter = IlluminationSwapNetUnsplitter(),
                                            last_kernel_size=last_kernel_size)
    def forward(self, image, target, ground_truth):
        relit_image, \
        image_light_latent, target_light_latent, groundtruth_light_latent, \
        image_scene_latent, target_scene_latent, groundtruth_scene_latent = \
        super(IlluminationSwapNet, self).forward(image, target, ground_truth)
        return relit_image,\
        image_light_latent.view(-1, 3, 16, 32), target_light_latent.view(-1, 3, 16, 32), groundtruth_light_latent.view(-1, 3, 16, 32),\
        image_scene_latent, target_scene_latent, groundtruth_scene_latent #those are None
        
class AnOtherSwapNet(SwapNet):
    def __init__(self, last_kernel_size=3):
        super(AnOtherSwapNet, self).__init__(splitter = AnOtherSwapNetSplitter(),
                                             unsplitter = AnOtherSwapNetUnsplitter(),
                                            last_kernel_size=last_kernel_size)
    def forward(self, image, target, ground_truth):
        return super(AnOtherSwapNet, self).forward(image, target, ground_truth)

class SwapNet512x1x1(SwapNet):
    def __init__(self, last_kernel_size=3):
        super(SwapNet512x1x1, self).__init__(splitter = Splitter512x1x1(),
                                              unsplitter = Unsplitter512x1x1(),
                                            last_kernel_size=last_kernel_size)
    def forward(self, image, target, ground_truth):
        relit_image, \
        image_light_latent, target_light_latent, groundtruth_light_latent, \
        image_scene_latent, target_scene_latent, groundtruth_scene_latent = \
        super(SwapNet512x1x1, self).forward(image, target, ground_truth)
        return relit_image,\
        image_light_latent.view(-1, 1, 4, 4), target_light_latent.view(-1, 1, 4, 4), groundtruth_light_latent.view(-1, 1, 4, 4),\
        image_scene_latent, target_scene_latent, groundtruth_scene_latent 
        