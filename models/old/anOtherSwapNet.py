#code is based on https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/models/DispNetS.py, but architecture is modified.

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_planes)
    )


def predict_disp(in_planes, out_planes = 1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_planes)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(out_planes)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

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

class Encoder(nn.Module):
    def __init__(self, conv_planes = [32, 64, 128, 256, 512, 512, 512]):
        super(Encoder, self).__init__()
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])
    def forward(self, x, skip_links=False):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)  
        latent = out_conv7
        if skip_links:
            return out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6, latent
        else: 
            return latent
        
class Spliter(nn.Module):
    def __init__(self):
        super(Spliter, self).__init__()
    def forward(self, x_latent):        
        illumination_x, scene_x = size_splits(x_latent, [256,256], dim=1)
        return illumination_x, scene_x
        
class Swaper(nn.Module):
    def __init__(self, spliter):
        super(Swaper, self).__init__()
        self.spliter = spliter
    def forward(self, x1_latent, x2_latent):        
        illumination_x1, scene_x1 = self.spliter(x1_latent)
        illumination_x2, scene_x2 = self.spliter(x2_latent)
        swapped = torch.cat((illumination_x2, scene_x1), dim = 1)
        return swapped
         
class Decoder(nn.Module):
    def __init__(self, conv_planes = [32, 64, 128, 256, 512, 512, 512], upconv_planes = [512, 512, 256, 128, 64, 32, 16]):
        super(Decoder, self).__init__()
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])
        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = conv(upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = conv(upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = conv(upconv_planes[6] + 3, upconv_planes[6])
        self.predict = predict_disp(upconv_planes[6],3)
    def forward(self, input1, out_conv1, out_conv2, out_conv3, out_conv4, out_conv5, out_conv6, swap_encoded):
        out_upconv7 = crop_like(self.upconv7(swap_encoded), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)
        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)
        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)
        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2), 1)
        out_iconv3 = self.iconv3(concat3)
        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1), 1)
        out_iconv2 = self.iconv2(concat2)
        out_upconv1 = crop_like(self.upconv1(out_iconv2), input1)
        concat1 = torch.cat((out_upconv1, input1), 1)
        out_iconv1 = self.iconv1(concat1)
        return out_iconv1

class Predicter(nn.Module):
    def __init__(self, in_planes = 16, out_planes = 3):
        super(Predicter, self).__init__()
        self.predict = predict_disp(in_planes, out_planes)
    def forward(self, x):
        return self.predict(x)
        
class IlluminationPredicter(nn.Module):
    def __init__(self, in_size = 256, out_reals = 2):
        super(IlluminationPredicter, self).__init__()
        self.fc = nn.Conv2d(in_size, out_reals, kernel_size=1)
    def forward(self, x):
        return self.fc(x)
        
class SwapModel(nn.Module):
    def __init__(self):
        super(SwapModel, self).__init__()
        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.encoder = Encoder(conv_planes=conv_planes)
        self.spliter = Spliter()
        self.swaper = Swaper(spliter = self.spliter)
        self.decoder = Decoder(conv_planes=conv_planes, upconv_planes=upconv_planes)
        self.predicter = Predicter(in_planes=upconv_planes[-1], out_planes=3)     
        self.illuminationPredicter = IlluminationPredicter(in_size = 256, out_reals = 2)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x1, x2, x3 = None):
        x1_encoded = self.encoder(x1, skip_links = True)
        x2_encoded = self.encoder(x2, skip_links = False)
        o1, o2, o3, o4, o5, o6, latent_x1 = x1_encoded
        latent_x2 = x2_encoded
        latent_swapped = self.swaper(latent_x1, latent_x2)
        out = self.decoder(x1, o1, o2, o3, o4, o5, o6, latent_swapped)
        pred = self.predicter(out)
        if x3 == None:
            return pred
        else:
            x3_encoded = self.encoder(x3, skip_links = False)
            latent_x3 = x3_encoded
            illumination_x1, scene_x1 = self.spliter(latent_x1)
            illumination_x2, scene_x2 = self.spliter(latent_x2)
            illumination_x3, scene_x3 = self.spliter(latent_x3)  
            ill_pred_x1 = self.illuminationPredicter(illumination_x1)
            ill_pred_x2 = self.illuminationPredicter(illumination_x2)
            ill_pred_x3 = self.illuminationPredicter(illumination_x3)
            return scene_x3, scene_x1, scene_x2, \
                   illumination_x3, illumination_x1, illumination_x2, \
                   ill_pred_x3, ill_pred_x1, ill_pred_x2, \
                   pred