import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from models.networks import get_norm_layer, init_net, ResnetBlock



def define_ColorG(output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):

  
    norm_layer = get_norm_layer(norm_type=norm)

    net = MultiscaleResnetGenerator(output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_ids) 

def define_WaveG(ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)

    net = WaveResnetGenerator(ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain, gpu_ids) 

def define_WaveD(input_nc, ndf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)

    net = WaveNLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)

    return init_net(net, init_type, init_gain, gpu_ids)





class WaveResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)

    """

    def __init__(self, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(WaveResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        ##############################################
        ############ feature extraction
        model_first = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2)]
                 #nn.ReLU(True)]
        self.model_first = nn.Sequential(*model_first)
        ############ downsample: for high feature
        model_downsample_1 = [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 2),
                      nn.LeakyReLU(0.2)]        
        self.model_downsample_1 = nn.Sequential(*model_downsample_1)  
        ############ downsample: for low feature
        model_downsample_2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.LeakyReLU(0.2)]        
        self.model_downsample_2 = nn.Sequential(*model_downsample_2)  
        ############ resnet block
        model_resnet_high = []                           
        for i in range(n_blocks-3):       # add ResNet blocks
            model_resnet_high += [ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_high = nn.Sequential(*model_resnet_high)

        model_resnet_low = []                           
        for i in range(n_blocks):       # add ResNet blocks
            model_resnet_low += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_low = nn.Sequential(*model_resnet_low)        
        ############ upsample: for low feature
        model_upsample = [nn.Conv2d(ngf * 4 , ngf * 16, kernel_size=3, stride=1, padding=1, bias=use_bias), 
                  nn.PixelShuffle(2),
                  norm_layer(ngf * 4),
                  nn.LeakyReLU(0.2)]
                  #nn.ReLU(True)]
        model_upsample += [nn.Conv2d(ngf * 4 , ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf *2),
                  nn.LeakyReLU(0.2)]
                  #nn.ReLU(True)]                       
        self.model_upsample = nn.Sequential(*model_upsample)  
        ############                           
        model_last_high = [nn.ReflectionPad2d(3)]
        model_last_high += [nn.Conv2d(ngf *2 , 3, kernel_size=7, padding=0)]
        model_last_high += [nn.LeakyReLU(0.2)]
        self.model_last_high = nn.Sequential(*model_last_high) 

        model_last_low = [nn.ReflectionPad2d(3)]
        model_last_low += [nn.Conv2d(ngf *2 , 1, kernel_size=7, padding=0)]
        # model_last_low += [nn.ReLU(0.2)]
        model_last_low += [nn.LeakyReLU(0.2)]
        self.model_last_low = nn.Sequential(*model_last_low)        
           
                          
    def forward(self, input):
        """Standard forward"""
        ##### grayimg --> feature
        feature_first = self.model_first(input)
        feature_downsample_1 = self.model_downsample_1(feature_first)
        feature_downsample_2 = self.model_downsample_2(feature_downsample_1)
        ##### feature_low(cA)
        feature_resnet_low = self.model_resnet_low(feature_downsample_2)
        feature_upsample = self.model_upsample(feature_resnet_low)   
        feature_last_low = self.model_last_low(feature_upsample) # (1,1,128,128)
        ##### feature_high(cH/cV/cD)
        feature_resnet_high = self.model_resnet_high(feature_downsample_1)
        feature_last_high = self.model_last_high(feature_resnet_high)
        feature_last_high = feature_last_high.unsqueeze(1) # (1,1,3,128,128)  

        return  feature_last_low,feature_last_high




class WaveNLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(WaveNLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1     
        ######## structure_d
        first = [nn.Conv2d(input_nc, ndf, kernel_size=2, stride=2, padding=0), nn.LeakyReLU(0.2, True)]
        self.first = nn.Sequential(*first)
        downsample = []
        nf_mult = 1
        nf_mult_prev = 1
        temp = 0
        for n in range(1, n_layers-1):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            downsample += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=2, stride=2, padding=0, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            temp = n
        self.downsample = nn.Sequential(*downsample)

        ############################# for gray
        sequence_c1 = []
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** temp, 8)        
        sequence_c1 += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]  
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)                   
        sequence_c1 += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence_c1 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model_c1 = nn.Sequential(*sequence_c1)
        ########################## for wavelet feature
        sequence_c4 = []
        sequence_c4 += [nn.Conv2d(ndf * nf_mult_prev, 1, kernel_size=2, stride=2, padding=0)]
        self.model_c4 = nn.Sequential(*sequence_c4)

    def forward(self, input1, input2):
        """Standard forward."""
        
        feature_c1 = self.first(input1)
        feature_c1 = self.downsample(feature_c1)
        feature_c1 = self.model_c1(feature_c1)
        feature_c4 = self.first(input2)
        feature_c4 = self.downsample(feature_c4)
        feature_c4 = self.model_c4(feature_c4)

        return feature_c1, feature_c4





class MultiscaleResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(MultiscaleResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
                                   
        ############################ feature extraction
        ###### for gray image
        model_first_1 = [nn.ReflectionPad2d(3),
                 nn.Conv2d(1, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, True)]                
        self.model_first_1 = nn.Sequential(*model_first_1) #####(n,ngf,256,256)  grayimg
        ###### for low feature
        model_first_2 = [
                 nn.Conv2d(1, ngf * 2, kernel_size=3, padding=1, bias=use_bias),
                 norm_layer(ngf*2),
                 nn.LeakyReLU(0.2, True)]
        self.model_first_2 = nn.Sequential(*model_first_2) #####(n,ngf,256,256)  cA_wavelet_feature
        ###### for high features
        model_first_3 = [
                 nn.Conv2d(3, ngf * 2, kernel_size=3, padding=1, bias=use_bias),
                 norm_layer(ngf*2),
                 nn.LeakyReLU(0.2, True)]
        self.model_first_3 = nn.Sequential(*model_first_3) #####(n,ngf,256,256) cH/V/D_wavelet_feature

        ############################ downsample
        ###### for gray-image features
        model_downsample_1_1 = [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 2),
                      nn.LeakyReLU(0.2, True)]                
        self.model_downsample_1_1 = nn.Sequential(*model_downsample_1_1) 
        ###### for fused features
        model_downsample_1_2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.LeakyReLU(0.2, True)]                
        self.model_downsample_1_2 = nn.Sequential(*model_downsample_1_2) 

        model_downsample_1_3 = [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 8),
                      nn.LeakyReLU(0.2, True)]                
        self.model_downsample_1_3 = nn.Sequential(*model_downsample_1_3) 

        model_downsample_1_4 = [nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 16),
                      nn.LeakyReLU(0.2, True)]                
        self.model_downsample_1_4 = nn.Sequential(*model_downsample_1_4) 
        ###### for high-frequency features
        model_downsample_3_1 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4),
                      nn.LeakyReLU(0.2, True)]
        self.model_downsample_3_1 = nn.Sequential(*model_downsample_3_1)
        model_downsample_3_2 = [nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 8),
                      nn.LeakyReLU(0.2, True)]
        self.model_downsample_3_2 = nn.Sequential(*model_downsample_3_2)
        model_downsample_3_3 = [nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 16),
                      nn.LeakyReLU(0.2, True)]
        self.model_downsample_3_3 = nn.Sequential(*model_downsample_3_3)

        ############################ resnet blocks
        ###### for fused features: style
        model_resnet_1_1 = []                           
        for i in range(n_blocks):       # add ResNet blocks
            model_resnet_1_1 += [ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_1_1 = nn.Sequential(*model_resnet_1_1)   
        model_resnet_1_2 = []                           
        for i in range(n_blocks-3):       # add ResNet blocks
            model_resnet_1_2 += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_1_2 = nn.Sequential(*model_resnet_1_2)
        model_resnet_1_3 = []  
        for i in range(n_blocks-5):       # add ResNet blocks
            model_resnet_1_3 += [ResnetBlock(ngf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_1_3 = nn.Sequential(*model_resnet_1_3) 
        model_resnet_1_4 = [] 
        for i in range(n_blocks-7):       # add ResNet blocks
            model_resnet_1_4 += [ResnetBlock(ngf * 16, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_1_4 = nn.Sequential(*model_resnet_1_4)

        ###### for high-frequency features: detail
        model_resnet_3_1 = []                           
        for i in range(n_blocks):       # add ResNet blocks
            model_resnet_3_1 += [ResnetBlock(ngf * 2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_3_1 = nn.Sequential(*model_resnet_3_1) 
        model_resnet_3_2 = []                           
        for i in range(n_blocks-3):       # add ResNet blocks
            model_resnet_3_2 += [ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_3_2 = nn.Sequential(*model_resnet_3_2)        
        model_resnet_3_3 = []                           
        for i in range(n_blocks-5):       # add ResNet blocks
            model_resnet_3_3 += [ResnetBlock(ngf * 8, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_3_3 = nn.Sequential(*model_resnet_3_3)   
        model_resnet_3_4 = []                           
        for i in range(n_blocks-7):       # add ResNet blocks
            model_resnet_3_4 += [ResnetBlock(ngf * 16, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.model_resnet_3_4 = nn.Sequential(*model_resnet_3_4)

        ############################ upsample

        model_upsample_1 = []
        model_upsample_1 += [nn.Conv2d(ngf * 16 , ngf * 64, kernel_size=3, stride=1, padding=1, bias=use_bias), 
                  nn.PixelShuffle(2),
                  norm_layer(ngf * 16),
                  nn.LeakyReLU(0.2)]                 
        model_upsample_1 += [nn.Conv2d(ngf * 16 , ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * 8),
                  nn.LeakyReLU(0.2)]
        self.model_upsample_1 = nn.Sequential(*model_upsample_1)  

        model_upsample_2 = []
        model_upsample_2 += [nn.Conv2d(ngf * 8 , ngf * 32, kernel_size=3, stride=1, padding=1, bias=use_bias), 
                  nn.PixelShuffle(2),
                  norm_layer(ngf * 8),
                  nn.LeakyReLU(0.2)]                 
        model_upsample_2 += [nn.Conv2d(ngf * 8 , ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * 4),
                  nn.LeakyReLU(0.2)]
        self.model_upsample_2 = nn.Sequential(*model_upsample_2)  

        model_upsample_3 = []
        model_upsample_3 += [nn.Conv2d(ngf * 4 , ngf * 16, kernel_size=3, stride=1, padding=1, bias=use_bias), 
                  nn.PixelShuffle(2),
                  norm_layer(ngf * 4),
                  nn.LeakyReLU(0.2)]                 
        model_upsample_3 += [nn.Conv2d(ngf * 4 , ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf * 2),
                  nn.LeakyReLU(0.2)]
        self.model_upsample_3 = nn.Sequential(*model_upsample_3)   

        model_upsample_4 = []               
        model_upsample_4 += [nn.Conv2d(ngf * 2, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), 
                  nn.PixelShuffle(2),
                  norm_layer(ngf * 2),
                  nn.LeakyReLU(0.2)] 
        model_upsample_4 += [nn.Conv2d(ngf * 2 , ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(ngf),
                  nn.LeakyReLU(0.2)]
        model_upsample_4 += [nn.ReflectionPad2d(3)]
        model_upsample_4 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample_4 += [nn.Tanh()]
        self.model_upsample_4 = nn.Sequential(*model_upsample_4) 

        ############################ gray-image features and low-frequency features
        self.conv1x1_ngf2 = nn.Conv2d(ngf * 4, ngf * 2, 1, 1, 0)

        ###########################
        self.resnet_style = nn.ModuleList()
        self.resnet_style.append(self.model_resnet_1_1)
        self.resnet_style.append(self.model_resnet_1_2)
        self.resnet_style.append(self.model_resnet_1_3)
        self.resnet_style.append(self.model_resnet_1_4)
        self.resnet_detail = nn.ModuleList()
        self.resnet_detail.append(self.model_resnet_3_1)
        self.resnet_detail.append(self.model_resnet_3_2)
        self.resnet_detail.append(self.model_resnet_3_3)
        self.resnet_detail.append(self.model_resnet_3_4)

    def forward(self, input1,input2):
        """Standard forward"""
        #######  first
        feature_first_gray = self.model_first_1(input1)
        feature_first_low = self.model_first_2(input2[:,0,:,:].unsqueeze(1))
        feature_first_high = self.model_first_3(input2[:,1:,:,:])
        ####### gray and low
        feature_downsample_gray_1 = self.model_downsample_1_1(feature_first_gray)   
        feature_downsample_1 = torch.cat((feature_downsample_gray_1,feature_first_low),1)
        feature_downsample_1 = self.conv1x1_ngf2(feature_downsample_1)
        feature_downsample_2 = self.model_downsample_1_2(feature_downsample_1)
        feature_downsample_3 = self.model_downsample_1_3(feature_downsample_2)
        feature_downsample_4 = self.model_downsample_1_4(feature_downsample_3)

        feature_resnet_1 = self.model_resnet_1_1(feature_downsample_1)
        feature_resnet_2 = self.model_resnet_1_2(feature_downsample_2)
        feature_resnet_3 = self.model_resnet_1_3(feature_downsample_3)
        feature_resnet_4 = self.model_resnet_1_4(feature_downsample_4)
                
        #######   high
        feature_downsample_high_1 = self.model_downsample_3_1(feature_first_high)
        feature_downsample_high_2 = self.model_downsample_3_2(feature_downsample_high_1)
        feature_downsample_high_3 = self.model_downsample_3_3(feature_downsample_high_2)

        feature_resnet_high_1 = self.model_resnet_3_1(feature_first_high)      
        feature_resnet_high_2 = self.model_resnet_3_2(feature_downsample_high_1)   
        feature_resnet_high_3 = self.model_resnet_3_3(feature_downsample_high_2)
        feature_resnet_high_4 = self.model_resnet_3_4(feature_downsample_high_3)

        ####### upsample
        feature_add_4 = feature_resnet_4 + feature_resnet_high_4 
        feature_upsample_fu1 = self.model_upsample_1(feature_add_4)
        feature_add_3 = feature_resnet_3 + feature_upsample_fu1 + feature_resnet_high_3 
        feature_upsample_0 = self.model_upsample_2(feature_add_3)
        feature_add_2 = feature_resnet_2 + feature_upsample_0 + feature_resnet_high_2 
        feature_upsample_1 = self.model_upsample_3(feature_add_2)
        feature_add_1 = feature_resnet_1 + feature_upsample_1 + feature_resnet_high_1  
        feature_last = self.model_upsample_4(feature_add_1)


        return feature_last
