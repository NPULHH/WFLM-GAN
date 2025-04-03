import torch
import torch.nn as nn 
class Extractedge(nn.Module):
    def __init__(self,channel=3):
        super(Extractedge,self).__init__()
        # self.kernel = self.gauss_kernal()
    # def gauss_kernel(self, device=torch.device('cuda'), channels=3):
    #     kernel = torch.tensor([[1., 4., 6., 4., 1],
    #                            [4., 16., 24., 16., 4.],
    #                            [6., 24., 36., 24., 6.],
    #                            [4., 16., 24., 16., 4.],
    #                            [1., 4., 6., 4., 1.]])
    #     kernel /= 256.
    #     kernel = kernel.repeat(channels, 1, 1, 1)
    #     kernel = kernel.to(device)
    #     return kernel

        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channel, 1, 1, 1)
        kernel = kernel.to(torch.device('cuda'))
        self.kernel = kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def forward(self,input) :
        x_downsample = self.downsample(input)
        x_upsample = self.upsample(x_downsample)
        edge = input - x_upsample
        return edge
