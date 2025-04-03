import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import wflm_networks
from pytorch_wavelets import DWTForward,DWTInverse
from models.extract_edge import Extractedge
from losses.ssimloss import SSIM

class WflmGANModel(BaseModel):
    """
    This class implements the WFLM-GAN model, for learning image-to-image translation with paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a least-square GANs objective ('--gan_mode lsgan').

    WFLM-GAN paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9912365
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For WFLM-GAN, in addition to GAN losses, we introduce lambda, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: 
        G_gray: real_A -> wave_fake_B -> gray_fake_B; G_color: gray_fake_B and gray -> fake_B.
        Discriminators: 
        D_gray: wave_fake_B vs. wave_real_B  gray_fake_B vs. gray_real_B; D_color: fake_B vs. real_B.
        Losses:
        (1) Supervised loss: 
        First stage: loss_s_gray = ||wave_fake_B - wave_real_B||1 + ||gray_fake_B - gray_real_B||1 (Eqn. (4) in the paper)
        Second stage: loss_s_color = loss_pix + loss_edge + loss_ssim (Eqn. (8) in the paper)
        loss_pix = ||fake_B - real_B||1 + ||fake_B - real_B||2 (Eqn. (5) in the paper)
        loss_edge = ||fake_B(edge) - real_B(edge)||1 (Eqn. (6) in the paper)
        loss_ssim = 1 - SSIM(fake_B, real_B) (Eqn. (7) in the paper)
        (2) Identity loss: idt_gray + idt_color
        loss_idt_gray = ||G_gray(real_B) - wave_real_B||1 + ||G_gray(real_B) - gray_real_B||1 (Eqn. (9) in the paper)
        loss_idt_color = || G_color(G_gray(real_B)) - real_B||1 (Eqn. (10) in the paper)
        Dropout is not used in the WFLM-GAN paper.
        """
        parser.set_defaults(no_dropout=True,norm='instance', dataset_mode='aligned')  # default WFLM-GAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_s', type=float, default=10.0, help='weight for supervised loss (A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_color', 'D_gray', 'G_color', 'G_gray', 'idt_color', 'idt_gray', 's_gray', 's_gray_gray', 's_gray_wave', 'sum_gray', 'pix', 'pix_L1', 'pix_L2', 'edge', 'ssim', 'sum_color', 'add']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  
        #     # visual_names_B.append('idt_A')

            

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_color', 'G_gray', 'D_color', 'D_gray']
        else:  # during test time, only load Gs
            self.model_names = ['G_color', 'G_gray']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_color = wflm_networks.define_ColorG(opt.output_nc, opt.ngf, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_gray = wflm_networks.define_WaveG(opt.ngf, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define discriminators
            self.netD_color = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_gray = wflm_networks.define_WaveD(1, opt.ndf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss(reduce=True, size_average=False)
            self.ssimloss = SSIM(window_size=11)
       
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_color.parameters(),self.netG_gray.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_color.parameters(),self.netD_gray.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # wavelet features: [n, 4, h, w] --> [n, 1, 2*h, 2*w]
    def featurecat(self,feature):
        feature_1 = torch.cat((feature[:,0,:,:],feature[:,2,:,:]),2).unsqueeze(1)   
        feature_2 = torch.cat((feature[:,1,:,:],feature[:,3,:,:]),2).unsqueeze(1)  
        feature_cat = torch.cat((feature_1,feature_2),2)         
        return feature_cat

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # Haar wavelet
        self.dwt = DWTForward(J=1,wave='haar').cuda()  
        self.idwt = DWTInverse(wave='haar').cuda() 
        # Other methods are allowed to choose, but the size of wavelet features needs to be noticed. 
        # self.dwt = DWTForward(J=1,wave='db4').cuda()  
        # self.idwt = DWTInverse(wave='db4').cuda() 

        # real_A: Single-channel SAR images are generally processed into three channels by copying. 
        # Our example: the size of SAR images is 256 * 256, the batchsize is set to 1.
        ################# generate wavelet features 
        self.feature_fake_B_low,self.feature_fake_B_high= self.netG_gray(self.real_A)
        # Low frequency wavelet feature: [1, 1, 128, 128]; High frequency wavelet feature: [1, 1, 3, 128, 128].
        ################# generate gray image
        # ###### If you choose other wavelet bases, the wavelet features should be processed by  interpolation or other methods. 
        # fake_low = F.interpolate(self.feature_fake_B_low,[131,131],mode='bilinear',align_corners=True)
        # data_fake = []
        # data_fake.append(F.interpolate(self.feature_fake_B_high.squeeze(0),[131,131],mode='bilinear',align_corners=True).unsqueeze(0))
        # self.gray_fake_B = self.idwt((fake_low,data_fake))
        # ######
        data_fake = []
        data_fake.append(self.feature_fake_B_high)        
        self.gray_fake_B = self.idwt((self.feature_fake_B_low,data_fake))
        self.gray_fake_B = (self.gray_fake_B/255.0 - 0.5)/0.5    
        ################# concat learned wavelet-features
        self.feature_cA = self.feature_fake_B_low # (1,1,128,128)
        self.feature_cH = self.feature_fake_B_high[:,:,0,:,:] # (1,1,128,128)
        self.feature_cV = self.feature_fake_B_high[:,:,1,:,:] # (1,1,128,128)
        self.feature_cD = self.feature_fake_B_high[:,:,2,:,:] # (1,1,128,128)
        wave_fake_B_1 = torch.cat((self.feature_cA,self.feature_cV),3) # (1,1,128,256)
        wave_fake_B_2 = torch.cat((self.feature_cH,self.feature_cD),3) # (1,1,128,256) 
        wave_fake_B = torch.cat((wave_fake_B_1,wave_fake_B_2),2) ###(1,1,256,256)
        self.wave_fake_B = wave_fake_B.byte()
        self.wave_fake_B = (self.wave_fake_B/255.0 - 0.5)/0.5
        ########################################################################################### Above: first stage

        ################# real_B -> gray_B
        self.gray_real_B = self.real_B[:,0,:,:] * 0.11 + self.real_B[:,1,:,:] * 0.59 + self.real_B[:,2,:,:] * 0.3 
        self.gray_real_B = self.gray_real_B.unsqueeze(1) 
        ################# get real(target) wavelet features     
        self.feature_real_B_low,data_real = self.dwt((self.gray_real_B*0.5+0.5)*255.0)###(1,1,128,128)  (1,1,3,128,128)
        self.feature_real_B_high = data_real[0]
        ################# concat real wavelet features
        self.feature_real_B = torch.cat((self.feature_real_B_low,self.feature_real_B_high[:,0,:,:,:]),1)   
        wave_real_B_1 = torch.cat((self.feature_real_B_low,self.feature_real_B_high[:,:,1,:,:]),3)
        wave_real_B_2 = torch.cat((self.feature_real_B_high[:,:,0,:,:],self.feature_real_B_high[:,:,2,:,:]),3)  
        wave_real_B = torch.cat((wave_real_B_1,wave_real_B_2),2) 
        self.wave_real_B = wave_real_B.byte()
        self.wave_real_B = (self.wave_real_B/255.0 - 0.5)/0.5
        ################# generate color image
        self.feature_fake_B = torch.cat((self.feature_fake_B_low,self.feature_fake_B_high[:,0,:,:,:]),1)
        self.fake_B = self.netG_color(self.gray_fake_B, self.feature_fake_B) ###(1,3,256,256)              
        ################# extract edge
        self.edge_fake_B = Extractedge(channel=3)(self.fake_B)
        self.edge_real_B = Extractedge(channel=3)(self.real_B)  



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 #* 2.0
        loss_D.backward()
        return loss_D

    def backward_D_color(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B
        self.loss_D_color = self.backward_D_basic(self.netD_color, self.real_B, fake_B)

    def backward_D_gray(self):
        """Calculate GAN loss for discriminator D_A"""      

        feature_real_B = self.featurecat(self.feature_real_B)
        feature_fake_B = self.featurecat(self.feature_fake_B)
        pred_real_c1,pred_real_c4 = self.netD_gray(self.gray_real_B,feature_real_B)
        pred_fake_c1,pred_fake_c4 = self.netD_gray(self.gray_fake_B.detach(),feature_fake_B.detach())        
        loss_real = self.criterionGAN(pred_real_c1, True) + self.criterionGAN(pred_real_c4, True)
        loss_fake = self.criterionGAN(pred_fake_c1, False) + self.criterionGAN(pred_fake_c4, False)
        self.loss_D_gray = loss_real * 0.5 + loss_fake * 0.5
        self.loss_D_gray = self.loss_D_gray 
        self.loss_D_gray.backward()
       


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_idt_1 = lambda_idt * 1.0 # for one stage
        lambda_idt_2 = lambda_idt * 0 # for second stage, the default value is set to 0.
        lambda_s = self.opt.lambda_s
        lambda_s_1 = lambda_s * 1.0 # for one stage
        lambda_s_2 = lambda_s * 1.0 # for second stage
        
        # Identity loss
        if lambda_idt > 0:
            ################# generate idt wavelet features
            self.feature_idt_A_low,self.feature_idt_A_high = self.netG_gray(self.real_B)
            #################  
            self.feature_idt_A = torch.cat((self.feature_idt_A_low,self.feature_idt_A_high[:,0,:,:,:]),1)
            self.cA_BB = self.feature_idt_A_low
            self.cH_BB = self.feature_idt_A_high[:,:,0,:,:]
            self.cV_BB = self.feature_idt_A_high[:,:,1,:,:]
            self.cD_BB = self.feature_idt_A_high[:,:,2,:,:]
            idt_wave_1 = torch.cat((self.cA_BB,self.cV_BB),3)
            idt_wave_2 = torch.cat((self.cH_BB,self.cD_BB),3)
            idt_wave = torch.cat((idt_wave_1,idt_wave_2),2) 
            self.wave_idt_A = idt_wave.byte()
            self.wave_idt_A = (self.wave_idt_A/255.0 - 0.5)/0.5
            data_idt = [self.feature_idt_A_high]
            self.gray_idt_A = self.idwt((self.feature_idt_A_low,data_idt))
            self.gray_idt_A = (self.gray_idt_A/255.0 - 0.5)/0.5             
            #################
            loss_idt_gray_wave = self.criterionIdt(self.feature_idt_A, self.feature_real_B)
            loss_idt_gray_wave += self.criterionIdt(self.wave_idt_A, self.wave_real_B) * lambda_s
            loss_idt_gray_gray = 0 #self.criterionIdt(self.gray_idt_A, self.gray_real_B)
            self.loss_idt_gray = (loss_idt_gray_wave + loss_idt_gray_gray)* lambda_idt_1 

            #################
            # self.idt_A = self.netG_color(self.gray_idt_A, self.feature_idt_A)
            # self.loss_idt_color = self.criterionIdt(self.idt_A, self.real_B) * lambda_s
            self.loss_idt_color = 0 
            self.loss_idt_color = self.loss_idt_color * lambda_idt_2
        else:
            self.loss_idt_gray = 0
            self.loss_idt_color = 0
            

        ################# first stage
        feature_fake_B = self.featurecat(self.feature_fake_B)
        # print(feature_fake_B.shape,self.gray_fake_B.shape)
        pred_c1,pred_c4 = self.netD_gray(self.gray_fake_B,feature_fake_B) 
        self.loss_G_gray = self.criterionGAN(pred_c1, True) + self.criterionGAN(pred_c4, True)
        self.loss_G_gray = self.loss_G_gray * 1.0 

        self.loss_s_gray_wave = self.criterionL1(self.feature_fake_B, self.feature_real_B) #(1,4,128,128)
        self.loss_s_gray_wave += self.criterionL1(self.wave_fake_B, self.wave_real_B) * lambda_s_1
        self.loss_s_gray_gray = self.criterionL1(self.gray_fake_B, self.gray_real_B) * lambda_s_1 
        self.loss_s_gray = self.loss_s_gray_wave + self.loss_s_gray_gray
        self.loss_sum_gray = self.loss_G_gray + self.loss_s_gray

        ################# seconde stage       
        
        self.loss_G_color = self.criterionGAN(self.netD_color(self.fake_B), True) 
        self.loss_G_color = self.loss_G_color * 1.0 

        self.loss_pix_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_s_2  
        self.loss_pix_L2 = (self.criterionMSE(self.fake_B, self.real_B)/(3*256*256)) * lambda_s_2   
        self.loss_pix = self.loss_pix_L1 + self.loss_pix_L2
        self.loss_edge = self.criterionL1(self.edge_fake_B,self.edge_real_B)
        self.loss_edge = self.loss_edge * lambda_s_2 
        self.loss_ssim = (1 - self.ssimloss(self.fake_B,self.real_B)) 
        self.loss_ssim = self.loss_ssim * lambda_s_2 
        self.loss_sum_color =  self.loss_G_color + self.loss_pix + self.loss_edge + self.loss_ssim

        # combined loss and calculate gradients
        self.loss_add = self.loss_sum_gray + self.loss_sum_color + self.loss_idt_color + self.loss_idt_gray
        self.loss_add.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_gray and G_color
        self.set_requires_grad([self.netD_color, self.netD_gray], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  
        self.backward_G()             
        self.optimizer_G.step()       
        # D_gray and D_color
        self.set_requires_grad([self.netD_color, self.netD_gray], True)
        self.optimizer_D.zero_grad()   
        self.backward_D_color()      
        self.backward_D_gray()
        self.optimizer_D.step()  
