import torch.utils.checkpoint as cp
from torch.nn import Parameter
from torch.hub import load_state_dict_from_url
from fire import Fire
from torch.nn.utils import spectral_norm
from fastai2.vision.all import *

from torch.nn import functional as tf
from Learning.learning import *
from sklearn.preprocessing import normalize

def load_hardnet(model_name, strict=True):
    path = os.path.join('Models', model_name, model_name + '.pt')
    return load_hardnet_abs(path, strict)

def load_hardnet_abs(model_path, is_strict=True): # returns pretrained model from absolute path, use model.eval() before testing
    print('Loading  ...')
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
        model = get_model_by_name(checkpoint['model_arch'])
        model.cuda()
    else:
        print('no gpu')
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = get_model_by_name(checkpoint['model_arch'])
    if model.name == 'TeacherNet':
        student = get_model_by_name(checkpoint['S_model_arch']); model.student = student
        teacher = get_model_by_name(checkpoint['T_model_arch']); model.teacher = teacher
        model.student.load_state_dict(checkpoint['S_state_dict'], is_strict)
        model.teacher.load_state_dict(checkpoint['T_state_dict'], is_strict)
    else:
        try:
            model.load_state_dict(checkpoint['state_dict'], is_strict)
        except KeyError:
            model.load_state_dict(checkpoint['S_state_dict'], is_strict)
    if 'pca' in checkpoint.keys():
        print('LOADING PCA')
        model.pca = checkpoint['pca']
    return model

def obsolete_map(name):
    m = {'h1':'h7', 'h3':'h8'}
    if name in m.keys():
        print('USING OBSOLETE NAME')
        return m[name]
    return name

def get_model_by_name(name):
    nlist = name.replace('_','-').split('-')
    print('arch:', nlist)
    if len(nlist)==1: nlist += [1] # assume grayscale
    nlist[0] = obsolete_map(nlist[0])
    model_name = archs()[nlist[0]]
    model = archs()[nlist[0]](channels=int(nlist[1]))
    model.kornia_transform = KoT.get_transform_kornia()
    model.name = name
    return model

def archs():
    return{
        'SAE': HardNet_8_SAE,
        'encoder' :HardNet_8_SAE_encoder,
        'decoder': HardNet_8_SAE_decoder,
        'TwoSAE':  HardNet_8_2SAE,
        'h7': HardNetx2,
        'h8s64': HardNet_8_256,                                        ###!
        'h8': HardNet_8_256_2,
        'MetricsNet': MetricsNet,
        'TwoNet':TwoNet,
        'dis': HardNet_discriminator,
        'h8x2': HardNet_8_512,
        'SDGMNet': SDGMNet,
        'SDGMNetpca128': SDGMNet_pca128,
        'SDGMNet128':SDGMNet_128,
        'SDGMNetSAE':SDGMNet_SAE,
        'SDGMNet128raw':SDGMNet_128_raw,
        'TeacherNet':TeacherNet,
        ##'HyNet': HyNet,
        # 'h8c': HardNet_8_128,
        # 'h8p48': HardNet_8_256_p48,
        # 'h8p48x': HardNet_8_256_p48x,
        # 'h8p64': HardNet_8_256_p64,
        # 'h8avg': HardNet_8_256_avg,
        # 'h8max': HardNet_8_256_max,
        'h8x2': HardNet_8_512,
        # 'h8x2c': HardNet_8_512C, # less channel on begin
        # # 'h8sa': HardNet_SA,
        # 'h8mb': HardNet_MB,
        #
        # 'h8pca256': HardNet_8_256_pca256,
        'h8pca128': HardNet_8_256_pca128,
        # 'h8pca96': HardNet_8_256_pca96,
        # 'h8pca64': HardNet_8_256_pca64,
        # 'h8pca32': HardNet_8_256_pca32,
        #
        # 'h7pca128': HardNetPCA128,
        # 'h7pca96': HardNetPCA96,
        # 'h7pca64': HardNetPCA64,
        # 'h7pca32': HardNetPCA32,
        #
        # # Exp 2 hardnet8_256, replace global pooling to
        # 'h8Econv3x3s2p1': HardNet_8_256Econv3x3_s2p1,
        # 'h8Econv3x3s2s2c22': HardNet_8_256Econv3x3_s2,
        # 'h8Econv3xnopad': HardNet_8_256Enopad,
        #
        # 'h8blocks': HardNet_8_256_blocks,
        # 'h8blocksc3x3': HardNet_8_256_blocksc3x3,
        #
        # # exp 3
        # 'h8Eavg11': HardNet_8_256_avg11,
        # 'h8Emax11': HardNet_8_256_max11,
        # # exp 4
        # 'h7E256': HardNet256,
        # 'h8Ecto128': HardNet_8_256_cto128,
        #
        # 'h8E512': HardNet_8_256_512,
        # 'h8E512pca512': HardNet_8_256_512pca512,
        # 'h8E512pca256': HardNet_8_256_512pca256,
        # 'h8E512pca128': HardNet_8_256_512pca128,
        # 'h8E512pca96': HardNet_8_256_512pca96,
        # 'h8E512pca64': HardNet_8_256_512pca64,
        # 'h8E512pca32': HardNet_8_256_512pca32,
        #
        # 'h9': HardNet_9_256,
        # 'h9cl': HardNet_9_256_convlater,
        # 'h9x': HardNet_9_256x,
        # 'h9t': HardNet_9_256t,
        # 'h9tt': HardNet_9_256tt,
        #
        # 'HReM': HardNetReshapeMishRes,
        # 'HReMIn': HardNetReshapeMishConvIN,
        # 'h8SBMIn': HardNetSingleBlockMishConvIN,
        #
        # 'HR': HardNetR,
        # 'HR2': HardNetR2,
        # 'hPS': HardNetPS,
        'SOSNet32x32': SOSNet32x32,
        #
        # 'h7sep': HardNet_7_sep,  # channels=2
        # 'h8sep': HardNet_8_sep,  # channels=2
        # 'h8pre': HardNet_8_256_pre,  # channels=3
        # 'h8E512pre': HardNet_8_512_pre,  # channels=3
        # 'h8x2pre': HardNet_8_256l_pre,  # channels=3
        #
        # 'De32': partial(DenseNet,growth_rate=32,block_config=(6, 12, 24, 16),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=64),
        # 'De16': partial(DenseNet,growth_rate=16,block_config=(6, 12, 24, 16),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=64),
        # 'De16s': partial(DenseNet,growth_rate=16,block_config=(6, 8, 16, 12),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=64),
        # 'De16xs': partial(DenseNet,growth_rate=16,block_config=(6, 8, 16, 12),bn_size=4, drop_rate=0.3, Nout_features=128,num_init_features=32),
        #
        # 'Rc3': partial(ResNet_new, layers=[16,32,64,128], resblock=ResBlock_c3),
        # 'Rc3x': partial(ResNet_new, layers=[32, 64, 64, 128], resblock=ResBlock_c3),
        # 'Rc2': partial(ResNet_new, layers=[32, 64, 64, 128], resblock=ResBlock_c2),
        #
        # 'l2net': L2Net,
    }

def try_kornia(x, transform):
    if (transform is not None) and (type(x) == type({})): # this format is only during training
        with torch.no_grad():
            return transform[x['loader']](x['data'])
    return x

def lossnet_from_name(name):
    if name in ['l1']:
        return LossNet()
    if name in ['l2']:
        return LossNet2()
    if name in ['l3']:
        return LossNet3()
    assert False, 'wrong model architecture, change --model_arch'

class BaseHardNet(nn.Module):
    def forward(self, x):
        # x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))   #norm_HW is normalization of patches
        x = x.view(x.size(0), -1)       #tensor  resize
        assert x.shape[1] == self.osize, 'osize!=number of channels, check model arch'
        return F.normalize(x, p=2)       #what is F: operated on the row (div by the norm of the row). normalization of the descriptors

class discriminator(nn.Module):
    def forward(self, x):
        # x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))   #norm_HW is normalization of patches
        x = self.fc_layers(x)
        # x = x.view(x.size(0), -1)
        assert x.shape[1] == 2, 'osize!=number of channels, check model arch'
        return x

class TeacherNet(nn.Module):
    testmode=False
    def __init__(self, channels=1, batchsize_ofTrainingBinary=128):
        super(TeacherNet, self).__init__()
        # self.student = SDGMNet(channels=1)
        # self.teacher = HardNet_8_256_pca128(channels=1)

    def forward(self, x):
        if self.testmode:
            return self.test_mode(x)
        else:
            if self.teacher == 'self':
                self.student.eval()
                with torch.no_grad():
                    T_output = self.student(x)
                self.student.train()
            else:
                self.teacher.eval()
                T_output = self.teacher(x)

            S_output = self.student(x)
            return {'teacher': T_output, 'student': S_output}

    def test_mode(self, x):
        """when use test_mode only output S_output"""
        S_output = self.student(x)
        return S_output

class TwoNet(nn.Module):
    
    def __init__(self, channels=1, batchsize_ofTrainingBinary=128):
        super(TwoNet, self).__init__()
        self.hardnet = SDGMNet(channels = 1)
        self.SAE = SDGMNet_SAE(channels = 1)
        #init values:
        self.input_size = self.hardnet.osize + self.SAE.hiddenout_size
        # self.input_size = self.hardnet.osize
        self.batchsize_ofTrainingBinary = batchsize_ofTrainingBinary * 2

        self.fc_layers = nn.Sequential(
            # nn.Linear(self.input_size, 512),
            # nn.ReLU(),

            # nn.Dropout(0.2),
            # nn.Linear(512, 512),
            # nn.ReLU(),

            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.fc_layers.apply(weights_init)

    def forward(self, x):
        if isinstance(x, type((0,1))):
            return self.forward_centercropped(x[0], x[1])
        x_copy = x
        #SAE net
        SAE_output, x = self.SAE(x)
        #hardnet

        hardnet_output = self.hardnet(x_copy)
        # FC[1]
        # binary_prob = self.fc_layers(torch.cat(hardnet_output, SAE_output, dim=1))  # dim=1?
        desc = torch.cat((hardnet_output, SAE_output), dim=1)
        # mode1_desc, mode2_desc = desc[0::2], desc[1::2]  # 512 length descripter
         #siamese_ft.shape = (batchsize,batchsize,2)
        # siamese_ft = hardnet_output[0:self.batchsize_ofTrainingBinary:2].unsqueeze(1) - SAE_output[1:self.batchsize_ofTrainingBinary:2].unsqueeze(0)  #siamese_ft.shape = (batchsize,batchsize,2)
        # mode1_desc, mode2_desc = self.fc_layers(F.normalize(mode1_desc, p=2)), self.fc_layers(F.normalize(mode2_desc, p=2))
        desc = self.fc_layers(desc)
        return {'hardnet': hardnet_output, 'SAE': (SAE_output, x), 'final_desc': F.normalize(desc, p=2)}

    def forward_centercropped(self, x, x_cropped):
        # x_copy = x
        hardnet_output = self.hardnet(x)
        SAE_output, x = self.SAE(x_cropped)
        # FC[1]
        # binary_prob = self.fc_layers(torch.cat(hardnet_output, SAE_output, dim=1))  # dim=1?
        desc = torch.cat((hardnet_output, SAE_output), dim=1)
        # mode1_desc, mode2_desc = desc[0::2], desc[1::2]  # 512 length descripter
         #siamese_ft.shape = (batchsize,batchsize,2)
        # siamese_ft = hardnet_output[0:self.batchsize_ofTrainingBinary:2].unsqueeze(1) - SAE_output[1:self.batchsize_ofTrainingBinary:2].unsqueeze(0)  #siamese_ft.shape = (batchsize,batchsize,2)
        # mode1_desc, mode2_desc = self.fc_layers(F.normalize(mode1_desc, p=2)), self.fc_layers(F.normalize(mode2_desc, p=2))
        desc = self.fc_layers(desc)
        return {'hardnet': hardnet_output, 'SAE': (SAE_output, x), 'final_desc': F.normalize(desc, p=2)}

    def test_mode(self, x):
        """when use test_mode directily output the binary_prob between x[0] and x[1]"""
        desc_list = []
        for i in range(2):
            #SAE net
            SAE_output, xx = self.SAE(x[i])
            #hardnet
            hardnet_output = self.hardnet(x[i])
            desc = torch.cat((hardnet_output, SAE_output), dim=1)
            desc = self.fc_layers(desc)
            desc_list.append(desc)
        return F.normalize(desc_list[0], p=2) , F.normalize(desc_list[1], p=2)

    def valid_return_desc(self, x):
        """input x is (pic1_patches, pic2_patches)"""
        # x_copy = x
        for i in range(2):
            x[i]
        #SAE net
        SAE_output, x = self.SAE(x)
        #hardnet
        hardnet_output = self.hardnet(x_copy)
        # self.useFC_out_Probmatrix()
        hardnet_output_forFC = torch.cat(
            (torch.zeros((hardnet_output.shape[0], hardnet_output.shape[1])).cuda(), hardnet_output), dim=1)
        SAE_output_forFC = torch.cat((SAE_output, torch.zeros((SAE_output.shape[0], SAE_output.shape[1])).cuda()), dim=1)
        hardnet_output_forFC = hardnet_output_forFC[0::2];
        SAE_output_forFC = SAE_output_forFC[1::2]
        siamese_ft = hardnet_output_forFC.unsqueeze(1) + SAE_output_forFC.unsqueeze(0)
        return siamese_ft

    def useFC_out_Probmatrix(self, siamese_ft):
         #siamese_ft.shape = (batchsize,batchsize,2)
        # siamese_ft = hardnet_output[0:self.batchsize_ofTrainingBinary:2].unsqueeze(1) - SAE_output[1:self.batchsize_ofTrainingBinary:2].unsqueeze(0)
        batchsize = siamese_ft.shape[0]
        siamese_ft = siamese_ft.view(-1, siamese_ft.shape[-1])
        binary_prob = self.fc_layers(siamese_ft[0:10000, :])
        for i in range(1, batchsize ** 2 // 10000 + 1):
            part = self.fc_layers(siamese_ft[10000 * i:10000 * (i + 1), :])  # return shape == (2*batch_size, 2)
            binary_prob = torch.cat([binary_prob, part])
        return {'binary_prob': binary_prob.view(batchsize, batchsize)}


class SAE(nn.Module):
    def forward(self, x):
        # x = try_kornia(x, self.kornia_transform)
        hidden_output = self.encoder(norm_HW(x.float()))   #norm_HW is normalization of patches
        x = self.decoder(hidden_output)
        hidden_output = hidden_output.view(hidden_output.size(0), -1)
       
        assert x.shape[1] == self.osize, 'osize!=number of channels, check model arch'
        assert x.shape[-1] == self.patch_width, 'patch_width!=output shape, check model arch'
        return  F.normalize(hidden_output,p=2),  x

class TwoSAE(nn.Module):
    def forward(self, x):
        hidden_outputs = []; y = [[],[]]
        for i in range(2):
            y[i] = x[i]
            hidden_output = self.encoder[i](norm_HW(y[i].float()))  # norm_HW is normalization of patches
            y[i] = self.decoder[i](hidden_output)
            hidden_output = hidden_output.view(hidden_output.size(0), -1)

            assert y[i].shape[1] == self.osize, 'osize!=number of channels, check model arch'
            assert y[i].shape[-1] == self.patch_width, 'osize!=number of channels, check model arch'
            hidden_outputs.append(F.normalize(hidden_output, p=2))
        return hidden_outputs, y

class Decoders(nn.Module):
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        assert x.shape[1] == self.osize, 'osize!=number of channels, check model arch'
        return x

class BaseHardNetPCA(nn.Module):
    def forward(self, x):
        # x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize, 'osize!=number of channels, check model arch'
        return x.cuda()

class HardNet(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetx2(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNetx2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),        #stride=1
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_7_256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_7_256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNetd01(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetd01, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPCA96(BaseHardNetPCA):
    osize = 96
    def __init__(self, channels=1):
        super(HardNetPCA96, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPCA128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetPCA128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPCA64(BaseHardNetPCA):
    osize = 64
    def __init__(self, channels=1):
        super(HardNetPCA64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPCA32(BaseHardNetPCA):
    osize = 32
    def __init__(self, channels=1):
        super(HardNetPCA32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNet256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_p48(BaseHardNet):
    isize = 48
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_p48, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 24
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 12
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False), # 6
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=6, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNetR2(nn.Module):
    osize = 128
    def __init__(self, channels=1):
        super(HardNetR2, self).__init__()
        self.c1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(32, affine=False)
        # nn.ReLU(),
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(32, affine=False)

        self.c2a = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.b2a = nn.BatchNorm2d(64, affine=False)
        # nn.ReLU(),
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(64, affine=False)
        # nn.ReLU(),
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(64, affine=False)

        self.c4a = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b4a = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.c5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b5 = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.c6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.b6 = nn.BatchNorm2d(128, affine=False)

        self.c6a = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.b6a = nn.BatchNorm2d(128, affine=False)
        # nn.ReLU(),
        self.d = nn.Dropout(0.3)
        self.c7 = nn.Conv2d(128, 128, kernel_size=8, bias=False)
        self.b7 = nn.BatchNorm2d(128, affine=False)

        for m in self.modules():
            weights_init(m)

    def forward(self, input):
        x = norm_HW(input.float())

        a = tf.relu(self.b1(self.c1(x)))
        x = tf.relu(x + self.b2(self.c2(a)))

        x = tf.relu(self.b2a(self.c2a(x)))

        a = tf.relu(self.b3(self.c3(x)))
        x = tf.relu(x + self.b4(self.c4(a)))

        x = tf.relu(self.b4a(self.c4a(x)))

        a = tf.relu(self.b5(self.c5(x)))
        x = tf.relu(x + self.b6(self.c6(a)))

        x = tf.relu(self.b6a(self.c6a(x)))

        x = self.d(x)
        x = tf.relu(self.b6(self.c6(x)))

        x = x.view(x.size(0), -1)
        return F.normalize(x,p=2)

class HardNet_8_sep(nn.Module): # gray + depth
    osize = 128
    def __init__(self, channels=None):
        super(HardNet_8_sep, self).__init__()
        self.features_g = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU())
        self.features_d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU())

        self.features_cat = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False)
        )

        self.features_g.apply(weights_init)
        self.features_d.apply(weights_init)
        self.features_cat.apply(weights_init)

    def forward(self, input):
        I0 = input[:,0,:,:].unsqueeze(1).float()
        I1 = input[:,1,:,:].unsqueeze(1).float()
        del input
        cat_features = self.features_cat(torch.cat([self.features_g(norm_HW(I0)), self.features_d(norm_HW(I1))], 1))
        x = cat_features.view(cat_features.size(0), -1)
        return F.normalize(x,p=2)

class HardNet_8_256(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),  #  Tensor 。
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  #
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),  # similar use as FC layers
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_2(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),  #  Tensor 。
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  #
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),  # similar use as FC layers
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_discriminator(discriminator):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),  #  Tensor 。
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  #
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),  # similar use as FC layers
            nn.BatchNorm2d(256, affine=False),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.osize, 512),         #self.osize=512
            nn.BatchNorm1d(512, affine=False),      #FC can delete this
            nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Dropout(0.23),
            nn.Linear(512, 256)
        )
        self.features.apply(weights_init)
        self.fc_layers.apply(weights_init)

class HardNet_8_SAE_encoder(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(BaseHardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
            #[2]  try to use FC 
            # pool5 = pool5.view(pool5.size(0), -1); print('pool5:', pool5.size())        #flatten
            # self.classifier = nn.Sequential(
            #     torch.nn.Linear(256, 4096),
            #     torch.nn.ReLU(),
            # )
        )
        self.features.apply(weights_init)        #use to intiate weight

class HardNet_8_SAE_decoder(Decoders):
    osize = 1      #input == 256*1*1
    def __init__(self, channels=256):
        super(Decoders, self).__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(channels, 256, kernel_size=8, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1, affine=False),
        )
        self.features.apply(weights_init)        #use to intiate weight

class HardNet_8_SAE(SAE):
    hiddenout_size = 256
    osize = 1
    patch_width = 32
    def __init__(self, channels=1):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),        #stride==2
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
            #[2]  try to use FC
            # pool5 = pool5.view(pool5.size(0), -1); print('pool5:', pool5.size())        #flatten
            # self.classifier = nn.Sequential(
            #     torch.nn.Linear(256, 4096),
            #     torch.nn.ReLU(),
            # )
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=8, bias=False),        #out = 8*8
            # MaxPool()
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),#out = 8*8     stride==2
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1, affine=False),
        )
        self.encoder.apply(weights_init)        #use to intiate weight
        self.decoder.apply(weights_init)  # use to intiate weight

class MetricsNet(SAE):
    osize = 1
    patch_width = 64
    flatten_size = int(patch_width**2)
    input_size = 256
    def __init__(self, channels=1):
        super(SAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 24, kernel_size=7, padding=3),
            nn.BatchNorm2d(24, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(24, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96, affine=False),
            nn.ReLU(), 
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96, affine=False),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),    ## conv4
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),       ## pool4
        )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 256, kernel_size=8, bias=False),        #out = 8*8
        #     # MaxPool()
        #     nn.BatchNorm2d(256, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
        #     nn.BatchNorm2d(128, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
        #     nn.BatchNorm2d(128, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
        #     nn.BatchNorm2d(64, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
        #     nn.BatchNorm2d(32, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32, affine=False),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(1, affine=False),
        # )
        self.classifer = nn.Sequential(
            nn.Linear(self.flatten_size, self.input_size),
            nn.ReLU(),
        )
        # self.deconvclassifer = nn.Sequential(
        #     nn.Linear(256, self.flatten_size),
        #     nn.ReLU(),
        # )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 2),
            # nn.ReLU(),
        )
        self.encoder.apply(weights_init)        #use to intiate weight
        # self.decoder.apply(weights_init)
        self.classifer.apply(weights_init)
        #self.deconvclassifer.apply(weights_init)
        self.fc_layers.apply(weights_init)

    def forward(self, x, test_mode=False):
        hidden_outputs = []
        for i in range(2):
            hidden_output = try_kornia(x[i], self.kornia_transform)
            hidden_output = self.encoder(norm_HW(hidden_output.float()))  # norm_HW is normalization of patches
            hidden_output = hidden_output.view(hidden_output.size(0), -1)
            hidden_output = self.classifer(hidden_output)  # x = self.deconvclassifer(hidden_output)
            hidden_outputs.append(F.normalize(hidden_output, p=2))
            # assert y[i].shape[1] == self.osize, 'osize!=number of channels, check model arch'
            # assert y[i].shape[-1] == self.patch_width, 'osize!=number of channels, check model arch'
        if not test_mode:
            siamese_ft = hidden_outputs[0].unsqueeze(1) - hidden_outputs[1].unsqueeze(0)
        else:
            siamese_ft = hidden_outputs[0] - hidden_outputs[1]
        del hidden_outputs, hidden_output, x

        batchsize = siamese_ft.shape[0]
        siamese_ft = siamese_ft.view(-1, 256)
        part_sum = self.fc_layers(siamese_ft[0:10000, :])
        for i in range(1, batchsize**2 // 10000 + 1):
            part = self.fc_layers(siamese_ft[10000*i:10000*(i+1), :])        #return shape == (2*batch_size, 2)
            part_sum = torch.cat([part_sum, part])
        if not test_mode:
            return part_sum.view(batchsize, batchsize, 2)
        else:
            assert part_sum.shape[0]==batchsize and part_sum.shape[1]==2
            return part_sum


class HardNet_8_2SAE(TwoSAE):
    osize = 1
    patch_width = 32
    def __init__(self, channels=1):
        super(TwoSAE, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=8, bias=False),        #out = 8*8
            # MaxPool()
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1, affine=False),
        )
        self.encoder1.apply(weights_init)        #use to intiate weight
        self.decoder1.apply(weights_init)  # use to intiate weight

        self.encoder2 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=8, bias=False),        #out = 8*8
            # MaxPool()
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, bias=False),#out = 8*8
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False,  output_padding=1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1, affine=False),
        )
        self.encoder2.apply(weights_init)        #use to intiate weight
        self.decoder2.apply(weights_init)  # use to intiate weight
        # make var to deposite 2 SAE
        self.encoder = [self.encoder1, self.encoder2]
        self.decoder = [self.decoder1, self.decoder2]

class HardNet_8_256Econv3x3_s2p1(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256Econv3x3_s2p1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256Econv3x3_s2(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256Econv3x3_s2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256Enopad(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256Enopad, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_pca128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_256_pca128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    # def forward(self, x): # cannot learn !!!
    #     x = try_kornia(x, self.kornia_transform)
    #     x = self.features(norm_HW(x.float()))
    #     x = x.view(x.size(0), -1)
    #     x = F.normalize(x,p=2)
    #     x = normalize(self.pca.transform(x.cpu().detach().numpy())) ############
    #     x = torch.tensor(x)
    #     assert x.shape[1] == self.osize
    #     return x

class HardNet_8_256_pca256(BaseHardNetPCA):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_pca256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_pca96(BaseHardNet):
    osize = 96
    def __init__(self, channels=1):
        super(HardNet_8_256_pca96, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet_8_256_pca64(BaseHardNet):
    osize = 64
    def __init__(self, channels=1):
        super(HardNet_8_256_pca64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet_8_256_pca32(BaseHardNet):
    osize = 32
    def __init__(self, channels=1):
        super(HardNet_8_256_pca32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, x):  # cannot learn !!!
        x = try_kornia(x, self.kornia_transform)
        x = self.features(norm_HW(x.float()))
        x = x.view(x.size(0), -1)
        x = F.normalize(x, p=2)
        x = normalize(self.pca.transform(x.cpu().detach().numpy()))  ############
        x = torch.tensor(x)
        assert x.shape[1] == self.osize
        return x

class HardNet_8_256_512(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_256_512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca256(BaseHardNetPCA):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca256, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca512(BaseHardNetPCA):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca96(BaseHardNetPCA):
    osize = 96
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca96, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca64(BaseHardNetPCA):
    osize = 64
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_512pca32(BaseHardNetPCA):
    osize = 32
    def __init__(self, channels=1):
        super(HardNet_8_256_512pca32, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_cto128(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_256_cto128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_Ins(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_Ins, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class SDGMNet(BaseHardNet):
    osize = 256
    def __init__(self, channels=1, is_bias=True, is_bias_FRN=True, dim_desc=256, drop_rate=0.3):
        super(SDGMNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(256, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        self.features = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
                                      self.layer7)
        self.features.apply(weights_init)

    # def forward(self, x):
    #     feat = self.features(x)
    #     feat_t = feat.view(-1, self.dim_desc)
    #     feat_norm = F.normalize(feat_t, dim=1)
    #     return feat_norm

class SDGMNet_128(BaseHardNet):
    osize = 128
    use_LSUV_init=False
    def __init__(self, channels=1, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.30):
        super(SDGMNet_128, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        self.features = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
                                      self.layer7)
        self.features.apply(weights_init)

    def forward(self, x):
        if self.use_LSUV_init:
            self.LSUV_init(x)
        else:
            feat = self.features(x)        #norm_HW
            feat_t = feat.view(-1, self.dim_desc)
            feat_norm = F.normalize(feat_t, dim=1)
            return feat_norm

    def LSUV_init(self, x):
        self.use_LSUV_init=False
        # x1 = x[0::9]
        torch.backends.cudnn.enabled = False
        model = LSUVinit(self, x, needed_std=1.0, std_tol=0.1, max_attempts=15, needed_mean=0., do_orthonorm=True, cuda=True)
        torch.save({'state_dict':model.state_dict(),
                    'model_arch':'SDGMNet128'}, 'LSUV_init.pt')
        exit(0)

class SDGMNet_pca128(BaseHardNetPCA):
    osize = 128
    def __init__(self, channels=1, is_bias=True, is_bias_FRN=True, dim_desc=256, drop_rate=0.3):
        super(SDGMNet_pca128, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(256, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        self.features = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
                                      self.layer7)
        self.features.apply(weights_init)

    def forward(self, x):
        feat = self.features(x)
        feat_t = feat.view(-1, self.dim_desc)
        feat_norm = F.normalize(feat_t, dim=1)
        feat_norm = normalize(self.pca.transform(feat_norm.cpu().detach().numpy()))  ############
        feat_norm = torch.tensor(feat_norm)
        assert feat_norm.shape[1] == self.osize, 'osize!=number of channels, check model arch'
        return feat_norm.cuda()

class SDGMNet_SAE(SAE):
    hiddenout_size = 256
    osize = 1
    patch_width = 32
    def __init__(self, channels=1, is_bias=True, is_bias_FRN=True, dim_desc=256, drop_rate=0.3):
        super(SDGMNet_SAE, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(256, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

        self.delayer7 = nn.Sequential(
            nn.ConvTranspose2d(256, self.dim_desc, kernel_size=8, bias=False),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256),
        )
        self.delayer6 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, bias=is_bias),
            FRN(256, is_bias=is_bias_FRN),
            TLU(256)
        )
        self.delayer5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.delayer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.delayer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.delayer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.delayer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1, bias=is_bias),
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
        )
        self.encoder = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
                                      self.layer7)
        self.decoder = nn.Sequential(self.delayer7, self.delayer6, self.delayer5, self.delayer4, self.delayer3, self.delayer2,
                                      self.delayer1)
        self.encoder.apply(weights_init)        #use to intiate weight
        self.decoder.apply(weights_init)  # use to intiate weight

class SDGMNet_128_raw(BaseHardNet):
    osize = 128
    testmode = False
    def __init__(self, channels=1, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.32):
        super(SDGMNet_128_raw, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )
        self.features = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6,
                                      self.layer7)
        self.features.apply(weights_init)

    def forward(self, x):
        feat = self.features(x)        #norm_HW
        feat_t = feat.view(-1, self.dim_desc)
        feat_norm = F.normalize(feat_t, dim=1)
        if self.testmode:
            return feat_norm
        else:
            return {'desc': feat_norm, 'raw_desc': feat_t}

class HardNet_8_256_blocks(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_blocks, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 64, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.Flatten(),
            nn.BatchNorm1d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_blocksc3x3(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_blocksc3x3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.Flatten(),
            nn.BatchNorm1d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_128(BaseHardNet):
    osize = 128
    def __init__(self, channels=1):
        super(HardNet_8_128, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_p48(BaseHardNet):
    isize = 48
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_p48, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 24
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 12
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 6
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=6, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_p48x(BaseHardNet):
    isize = 48
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_p48x, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 24
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 12
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=12, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_p64(BaseHardNet):
    isize = 64
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_p64, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 32
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 16
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 8
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=8, bias=False),
            nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_avg(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_avg, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Conv2d(256, 256, kernel_size=8, bias=False),
            # nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_avg11(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_avg11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.features.apply(weights_init)

class HardNet_8_256_max(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_max, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0),
            # nn.Conv2d(256, 256, kernel_size=8, bias=False),
            # nn.BatchNorm2d(256, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_256_max11(BaseHardNet):
    osize = 256
    def __init__(self, channels=1):
        super(HardNet_8_256_max11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0),
        )
        self.features.apply(weights_init)

def _conv1d_spect(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

class SimpleSelfAttention(Module):
    def __init__(self, n_in:int, ks=1):
        self.bn = nn.BatchNorm1d(n_in)
        self.bn.weight.data.fill_(0.1)
        self.bn.bias.data.fill_(0)
        self.conv = _conv1d_spect(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self,x):
        size = x.size()
        x = x.view(*size[:2],-1)
        xbn = self.bn(x)
        convx = self.conv(xbn)
        xxT = torch.bmm(xbn,xbn.permute(0,2,1).contiguous()).clamp_(-10,10)
        o = torch.bmm(xxT, convx.view(*size[:2],-1))
        o = F.tanh(self.gamma) * o + x
        return o.view(*size).contiguous()

class HardNet_8_512(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_512, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNet_8_512C(BaseHardNet):
    osize = 512
    def __init__(self, channels=1):
        super(HardNet_8_512C, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, affine=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 512, kernel_size=8, bias=False),
            nn.BatchNorm2d(512, affine=False),
        )
        self.features.apply(weights_init)

class HardNetPS(nn.Module):
    osize = 128
    def __init__(self, channels=None):
        super(HardNetPS, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8, bias=True),
        )

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return F.normalize(x, p=2)


class SOSNet32x32(nn.Module):
    osize = 128
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """
    def __init__(self, dim_desc=128, drop_rate=0.1, channels=None):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        eps_fea_norm = 1e-5

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch, eps_l2_norm = 1e-10):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr

class L2Net(nn.Module):
    """L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space"""
    def __init__(self, out_dim=128, binary=False, dropout_rate=0.1, channels=None):
        super().__init__()
        self._binary = binary
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, out_dim, kernel_size=8, bias=False),
            nn.BatchNorm2d(out_dim, affine=False),
        )

        if self._binary:
            self.binarizer = nn.Tanh()
        self.features.apply(L2Net.weights_init)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        input = self.input_norm(input)
        x = self.features(input)
        x = x.view(x.size(0), -1)
        if self._binary:
            return self.binarizer(x)
        else:
            return F.normalize(x, p=2, dim=1)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(*inputs)
        return cp.checkpoint(closure, input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    _version = 2
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_"""
    def __init__(self, channels=1, name='unset', growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, drop_rate=0.3, Nout_features=128, memory_efficient=True, num_init_features=64, load=True):
        super(DenseNet, self).__init__()
        self.name=name
        self.osize = Nout_features
        self.features = nn.Sequential(OrderedDict([ # First convolution
            ('conv01', nn.Conv2d(channels, num_init_features//4, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu01', nn.ReLU()),
            ('norm01', nn.BatchNorm2d(num_init_features//4)),
            ('conv02', nn.Conv2d(num_init_features//4, num_init_features//2, kernel_size=3, stride=1, padding=1, bias=False)),
            ('relu02', nn.ReLU()),
            ('norm02', nn.BatchNorm2d(num_init_features//2)),
            ('conv03', nn.Conv2d(num_init_features//2, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu04', nn.ReLU()),
            ('norm04', nn.BatchNorm2d(num_init_features)),

            # ('conv02', nn.Conv2d(channels, num_init_features // 2, kernel_size=3, stride=1,padding=1, bias=False)),
            # ('relu02', nn.ReLU(inplace=True)),
            # ('norm02', nn.BatchNorm2d(num_init_features // 2)),
            # ('conv03', nn.Conv2d(num_init_features // 2, num_init_features, kernel_size=3, stride=2,padding=1, bias=False)),
            # ('relu04', nn.ReLU(inplace=True)),
            # ('norm04', nn.BatchNorm2d(num_init_features)),

            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
            #                     padding=3, bias=False)),
            # ('norm0', nn.BatchNorm2d(num_init_features)),
            # ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features //= 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features)) # Final batch norm
        self.lastconv = nn.Conv2d(num_features, Nout_features, kernel_size=2, stride=1, bias=False)
        self.lastbnorm = nn.BatchNorm2d(Nout_features) # Final batch norm

        for m in self.modules(): # Official init from torch repo.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        if load:
            self._load_state_dict(model_urls['densenet121'], progress=True)

    def forward(self, x):
        features = self.features(norm_HW(x.float()))
        x = F.relu(features, inplace=True)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # out = torch.flatten(out, 1)
        x = self.lastconv(x)
        x = self.lastbnorm(x)
        x = torch.flatten(x, 1)
        # return out
        # x = out.view(out.size(0), -1)
        return F.normalize(x, p=2)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        loaded, failed = 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter): # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
                loaded += 1
            except:
                failed += 1
        print('loaded',loaded,'failed',failed)

    def _load_state_dict(self, model_url, progress):
        printc.yellow('(pre)LOADING DNet WEIGHTS - whatever I can')
        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used to find such keys.
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        state_dict = load_state_dict_from_url(model_url, progress=progress)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.load_my_state_dict(state_dict)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1): #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1): #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    # __constants__ = ['downsample']
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class SpaceToDepth2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // 2, 2, W // 2, 2)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 4, H // 2, W // 2)  # (N, C*bs^2, H//bs, W//bs)
        return x


def norm_HW(x): # channel-wise mean, format is BCHW, dims 2,3 are H,W, broadcasting applies
    mp = torch.mean(x, dim=(2,3), keepdim=True)
    sp = torch.std(x, dim=(2,3), keepdim=True) + 1e-7
    return (x - mp) / sp

def weights_init(m):        #m is layer
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except: pass

class Interface:
    def params(self, model=None):
        if model is not None:
            if type(model)==str:
                model = get_model_by_name(model)
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        res = []
        for a in archs().keys():
            aux = self.params(a)
            res += [(a,aux)]
            print(aux)
        print(' ≤ '.join([c[0] for c in sorted(res,key=lambda x:x[1])]))

if __name__ == "__main__":
    Fire(Interface)
