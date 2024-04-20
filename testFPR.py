#import Losses
import PhotoTour
from models import SDGMNet_128
import torch
import EvalMetrics
import os
import torchvision.transforms as transforms
import PIL
import numpy as np
# from Utils import transform_test
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def np_reshape64(x):
    x_ = np.reshape(x, (64, 64, 1))
    return x_

transform_test = transforms.Compose([
    transforms.Lambda(np_reshape64),
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor()])

def CreateTestLoader(name='yosemite', norm=False):
    print(name)
    if norm == True:
        root = 'Datasets/6Brown/normData'
    else:
        root =  'Datasets/6Brown/Data'
    TestData = PhotoTour.PhotoTour(root = root, name = name, download = False, train = False, transform = transform_test)
    Testloader = torch.utils.data.DataLoader(TestData, batch_size = 1024,
                                                shuffle=False, num_workers=10, pin_memory = True)
    return Testloader

def test(model, testloader, GPU=True):
    model.eval()
    with torch.no_grad():
        if GPU:
            simi = torch.zeros(0).cuda()
        else:
            simi = torch.zeros(0)
        lbl = torch.zeros(0, dtype=torch.int64)
        labels, distances = [], []
        model.testmode = True
        for i, (data1, data2, m) in enumerate(testloader):
            if GPU:
                data1 = data1.cuda(non_blocking=True)
                data2 = data2.cuda(non_blocking=True)
            t1 = model(data1)
            t2 = model(data2)
            t3 = torch.sum(t1*t2, dim=1).detach().view(-1)
            simi = torch.cat((simi, t3), dim=0)
            lbl = torch.cat((lbl, m.view(-1)), dim=0)
        lbl = lbl.numpy()
        simi = simi.cpu().numpy()
        FPR = EvalMetrics.ErrorRateAt95Recall(labels=lbl, scores=simi+10)
    return FPR

modelid = 'Self-TNet'                    #
pretrained_model = 'checkpoint_5.pt'                     #SELFTN_EP25_NEW2_UP0_XND_22.pt
model = SDGMNet_128()                                   #id:self-2stB-thr15.pt
checkpoint = torch.load(pretrained_model)
model.load_state_dict(checkpoint['S_state_dict'])
# model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
useNorm = False                          #
# TestLoader1 = CreateTestLoader('yosemite')
TestLoader1 = CreateTestLoader('yosemite', norm=useNorm) #
TestLoader2 = CreateTestLoader('liberty', norm=useNorm)
# TestLoader2 = CreateTestLoader('notredame',norm=useNorm) #

FPR1 = test(model,TestLoader1)
FPR2 = test(model,TestLoader2)
print(FPR1)
print(FPR2)
print((FPR1+FPR2)/2)