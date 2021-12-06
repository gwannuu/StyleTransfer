import torch 
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.models as models
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        self.mean=torch.tensor([0.485,0.486,0.406]).view(-1,1,1).to(device)
        self.std=torch.tensor([0.229,0.224,0.225]).view(-1,1,1).to(device)
    
    def forward(self,img):
        return (img-self.mean)/self.std

class ContentLoss(nn.Module):
    def __init__(self,target):
        super().__init__()
        self.target=target.detach()  #detach 유무 차이?

    def forward(self, input):
        self.loss=F.mse_loss(self.target,input)
        return input

class StyleLoss(nn.Module):
    def __init__(self,feature):
        super().__init__()
        b,c,h,w=feature.size()
        feature_=feature.view(b*c,h*w)
        gram=torch.mm(feature_,feature_.t())
        self.target=gram.div(c*h*w).detach()  #detach 유무 차이?

    def forward(self,input):
        b,c,h,w=input.size()
        input_=input.view(b*c,h*w)
        gram=torch.mm(input_,input_.t())
        gram.div_(c*h*w).detach()
        self.loss=F.mse_loss(gram,self.target)
        return input


def load_model(content_loss_list,style_loss_list,inputImg,contentImg,styleImg,device='cuda'):

    normalization=Normalization().to(device)
    model=nn.Sequential(normalization)

    content_losses=[]
    style_losses=[]

    vgg19=models.vgg19(pretrained=True).features.to(device).eval()  #BatchNorm layer의 경우 training time과 test time에 행동이 달라진다.

    i=0
    for layer in vgg19.children():  #그냥 vgg19와 vgg19.children의 차이?
        if isinstance(layer,nn.Conv2d):
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name='pool_{}'.format(i)
        else:
            raise RuntimeError('No existing layer')

        model.add_module(name,layer.to(device))

        if name in content_loss_list:
            target=model(contentImg).detach()  #근데 여기에서 detach가 굳이 들어갈 필요가 있나? storage가 공유되고 전파가 안되는 텐서.  clone은 들어가면 안되나?
            content_loss=ContentLoss(target)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)
        if name in style_loss_list:
            target=model(styleImg).detach()
            style_loss=StyleLoss(target).to(device)
            model.add_module('style_loss_{}'.format(i),style_loss)
            style_losses.append(style_loss)

    idx=len(model)-1
    while not isinstance(model[idx],ContentLoss) and not isinstance(model[idx],StyleLoss):
        idx-=1
    model=model[:(idx+1)]

    return model,style_losses,content_losses


