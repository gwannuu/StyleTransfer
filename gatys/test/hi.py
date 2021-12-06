import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import matplotlib.pyplot as plt

import torch.nn as nn

import torchvision.models as models
import torch.nn.functional as F

from torchvision.utils import save_image
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

imsize = 512

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()]) 

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image)
    print(image.shape)
    image = image.unsqueeze(0)
    print(image.shape)
    print(image.dtype)
    return image.to(device, torch.float)


style_img = image_loader("./paintings/style/picasso.jpg")
content_img = image_loader("./paintings/content/dancing.jpg")

assert style_img.size()==content_img.size()

unloader = transforms.ToPILImage()

## plt.ion() plt.ioff()의 반대 기능을 하는 함수. 아마 필요 없을듯?

def imshow(tensor, title=None):
    image=tensor.cpu().clone()
    image=image.squeeze(0)
    image=unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

plt.figure()
imshow(style_img,title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()    #왜 super에 contentloss를 상속받는거지?
        self.target=target.detach()   #detach -> tensor복사 방법중 하나로 gradient전파가 안되는 텐서 생성함

    def forward(self, input):
        self.loss=F.mse_loss(input,self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean=torch.tensor([0.485,0.486,0.406]).to(device)
cnn_normalization_std=torch.tensor([0.229,0.224,0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean=torch.tensor(mean).view(-1,1,1) # 3*1*1이 되겠구마잉
        self.std=torch.tensor(std).view(-1,1,1)

    def forward(self,img):
        return (img-self.mean)/self.std
    
content_layers_default=['conv_4']
style_layers_default=['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_style_model_and_losses(cnn,normalization_mean,normalization_std,style_img,content_img,
                                content_layers=content_layers_default,
                                style_layers=style_layers_default):
                    
    normalization=Normalization(normalization_mean,normalization_std).to(device)

    content_losses=[]
    style_losses=[]

    model=nn.Sequential(normalization)

    i=0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1  #convolution뒤에는 ReLU와 MaxPool과 BatchNorm이 이어지기 때문에 Conv가 나올때만 +1 해주면 됨
            name='conv_{}'.format(i)

        elif isinstance(layer, nn.ReLU):
            name='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)  #inplace가 true이면 input자체를 수정해버린다
        elif isinstance(layer, nn.MaxPool2d):
            name='pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name='bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name,layer)

        if name in content_layers:
            target=model(content_img).detach()
            content_loss=ContentLoss(target)
            model.add_module("content_loss_{}".format(i),content_loss)  # 그냥 원래 pretrain에서 중간중간에 loss단이 들어간 모습이구나
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature=model(style_img).detach()
            style_loss=StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1,-1,-1):  #len(model)-1 부터 0까지 차례대로
    # 뒤에서부터 돌고있는것을 알 수 있다. 뒤에서부터 돌려고 range함수 쓴건가?
        if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
            break
    model=model[:(i+1)]  #VGG Net 에서 style loss와 content loss를 계산하는 layer까지만 필요하고 그뒤의 레이어는 버리겠다는 마인드인듯

    return model, style_losses, content_losses


input_img = content_img.clone()

plt.figure()
imshow(input_img, title='Input Image')

def get_input_optimizer(input_img):
    optimizer=optim.LBFGS([input_img])  #왜 input을 list형태로 넣어주는거지? 그냥넣으면 작동이 안되나?
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img,input_img,num_steps=500,style_weight=100000,content_weight=1):
    print("Building the style transffer model..")
    model,style_losses,content_losses=get_style_model_and_losses(cnn,normalization_mean,normalization_std,style_img,content_img)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer=get_input_optimizer(input_img)

    print('Optimizing..')
    run=[0]
    while run[0]<=num_steps:
        def closure():
            with torch.no_grad():  #no_grad -> context manager that disabled gradient calculation [0,1]로 잘라주는 과정은 gradient추적에 들어가지 않기 때문에 no_grad를 사용해준다
                input_img.clamp_(0,1)  #[0,1] 범위로 자르는 함수 

            optimizer.zero_grad()
            model(input_img)
            style_score=0
            content_score=0

            for sl in style_losses:
                style_score +=sl.loss
            for cl in content_losses:
                content_score+=cl.loss

            style_score *=style_weight
            content_score *=content_weight

            loss=style_score+content_score
            loss.backward()

            run[0] +=1
            if run[0]%50==0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(),content_score.item()))
                print()

            return style_score + content_score
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0,1)
    return input_img

def imsave(tensor, title=None):
    image=tensor.cpu().clone()
    image=image.squeeze(0)
    image=unloader(image)
    image.save("./paintings/results/output.png",'PNG')
 #   plt.imshow(image)
 #   if title is not None:
 #       plt.title(title)
 #   plt.pause(0.001)


output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,content_img, style_img, input_img)
plt.figure()
imshow(output, title='Output Image')
save_image(output, 'output.png')
#imsave(output)
#output.save("./paintings/results/output.png",'PNG')

plt.ioff()
plt.show()



