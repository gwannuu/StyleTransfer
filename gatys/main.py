from data.data_loader import load_image
from model.layer import load_model

import os

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image

import argparse

# 인자를 받는 인스턴스 생성
parser=argparse.ArgumentParser(description='Stylization input image to style image maintaining content image')

# 입력받을 인자값 등록
parser.add_argument('--styleweight',type=int,required=False,default=100000,help='Set weight of style loss')
parser.add_argument('--contentweight',type=int,required=False,default=1,help='Set weight of content loss')
parser.add_argument('--cudanum',required=True,help='set cuda number ex)"3,5"')
parser.add_argument('--itersize',type=int,required=False,default=400,help='Set the number of iteration')
parser.add_argument('--title',required=False,default='output.png',help='Set the title of result')

# 입력받은 인자값을 args에 저장
args=parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #GPU or CPU

if device=='cuda':       
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cudanum #if you use cuda, enter your cuda number


contentImg=load_image("./data/content/effeltower.jpg",(512,512),device)
styleImg=load_image("./data/style/TheWeepingWoman.jpg",(512,512),device)

print(contentImg.shape)
print(styleImg.shape)

content_loss_list=['conv_4']  #default content loss layer. you can change this
style_loss_list=['conv_1','conv_2','conv_3','conv_4','conv_5']  #default content loss layer. you can change this

inputImg=contentImg.clone().requires_grad_(True)

model,style_losses,content_losses=load_model(content_loss_list,style_loss_list,inputImg,contentImg,styleImg,device)

model.requires_grad_(False)  #그런데 requires grad가 false인데 input image까지 gradient가 전달이 어떻게 되는거지? 그리고 어짜피 optimizer에 input_img만 parameter로써 넣어주었으니까 업데이트 안되는거 아닌가? model은?

optimizer=optim.LBFGS([inputImg])

iter=[0]
while iter[0]<=args.itersize:
    def closure():
        with torch.no_grad():
            inputImg.clamp_(0,1)
        
        optimizer.zero_grad()
        model(inputImg)

        style_score=0
        content_score=0

        for sl in style_losses:
            style_score+=sl.loss
        for cl in content_losses:
            content_score+=cl.loss

        style_score *=args.styleweight
        content_score *=args.contentweight

        loss=style_score+content_score
        loss.backward()

        iter[0]+=1

        if iter[0]%10==0:
            print("iter ",iter)
            print("Style Loss : {} Content Loss : {}".format(style_score.item(),content_score.item()))
            print()

        return style_score+content_score
    optimizer.step(closure)

with torch.no_grad():
    inputImg.clamp_(0,1)

save_image(inputImg,os.path.join(os.getcwd(),args.title))



