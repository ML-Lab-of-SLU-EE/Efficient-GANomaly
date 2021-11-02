import torch.utils.data
from options import Options
from lib.data import load_data
from lib.model import Ganomaly
from lib.model import BaseModel
import PIL.Image as pil
import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from lib.networks import NetG, NetD, weights_init
from model import efficientnet_b0 as create_model
from model import EfficientNet

# TODO：定义模型可以放在train中进行
class efficient_gannet(Ganomaly, EfficientNet):
    def __init__(self, opt, batch_size, dataloader):  # 出错了
        super(efficient_gannet, self).__init__(opt, batch_size, dataloader)
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.latent_i_ball = []  # 定义隐变量i  # TODO:这个隐变量的作用是什么
        self.latent_o_ball = []  # 定义隐变量o
        self.pretrained_dict = []
        self.gan_output = []

        #导入Efficientnet权重
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_ef = create_model(num_classes=5).to(device)
        # load model weights
        try:
            model_weight_path = "weights/model-49.pth"
            self.model_ef.load_state_dict(torch.load(model_weight_path, map_location=device))
        except IOError:
            raise IOError("efficientnet B0 weights not found")
        print(' Loaded weights for efficientnet B0.')

        #导入ganormaly权重
        path_ball = "weights/netG_mixball.pth"
        path_holder = "weights/netG_mixball.pth"
        path_inner = "weights/netG_mixball.pth"
        path_normal = "weights/netG_mixball.pth"
        path_outer = "weights/netG_mixball.pth"
        self.netg_list = [path_ball, path_holder, path_inner, path_normal, path_outer]

        self.classifier = nn.Sequential(
            nn.Linear(16 * (5 + 5), 256),
            nn.ReLU(True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 5),
            nn.Softmax(5)
        )
    def img_trans(self, x: torch.Tensor):
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 缩放
            # transforms.RandomCrop(32, padding=4),  # 裁剪，属于数据扩增，在训练时使用、测试时不用
            transforms.ToTensor(),  # 图片转为张量,同时归一化像素值从[0,255]到[0,1]
        ])
        return train_transform(x.numpy())


    def model_efficient(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model_ef(x.to(self.device))

    def model_ganomaly(self, x:torch.Tensor):
        img = efficient_gannet.img_trans(x)
        with torch.no_grad():
            self.latent_i_ball = torch.zeros(size=(self.batch_size, self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o_ball = torch.zeros(size=(self.batch_size, self.opt.nz), dtype=torch.float32, device=self.device)
            for i in range(5):
                pretrained_dict = torch.load(self.netg_list[i])['state_dict']
                try:
                    self.netg.load_state_dict(pretrained_dict)
                    for para in list(self.netg.parameters()):
                        para.requires_grad = False
                except IOError:
                    raise IOError("netG weights not found")
                self.fake, self.latent_i_ball, self.latent_o_ball = self.netg(img)
                self.error = torch.mean(torch.pow((self.latent_i_ball - self.latent_o_ball), 2), dim=1)
                self.error = (self.error - torch.min(self.error)) / (torch.max(self.error) - torch.min(self.error))
                self.gan_output = torch.cat((self.gan_output, self.error), 0)
        return self.gan_output.transpose(0, 1)   # 将输出的[5,batchsize] 变为[batchsize, 5]


    def forward(self, x:torch.Tensor):
        self.num_pic = x.shape[:, 0]
        with torch.no_grad():
                # predict class
            ef_out = self.model_ef(x.to(self.device))
            gan_out = self.model_ganomaly(x.to(self.device))
        x = torch.cat((ef_out, gan_out), 0)
        x = torch.flatten(x, 0, 1)
        return self.classifier(x)



if __name__ == '__main__':
    opt = Options().parse()
    # efficient_gannet()








