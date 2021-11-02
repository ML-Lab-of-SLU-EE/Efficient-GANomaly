from collections import OrderedDict
import os
import time
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
import torchvision.utils as vutils
from PIL import Image
from lib.networks import NetG, NetD, weights_init
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import evaluate
from options import Options
from lib.data import load_data
from lib.model import Ganomaly
from lib.model import BaseModel
import PIL.Image as pil
class test(Ganomaly):
    def test_G(self, dataloader ):
        """ Test GANomaly model.

        Args:
            dataloader ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "output/ganomaly/Mixball/train/weights/netG.pth"
                pretrained_dict = torch.load(path)['state_dict']
                try:
                    self.netg.load_state_dict(pretrained_dict)
                    for para in list(self.netg.parameters()):
                        para.requires_grad = False
                except IOError:
                    raise IOError("netG weights not found")
                print(' Loaded weights.')
            self.opt.phase = 'test'
            print(self.netg)
            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32,
                                         device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,
                                         device=self.device)
            self.latent_i = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            self.latent_o = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)

            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            self.paths = "data/Mixball/test/normal/2.jpg"
            self.dataloader_on = 1
            if self.dataloader_on:
                for i, data in enumerate(self.dataloader['test'], 0):
                    self.total_steps += self.opt.batchsize
                    epoch_iter += self.opt.batchsize
                    time_i = time.time()
                    self.set_input(data)
                    self.fake, latent_i, latent_o = self.netg(self.input)
                    error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                    # print(error.tolist())
                    self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                        error.size(0))
                    self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                                torch.max(self.an_scores) - torch.min(self.an_scores))
                    time_o = time.time()
                    # print("scores:",self.an_scores)
                    self.times.append(time_o - time_i)
                    print(self.times[i])
                    print(self.total_steps)
                error_final = []
                print("score is :", self.an_scores)
                print(self.an_scores.shape)
                error_final = (error - torch.min(error)) / (
                        torch.max(error) - torch.min(error))
                print("erro is :", error_final.tolist())

            # for idx, image_path in enumerate(self.paths):
            #     if image_path.endswith("_disp.jpg"):
            #         # don't try to predict disparity for a disparity image!
            #         continue
            # self.error = []
            # input_image = pil.open(self.paths).convert('RGB')
            # input_image = input_image.resize((128, 128), pil.LANCZOS)
            # input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            # self.fake, latent_i, latent_o = self.netg(input_image)
            # self.error = self.error.append(torch.mean(torch.pow((latent_i - latent_o), 2), dim=1))
            # print(self.error)
            # print(torch.max(self.error))
            # print(torch.min(self.error))
            # self.an_scores = (self.error - torch.min(self.error)) / (
            #         torch.max(self.error) - torch.min(self.error))
            # self.an_scores = self.an_scores.squeeze_()
            # print(self.an_scores)
            # self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
            #         torch.max(self.an_scores) - torch.min(self.an_scores))
            # print(self.an_scores)
            # print(self.an_scores.shape)




if __name__ == '__main__':
    opt = Options().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    test(opt, dataloader).test_G(dataloader)


