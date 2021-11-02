import torch
import os
import json
import math
import tqdm
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet, my_test_dataset
from utils import read_split_data, evaluate
import torch.nn as nn
from lib.networks import Encoder, Decoder
from train_final_1 import Classifier, GANomly_NetG, load_single_ganomly, \
    compute_error  # place the model into networks.py would be better

# def embedded_numbers(path_list):
#     pieces = re.compile(r'\d+').split(s)

def main(test_dir):
    # data transform
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(128),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(128),
                                   transforms.CenterCrop(128),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # read json file
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, 'r')
    class_indict = json.load(json_file)

    # define test dataset & dataloader
    test_images_path = sorted(os.listdir(test_dir))
    new_test_images_path = []
    for test_image_path in test_images_path:
        test_image_path = os.path.join(test_dir, test_image_path)
        new_test_images_path.append(test_image_path)

    test_dataset = my_test_dataset(images_path=new_test_images_path,
                            transform=data_transform["val"])
    # print('test_dataset', test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             pin_memory=True)
    # print('test_loader', test_loader)


    # create models
    efficientNet = create_model(num_classes=5).to(torch.device('cpu'))

    ganomly_ball = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_inner = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_outer = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_holder = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_normal = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)

    final_classifier = Classifier().to(torch.device('cpu'))

    # load weights
    efficientNet.load_state_dict(torch.load('./weights/EfficientNet/model-49.pth', map_location=torch.device('cpu')))

    load_single_ganomly(netg=ganomly_ball, path='./weights/GANomly/ball/netG.pth', d='cpu')
    load_single_ganomly(netg=ganomly_inner, path='./weights/GANomly/inner/netG.pth', d='cpu')
    load_single_ganomly(netg=ganomly_outer, path='./weights/GANomly/outer/netG.pth', d='cpu')
    load_single_ganomly(netg=ganomly_holder, path='./weights/GANomly/holder/netG.pth', d='cpu')
    load_single_ganomly(netg=ganomly_normal, path='./weights/GANomly/normal/netG.pth', d='cpu')

    final_classifier.load_state_dict(
        torch.load('./weights/final_classifier/final_classifier_model-49.pth', map_location=torch.device('cpu')))

    models = [efficientNet, ganomly_ball, ganomly_inner, ganomly_outer, ganomly_holder, ganomly_normal,
              final_classifier]
    for model in models:
        model.eval()

    print('Loading dataloader images...')
    with torch.no_grad():
        # loop test_dataloader
        for step, (img, img_path) in enumerate(test_loader):  # data.shape=[16, 3, 128, 128]

            # EfficientNet interference process
            efficientNet_score = efficientNet(img)  # score.shape=[1, 5]
            # print('efficient score:', torch.squeeze(efficientNet_score).numpy())

            # GANomly interference process
            fake1, latent_i_1, latent_o_1 = ganomly_ball(img)
            error_1 = compute_error(latent_i_1, latent_o_1)
            fake2, latent_i_2, latent_o_2 = ganomly_inner(img)
            error_2 = compute_error(latent_i_2, latent_o_2)
            fake3, latent_i_3, latent_o_3 = ganomly_outer(img)
            error_3 = compute_error(latent_i_3, latent_o_3)
            fake4, latent_i_4, latent_o_4 = ganomly_holder(img)
            error_4 = compute_error(latent_i_4, latent_o_4)
            fake5, latent_i_5, latent_o_5 = ganomly_normal(img)
            error_5 = compute_error(latent_i_5, latent_o_5)  # error.shape=[16, 1]
            total_error = torch.cat([error_1, error_2, error_3, error_4, error_5], dim=-1)  # total_error.shape=[16, 5]
            final_input = torch.cat([total_error, efficientNet_score], dim=-1)  # final_input.shape=[16, 10]
            # print('ganomly score:', torch.squeeze(total_error).numpy())

            # final_classifier interference process
            final_output = final_classifier(final_input)  # final_output.shape=[16, 5]
            # print('final score:', torch.squeeze(final_output).numpy())
            # print('final_score.shape:', final_output.shape)
            predicts = torch.softmax(final_output, dim=0)  # predict.shape=[16, 5]
            predict_clas = torch.argmax(predicts, dim=-1).numpy()

            batch_results = []

            print('batch%d results' % (step+1))
            for predict_cla, predict, image_path in zip(predict_clas, predicts, img_path):
                class_name = class_indict[str(predict_cla)]
                confidence = float(predict[predict_cla])
                batch_results.append([image_path + '\n', {'class': str(class_name)}, {'confidence': round(confidence, 3)}])

            for batch_result in batch_results:
                print(batch_result)

if __name__ == '__main__':
    test_image_path = './data/final_test_data'
    main(test_image_path)