import torch
import math
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet
from utils import read_split_data
import torch.nn as nn
from lib.networks import Encoder, Decoder

# load GANomly weigts
def load_single_ganomly(netg, path, d='cuda'):
    device = torch.device(d)
    netg = netg.to(device)
    pretrained_dict = torch.load(path)['state_dict']
    try:
        netg.load_state_dict(pretrained_dict)
        for para in list(netg.parameters()):
            para.requires_grad = False
    except IOError:
        raise IOError('netG weights not found')


# compute the GANomly's final error
def compute_error(latent_i, latent_o):
    error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
    error_final = (error - torch.min(error)) / (torch.max(error) - torch.min(error))
    shape = error_final.shape
    error_final = error_final.reshape(shape[0], -1)
    return error_final


class Classifier(nn.Module):
    def __init__(self, num_class=5):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(in_features=10, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=64)
        self.linear4 = nn.Linear(in_features=64, out_features=num_class)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

# define GANomly's NetG
class GANomly_NetG(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, extralayers):
        super(GANomly_NetG, self).__init__()
        self.encoder1 = Encoder(isize, nz, nc, ngf, ngpu, extralayers)
        self.decoder = Decoder(isize, nz, nc, ngf, ngpu, extralayers)
        self.encoder2 = Encoder(isize, nz, nc, ngf, ngpu, extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o

def main(data_path, batch_size, epochs):
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    # 加载数据
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(128),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(128),
                                   transforms.CenterCrop(128),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # define tensorboard writer
    tb_writer = SummaryWriter()

    # create model
    final_classifier = Classifier().to(torch.device('cuda'))

    efficient_model = create_model(num_classes=5).to(torch.device('cuda'))

    ganomly_ball = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_inner = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_outer = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_holder = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    ganomly_normal = GANomly_NetG(isize=128, nz=100, nc=3, ngf=64, ngpu=1, extralayers=0)
    print('Successfully create models!')

    # load weights
    efficient_model.load_state_dict(torch.load('./weights/EfficientNet/model-49.pth', map_location=torch.device('cuda')))
    print('Successfully load EfficientNet weights!')

    load_single_ganomly(netg=ganomly_ball, path='./weights/GANomly/ball/netG.pth')
    load_single_ganomly(netg=ganomly_inner, path='./weights/GANomly/inner/netG.pth')
    load_single_ganomly(netg=ganomly_outer, path='./weights/GANomly/outer/netG.pth')
    load_single_ganomly(netg=ganomly_holder, path='./weights/GANomly/holder/netG.pth')
    load_single_ganomly(netg=ganomly_normal, path='./weights/GANomly/normal/netG.pth')
    print('Successfully load GANomaly weights!')

    models = [efficient_model, ganomly_ball, ganomly_inner, ganomly_outer, ganomly_holder, ganomly_normal]

    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(torch.device('cuda'))
    # only set optimizer for final_classifier
    pg = [p for p in final_classifier.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=1e-4, momentum=0.9, weight_decay=1e-4)
    # set learning rate schedule
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(epochs):
        # set all fixed model to eval model, final_classifier to train mode
        for model in models:
            model.eval()
        final_classifier.train()

        for step, data in enumerate(train_loader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()  # images.shape=[16, 3, 128, 128]; labels.shape=[16]; labels=[0, 3, 2, ..., 3]
            with torch.no_grad():
                # GANomly interference process
                fake1, latent_i_1, latent_o_1 = ganomly_ball(images)
                error_1 = compute_error(latent_i_1, latent_o_1)

                fake2, latent_i_2, latent_o_2 = ganomly_inner(images)
                error_2 = compute_error(latent_i_2, latent_o_2)

                fake3, latent_i_3, latent_o_3 = ganomly_outer(images)
                error_3 = compute_error(latent_i_3, latent_o_3)

                fake4, latent_i_4, latent_o_4 = ganomly_holder(images)
                error_4 = compute_error(latent_i_4, latent_o_4)

                fake5, latent_i_5, latent_o_5 = ganomly_normal(images)
                error_5 = compute_error(latent_i_5, latent_o_5)  # error.shape=[16, 1]

                total_error = torch.cat([error_1, error_2, error_3, error_4, error_5], dim=-1)  # total_error.shape=[16, 5]
                # print('ganomly', total_error)

                # EfficientNet interference process
                score = efficient_model(images)  # score.shape=[16, 5]
                final_input = torch.cat([total_error, score], dim=-1)  # final_input.shape=[16, 10]


            out = final_classifier(final_input)  # out.shape=[16, 5]

            loss = loss_function(out, labels)
            loss.backward()
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
            print('epoch {} mean loss {} '.format(epoch, round(mean_loss.item(), 3)))
            # train_loader.desc = '[epoch {}] mean loss {}'.format(epoch, round(mean_loss.item(), 3))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            tags = ['loss', 'learning_rate']
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], optimizer.param_groups[0]['lr'], epoch)

        torch.save(final_classifier.state_dict(), './weights/final_classifier/final_classifier_model-{}.pth'.format(epoch))

if __name__ == '__main__':
    data_path = './data/mix12/'
    batch_size = 16
    epochs = 50
    main(data_path, batch_size, epochs)

















