import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)  # 得到的class_indict是一个字典

    # create model
    model = create_model(num_classes=5).to(device)  # 换数据集需要修改num_classes
    # load model weights
    model_weight_path = "D:\DeepLearningTEMP\Test9_efficientNet 2\weights\model-29.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # new_load_image
    root_path = './final_test_data'
    test_images = os.listdir(root_path)
    for test_image in test_images:
        img_path = os.path.join(root_path, test_image)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img_org = Image.open(img_path)
        img = data_transform(img_org)
        img = torch.unsqueeze(img, dim=0)  # 读取图片并扩展维度

        with torch.no_grad():  # 进行前向推理预测类别
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()  # 得到预测类别的编号
        class_name = class_indict[str(predict_cla)]
        save_root_path = './final_test_data_result'
        image_save_path = os.path.join(save_root_path, class_name, test_image)
        img_org.save(image_save_path)
        print("class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy()))

if __name__ == '__main__':
    main()
