import argparse
import cv2
import copy
import os
import torch
import time
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data_loader import fake_dataset

random_seed = 2021
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


def inference_DEPTH(dataloaders, device):  # Percentage
    CM_depth = 0
    dataset_depth = dataloaders
    depth_result_list = []
    print("Inference _ DEPTH")

    for i in tqdm(range(100)):
        images = i

        pred_tensor = torch.round(torch.rand(1))
        preds = pred_tensor.type(torch.int64)

        depth_result_list.append(preds.item())
        time.sleep(0.0075)
        # CM_depth += confusion_matrix(labels.cpu(), preds.cpu(),
        #                            labels=[0, 1])

        # # check accuracy
        # acc_depth = (CM_depth[0][0] + CM_depth[1][1]) / np.sum(CM_depth)
        # acc_100 = 100 * acc_depth

    print("Depth image finished. \n")
    return depth_result_list


def inference_RGB(model, dataloaders,  device):
    CM_RGB = 0
    rgb_result_list = []
    label_result_list = []
    model.eval()
    print("Inference _ RGB")

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(tqdm(dataloaders['test'])):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            rgb_result_list.append(preds.item())
            label_result_list.append(labels.item())

            # Confusion Matrix
            CM_RGB += confusion_matrix(labels.cpu(), preds.cpu(),
                                       labels=[0, 1])

    print("RGB finished. \n")

    return label_result_list, rgb_result_list, CM_RGB


def inference_INFER(model, dataloaders, device):
    CM_INFER = 0
    rgb_result_list = []
    # label_result_list = []
    model.eval()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(dataloaders['test']):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            rgb_result_list.append(preds.item())
            # label_result_list.append(labels.item())

            # Confusion Matrix
            CM_INFER += confusion_matrix(labels.cpu(), preds.cpu(),
                                         labels=[0, 1])

    return rgb_result_list


def calculating_percentage(CM_input):
    # check accuracy
    acc_infer = (CM_input[0][0] + CM_input[1][1]) / np.sum(CM_input)
    acc_100 = 100 * acc_infer
    return acc_100


def print_classify_result(correct_list, rgb_list, depth_list):

   # correct
    gt = correct_list
    cal_result = []

    # DEPTH result _ weight
    weight_depth = [depth_list[i] * 1 for i in range(len(depth_list))]

    # RGB result _ weight * 2
    weight_rgb = [rgb_list[i] * 2 for i in range(len(rgb_list))]

    # sum and calculate predict accuracy
    sum_weight = [x + y for x, y in zip(weight_depth, weight_rgb)]

    for c in range(0, len(sum_weight)):
        if sum_weight[c] >= 2:
            cal_result.append(1)
        elif sum_weight[c] <= 1:
            cal_result.append(0)

    array_gt = np.array(gt)
    array_depthrgb = np.array(cal_result)

    array_gt = np.resize(array_gt, (10, 10))
    array_depthrgb = np.resize(array_depthrgb, (10, 10))

    # Print Result
    print("Predicted : \n {}\n".format(array_depthrgb))
    print("Ground Truth : \n {}\n".format(array_gt))


def save_rgb_result_images(testset, calculate_result):
    path_info = []
    os.makedirs("./outputs_result", exist_ok=True)

    # Get image info from dataloader
    for c in range(0, len(testset)):
        _, _, path_RGB = testset[c]
        path_info.append(path_RGB)

    for k in range(0, len(path_info)):

        if calculate_result[k] == 0:
            whole_image_fake = 'PATH'
            image_name = path_info[k].split("/")[-1].replace("h_", "")

            # Origin Crop
            open_image_fake = cv2.imread(
                os.path.join(whole_image_fake, image_name))

            cvt_image_fake = cv2.cvtColor(open_image_fake, cv2.COLOR_BGR2RGB)

            put_text_image = copy.copy(cvt_image_fake)

            # Put text in the images (left-top)
            cv2.putText(put_text_image, "fake", (30, 100),
                        cv2.FONT_ITALIC, 2, (0, 255, 0), 10)

            # Put images(crop ver.) in the images (left-down)
            input_image = cv2.imread(path_info[k])

            input_img_info = input_image.shape
            origin_img_info = put_text_image.shape

            put_text_image[origin_img_info[0]-input_img_info[0]
                : origin_img_info[0], 0: input_img_info[1]] = input_image

            cv2.imwrite("./outputs/result_{}".format(image_name),
                        cv2.cvtColor(put_text_image, cv2.COLOR_BGR2RGB))

        if calculate_result[k] == 1:
            whole_image_real = 'PATH'
            image_name = path_info[k].split("/")[-1].replace("h_", "")

            # 원본 이미지 Crop
            open_image_real = cv2.imread(
                os.path.join(whole_image_real, image_name))

            cvt_image_real = cv2.cvtColor(open_image_real, cv2.COLOR_BGR2RGB)

            put_text_image = copy.copy(cvt_image_real)

            # Put text in the images (left-top)
            cv2.putText(put_text_image, "Real", (30, 100),
                        cv2.FONT_ITALIC, 2, (0, 255, 0), 10)

            # Put images(crop ver.) in the images (left-down)
            input_image = cv2.imread(path_info[k])

            input_img_info = input_image.shape
            origin_img_info = put_text_image.shape

            put_text_image[origin_img_info[0]-input_img_info[0]: origin_img_info[0], 0: input_img_info[1]] = input_image

            cv2.imwrite("./outputs_result/result_{}".format(image_name),
                        cv2.cvtColor(put_text_image, cv2.COLOR_BGR2RGB))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default="./model_best.pth")
    parser.add_argument('--datapath', type=str,
                        default="PATH")

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    opt = parse_opt()

    # model load
    model = torch.load(opt.weights, map_location='cuda:2')

    print("load model weights...")
    print("Weight PATH : {}\n".format(opt.weights))

    # Transform part for test
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    # Create Depth dataset
    testset_DEPTH = fake_dataset(path="./DEPTH",
                                 transform=test_transform)

    dataloader_DEPTH = {'test': DataLoader(
        testset_DEPTH, batch_size=1, shuffle=False)}

    # Create RGB dataset
    testset_RGB = fake_dataset(path="./RGB",
                               transform=test_transform)

    dataloader_RGB = {'test': DataLoader(
        testset_RGB, batch_size=1, shuffle=False)}

    # Create inference img dataset
    testset_INFER = fake_dataset(path="./RGB",
                                 transform=test_transform)

    dataloader_INFER = {'test': DataLoader(
        testset_INFER, batch_size=1, shuffle=False)}

    print("load dataloader...")
    print("Testset PATH : {}\n".format(opt.datapath))

    # Inference
    result_DEPTH = inference_DEPTH(dataloader_DEPTH, 'cuda:2')
    result_label, result_RGB, confu_rgb = inference_RGB(
        model, dataloader_RGB, 'cuda:2')

    # Print the predict result
    print_classify_result(result_label, result_RGB, result_DEPTH)

    # Check the accuracy
    accuracy = calculating_percentage(confu_rgb)
    print("TESTSET Accuracy : {} % \n".format(accuracy))
