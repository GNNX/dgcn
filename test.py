import sys
import torch
import numpy as np

from torchvision.transforms import transforms
from train import Cityscapes
from resnet_dgcn import resnet_with_dgcn
from PIL import ImageFile


def confusion_matrix(pred, smnt):

    # IoU = (true positive) / (true_positive + false_positive + false_negative)

    batch,_,H,W = smnt.shape
    smnt = torch.reshape(smnt, (batch, H, W))
    _, indices = torch.max(pred, dim=1)


    confusion_mtrx = np.zeros((19, 2)) # [classes]*[true_pos, false_cases]
    ones = torch.ones((batch, H, W))
    zeros = torch.zeros((batch, H, W))

    unclassified = torch.where(smnt == 0, ones, zeros)
    for i in range(1, 20):
        ground_truth = torch.where(smnt == i, ones, zeros)
        prediction = torch.where(indices == i, ones, zeros)
        prediction = prediction - prediction * unclassified # unclassifed label not considered

        true_pos = torch.sum(ground_truth * prediction).item()
        false_cases = torch.sum(torch.logical_xor(ground_truth, prediction)).item()

        confusion_mtrx[i-1, 0] = true_pos
        confusion_mtrx[i-1, 1] = false_cases

    return confusion_mtrx


def iou_print(total):

    total
    true_pos_only = total[:,0]
    true_pos_plus_false = np.sum(total, axis=1)
    iou = true_pos_only/true_pos_plus_false
    miou = np.sum(iou)/19

    class_dict = {
        0: 'unlabeled',
        1: 'road',
        2: 'sidewalk',
        3: 'building',
        4: 'wall',
        5: 'fence',
        6: 'pole',
        7: 'traffic light',
        8: 'traffic sign',
        9: 'vegetation',
        10: 'terrain',
        11: 'sky',
        12: 'person',
        13: 'rider',
        14: 'car',
        15: 'truck',
        16: 'bus',
        17: 'train',
        18: 'motorcycle',
        19: 'bicycle'
        }

    print("__________________________")
    for i in range(19):

        print("|               |        |")
        print("|{:<13s}  |{:<6.4f}  |".format(class_dict[i+1], iou[i]))
        print("|_______________|________|")

    print("|               |        |")
    print("|MIoU           |{:<6.4f}  |".format(miou))
    print("|_______________|________|")

    return


def test(device, model_path):

    model = torch.load(model_path)
    resnet = resnet_with_dgcn()
    resnet.load_state_dict(model)
    resnet.to(device)
    print('Model Ready')

    cityscapes = Cityscapes()
    validloader = cityscapes.validloader

    total = np.zeros((19, 2))

    with torch.no_grad():
        for i, data in enumerate(validloader, 0):

            img, semantic = data
            img = img.to(device)
            prediction = resnet(img)['out'].to('cpu').detach()
            conf_mtrx = confusion_matrix(prediction, semantic)
            total = total + conf_mtrx

            if (i+1)%50 == 0:
                true_pos_only = total[:,0]
                true_pos_plus_false = np.sum(total, axis=1)
                true_pos_plus_false = np.where(true_pos_plus_false == 0, 1, true_pos_plus_false)
                run_miou = np.sum(true_pos_only/true_pos_plus_false)/19
                print('Iter {} Running MIoU: {:<6.4}'.format(i+1, run_miou))

    print('Prediction Finished')

    iou_print(total)

    return

def main(MODEL_PATH='./model/resnet_dgcn_epoch_180.pth'):

    if not torch.cuda.is_available():

        print('Test is only for CUDA available environment')
        return

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    test(device, MODEL_PATH)

    return


if __name__ == '__main__':

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    sys.exit(main())











