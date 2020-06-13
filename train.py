import sys
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from resnet_dgcn import resnet_with_dgcn
from PIL import ImageFile


def adaptive_lr(optimizer, epoch, total_epoch, init=0.01):

    
    lr = init * math.pow(1.0 - epoch/total_epoch, 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


class Cityscapes:

    def __init__(self):

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            )
        self.trainset = torchvision.datasets.Cityscapes(
            root='./cityscapes',
            split='train',
            mode='fine',
            target_type='semantic',
            transforms=self.cropNflip
            )
        self.trainloader = torch.utils.data.DataLoader(self.trainset,batch_size=1,shuffle=True, num_workers=4)

        self.validset = torchvision.datasets.Cityscapes(
            root='./cityscapes',
            split='val',
            mode='fine',
            target_type='semantic',
            transform=img_transform,
            target_transform=self.label_heat_map
            )
        self.validloader = torch.utils.data.DataLoader(self.validset,batch_size=1,shuffle=False, num_workers=4)

        return

    def cropNflip(self, img, smnt):

        crop_size=768
        assert img.size[0] == smnt.size[0]
        assert img.size[1] == smnt.size[1]

        if random.random() > 0.5:
            img = TF.hflip(img)
            smnt = TF.hflip(smnt)

        W, H = img.size
        x = random.randint(0, W-crop_size-1)
        y = random.randint(0, H-crop_size-1)
        img = TF.crop(img, y, x, crop_size, crop_size)
        smnt = TF.crop(smnt, y, x, crop_size, crop_size)

        img = TF.to_tensor(img)
        img = TF.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        smnt = torch.round(TF.to_tensor(smnt)*256)
        smnt = self.label(smnt)

        return img, smnt

    def label(self, smnt):

        '''
        class_name     class_num
        road           1
        sidewalk       2
        building       3
        wall           4
        fence          5
        pole           6
        traffic light  7
        traffic sign   8
        vegetation     9
        terrain        10
        sky            11
        person         12
        rider          13
        car            14
        truck          15
        bus            16
        train          17
        motorcycle     18
        license plate  19
        others         0
        '''

        smnt = torch.where((smnt<7) | (smnt==9) | (smnt==10) | (smnt==14) | (smnt==15) | (smnt==16) | (smnt==18) | (smnt==29) | (smnt==30), torch.zeros(smnt.shape), smnt)

        smnt = smnt - 6
        smnt = torch.where(smnt >= 5, smnt-2, smnt)
        smnt = torch.where(smnt >= 9, smnt-3, smnt)
        smnt = torch.where(smnt >= 8, smnt-1, smnt)
        smnt = torch.where(smnt >= 19, smnt-2, smnt)
        smnt = torch.clamp(smnt, min=0)

        return smnt

    def label_heat_map(self, smnt):

        smnt = torch.round(TF.to_tensor(smnt)*256)

        return self.label(smnt)


def train(device, use_spatial=True, use_feature=True, d=8, total_epoch=180, init_lr=0.01):

    cityscapes = Cityscapes()
    resnet = resnet_with_dgcn(use_spatial, use_feature, d)
    trainloader = cityscapes.trainloader

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = optim.SGD(resnet.parameters(), lr=init_lr, momentum=0.9,weight_decay=0.0001)
    resnet.to(device)

    print('Training Start')
    for epoch in range(total_epoch):

        for i, data in enumerate(trainloader, 0):

            try:
                img, semantic = data
                img = img.to(device)
                optimizer.zero_grad()

                prediction = resnet(img)['out']
                prediction = prediction.to(device)
                batch, _, H, W = semantic.shape
                semantic = torch.reshape(semantic, (batch, H, W))
                semantic = semantic.long().to(device)

                loss = criterion(prediction, semantic)
                optimizer = adaptive_lr(optimizer, epoch, total_epoch, init_lr)
                loss.backward()
                optimizer.step()

                if i%100 == 99:
                    print("[Epoch: %d, Trained Images in Batch: %d] Loss = %f" % (epoch+1, i+1, loss.item()))

            except InterruptedError:
                continue

        if (epoch+1)%20 == 0:

            torch.save(resnet.state_dict(), './model/resnet_dgcn_epoch_{}.pth'.format(epoch+1))

    print('Training Finished')

    return


def main():

    if not torch.cuda.is_available():

        print("Training is only for CUDA available environment")
        return

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    train(device)

    return


if __name__ == "__main__":

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    sys.exit(main())


















