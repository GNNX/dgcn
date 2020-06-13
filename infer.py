import sys
import torch

from torchvision.transforms import transforms

from resnet_dgcn import resnet_with_dgcn
from PIL import Image

# RGB color dictionary for each class
color_dict = {
        0:  (0,0,0), # unlabeled
        1:  (128,64,128),  # road
        2:  (244,35,232),  # sidewalk
        3:  (70,70,70),    # building
        4:  (102,102,156), # wall
        5:  (190,153,153), # fence
        6:  (153,153,153), # pole
        7:  (250,170,30),  # traffic light
        8:  (220,220,0),   # traffic sign
        9:  (107,142,35),  # vegetation
        10: (152,251,152), # terrain
        11: (70,130,180),  # sky
        12: (220,20,60),   # person
        13: (255,0,0),     # rider
        14: (0,0,142),     # car
        15: (0,0,70),      # truck
        16: (0,60,100),    # bus
        17: (0,80,100),    # train
        18: (0,0,230),     # motorcycle
        19: (119,11,32)    # bicycle
    }

def get_ground_image(img_name):

    city_name = img_name.split('_')[0] + '/'
    if city_name in ['frankfurt/', 'lindau/', 'munster/']:
        set_name = 'val/'
    else:
        set_name = 'train/'

    im = Image.open('./cityscapes/gtFine/' + set_name + city_name + img_name + '_gtFine_color.png')

    return im

def get_input_tensor(img_name):

    city_name = img_name.split('_')[0] + '/'
    if city_name in ['frankfurt/', 'lindau/', 'munster/']:
        set_name = 'val/'
    else:
        set_name = 'train/'

    im = Image.open('./cityscapes/leftImg8bit/' + set_name + city_name + img_name + '_leftImg8bit.png')
    img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))]
        )
    tensor = img_transform(im)
    C, H, W = tensor.shape
    tensor = torch.reshape(tensor, (1, C, H, W))
    return tensor

def prediction_to_img(pred):

    _, C, H, W = pred.shape
    pred = torch.reshape(pred, (C, H, W))
    _, indices = torch.max(pred, dim=0)

    indices = torch.reshape(indices, (1, H, W))
    output = torch.cat((indices, indices, indices))
    output_img = torch.zeros(3, H, W)

    for i in range(C):

        R, G, B = color_dict[i]
        color_mask = torch.cat((R*torch.ones((1, H, W)), G*torch.ones((1, H, W)), B*torch.ones((1, H, W))))
        output_img = torch.where(output == i, color_mask, output_img)

    return transforms.ToPILImage()(output_img/255).convert('RGB')

def main(MODEL_PATH='./model/resnet_dgcn_epoch_180.pth', IMAGE_NAME=''):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = torch.load(MODEL_PATH, map_location=device)
    net = resnet_with_dgcn()
    net.load_state_dict(model)

    print('Model Ready')

    with torch.no_grad():
        prediction = net(get_input_tensor(IMAGE_NAME))['out']
        print('Prediction Finished')

        ground_img = get_ground_image(IMAGE_NAME)
        predicted_img = prediction_to_img(prediction)

        dst = Image.new('RGB', (ground_img.width + predicted_img.width, ground_img.height))
        dst.paste(ground_img, (0,0))
        dst.paste(predicted_img, (ground_img.width,0))
        dst.save(IMAGE_NAME+'_result.png', 'PNG')

        print('Result saved at the result.png')

    return


if __name__ == '__main__':

    sys.exit(main(IMAGE_NAME=sys.argv[1]))









