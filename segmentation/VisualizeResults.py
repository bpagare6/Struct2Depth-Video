import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
from PIL import Image as PILImage
import segmentation.Model as Net
import os
import time
from argparse import ArgumentParser

pallete = [[128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def evaluateModel(model, image, output_dir, up=None, overlay=True, modelType=1, cityFormat=True):
    # gloabl mean and std values
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    # img = cv2.imread(imgName)
    if overlay:
        image_orig = np.copy(image)

    image = image.astype(np.float32)
    for j in range(3):
        image[:, :, j] -= mean[j]
    for j in range(3):
        image[:, :, j] /= std[j]

    # resize the image to 1024x512x3
    image = cv2.resize(image, (1024, 512))
    if overlay:
        image_orig = cv2.resize(image, (1024, 512))

    image /= 255
    image = image.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)  # add a batch dimension
    image_variable = Variable(image_tensor, volatile=True)
    image_out = model(image_variable)

    if modelType == 2:
        image_out = up(image_out)

    classMap_numpy = image_out[0].max(0)[1].byte().cpu().data.numpy()

    classMap_numpy_color = np.zeros(
        (image.shape[1], image.shape[2], image.shape[0]), dtype=np.uint8)
    for idx in range(len(pallete)):
        [r, g, b] = pallete[idx]
        classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
    # cv2.imwrite(output_dir + os.sep + 'c_' +
    #             name.replace('png', 'png'), classMap_numpy_color)

    if overlay:
        # overlayed = (image_orig * 0.5) + (classMap_numpy_color * 0.5)
        overlayed = cv2.addWeighted(
            image_orig, 0.5, classMap_numpy_color, 0.5, 0, dtype=cv2.CV_64F)
        # overlayed = cv2.addWeighted(
        #     image_orig, 0.5, classMap_numpy_color, 0.5, 0, dst=overlayed)
        # cv2.imwrite(output_dir + os.sep + 'over_' +
        #             name.replace('png', 'jpg'), overlayed)
        print("Shape of overlayed:", overlayed.shape)
        return overlayed

    print("Classmap shape:", classMap_numpy_color.shape)
    return classMap_numpy_color

    # if cityFormat:
    #     classMap_numpy = relabel(classMap_numpy.astype(np.uint8))

    # return classMap_numpy


def main(image, output_dir, weightsDir, modelType=1, decoder=True, overlay=True, cityFormat=True):
    # read all the images in the folder
    # image_list = glob.glob(args.data_dir + os.sep + '*.' + args.img_extn)
    up = None
    if modelType == 2:
        up = torch.nn.Upsample(scale_factor=8, mode='bilinear')

    p = 2
    q = 8
    classes = 20
    if modelType == 2:
        # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        modelA = Net.ESPNet_Encoder(classes, p, q)
        model_weight_file = weightsDir + os.sep + 'encoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(
            q) + '.pth'
        if not os.path.isfile(model_weight_file):
            print(
                'Pre-trained model file does not exist. Please check ../pretrained/encoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    elif modelType == 1:
        # Net.Mobile_SegNetDilatedIA_C_stage1(20)
        modelA = Net.ESPNet(classes, p, q)
        model_weight_file = weightsDir + os.sep + 'decoder' + \
            os.sep + 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'
        if not os.path.isfile(model_weight_file):
            print(
                'Pre-trained model file does not exist. Please check ../pretrained/decoder folder')
            exit(-1)
        modelA.load_state_dict(torch.load(model_weight_file))
    else:
        print('Model not supported')
    # modelA = torch.nn.DataParallel(modelA)
    # if args.gpu:
    #     modelA = modelA.cuda()

    # set to evaluation mode
    modelA.eval()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    return evaluateModel(modelA, image, output_dir, up,
                  overlay, modelType, cityFormat)


if __name__ == '__main__':
    '''
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet", help='Model name')
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--img_extn', default="png", help='RGB Image format')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--modelType', type=int, default=1, help='1=ESPNet, 2=ESPNet-C')
    parser.add_argument('--savedir', default='./results', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--decoder', type=bool, default=True,
                        help='True if ESPNet. False for ESPNet-C')  # False for encoder
    parser.add_argument('--weightsDir', default='../pretrained/', help='Pretrained weights directory.')
    parser.add_argument('--p', default=2, type=int, help='depth multiplier. Supported only 2')
    parser.add_argument('--q', default=8, type=int, help='depth multiplier. Supported only 3, 5, 8')
    parser.add_argument('--cityFormat', default=True, type=bool, help='If you want to convert to cityscape '
                                                                       'original label ids')
    parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks in color')
    parser.add_argument('--overlay', default=True, type=bool, help='If you want to visualize the '
                                                                   'segmentation masks overlayed on top of RGB image')
    parser.add_argument('--classes', default=20, type=int, help='Number of classes in the dataset. 20 for Cityscapes')

    args = parser.parse_args()
    assert (args.modelType == 1) and args.decoder, 'Model type should be 2 for ESPNet-C and 1 for ESPNet'
    if args.overlay:
        args.colored = True # This has to be true if you want to overlay
    '''
    pass
    # main(args)
