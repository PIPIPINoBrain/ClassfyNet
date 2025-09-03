import os
import torch
import cv2
import numpy as np
import argparse



class Predict():
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(args.weight, map_location='cpu')
        self.model.to(self.device)
        self.model.eval()
        self.color = args.color
        self.imgsize = args.img_size

    def imagepreprocess(self, image):
        image = cv2.resize(image, self.imgsize, interpolation=cv2.INTER_AREA)
        if (self.color == 0):
            img = np.expand_dims(image, axis=0)
            img = np.expand_dims(img, axis=0)
            img = np.ascontiguousarray(img)
        else:
            img = image.transpose((2, 0, 1))[::-1]
            img = np.expand_dims(img, axis=0)
        return img

    def predicter(self, path):
        data = cv2.imread(path, self.color)
        img = self.imagepreprocess(data)
        img = torch.from_numpy(img)
        img = img.to(self.device, non_blocking=True).float() * 0.00390625
        pred = self.model(img)
        _, result = torch.max(pred,1)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--path", default = "C:\\Users\\18829\\Desktop\\Classfy\\data\\test\\cat", type=str)   
    parser.add_argument("--color", default= 0, type=int)
    parser.add_argument("--weight", default='.\\run\\best.pth', type=str)
    parser.add_argument("--img_size", default=(224, 224), type=int)
    args = parser.parse_args()

    T = Predict(args)
    path = args.path
    names = os.listdir(path)
    for name in names:
        p = os.path.join(path, name)
        print(T.predicter(p))