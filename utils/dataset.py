import os
import cv2
from torch.utils import data
import numpy as np
import yaml

class DatasetLoader(data.Dataset):
    def __init__(self, root, classes, imgsize, color=0):
        self.classnames = os.listdir(root)
        for key in classes.keys():
            assert key in self.classnames, "check your calssfiles and your labels defined"

        self.images = []
        self.labels = []
        self.classes = classes
        self.root = root
        self.H = imgsize[0]
        self.W = imgsize[1]
        self.color = color
        self.checklist= self.GetImageLabels()
        print("Datasets checked completed：\n")
        for key in self.checklist.keys():
            print(key+": "+str(self.checklist[key])+"")
    def GetImageLabels(self):
        checklist = {}
        for key in self.classes.keys():
            path = os.path.join(self.root, key)
            names = os.listdir(path)
            checklist[key]=len(names)
            label = self.classes[key]
            for name in names:
                image = os.path.join(path, name)
                self.images.append(image)
                self.labels.append(label)
        return checklist

    def imageresize(self, image):
        image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return image

    def __getitem__(self, i):
        img_p = self.images[i]
        label = self.labels[i]
        data = cv2.imread(img_p, self.color)
        data = self.imageresize(data)
        if(self.color == 0):
            img = np.expand_dims(data, axis=0)
            img = np.ascontiguousarray(img)
        else:
            img = data.transpose((2, 0, 1))[::-1]
        return  img, label

    def __len__(self):
        return len(self.images)



if __name__ == '__main__':
    with open('', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    path = result['train']
    classes = result['classes']
    dataset = DatasetLoader(path, classes, [512,512], 0)
    for img, label in dataset:
        print(img, label, img.shape)







