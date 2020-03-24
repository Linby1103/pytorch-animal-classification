import os
from torch.utils.data import DataLoader
import torch as t
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision as tv
import numpy as np

import matplotlib.image as image

classes={'Abyssinian':0,'bulldog':1,'beagle':2,'Bengal':3,'Birman':4,'Bombay':5,\
'boxer':6,\
'basset_hound':7,\
'american_pit':8,\
'setter':9,\
'British_Shorthair':10,\
'chihuahua':11,\
'Egyptian_Mau':12,\
'english_cocker_spaniel':13,\
'german_shorthaired':14,\
'great_pyrenees':15,\
'havanese':16,\
'japanese_chin':17,\
'keeshond':18,\
'leonberger':19,\
'Maine_Coon':20,\
'miniature_pinscher':21,\
'newfoundland':22,\
'Persian':23,\
'pomeranian':24,\
'pug':25,\
'Ragdoll':26,\
'Russian_Blue':27,\
'saint_bernard':28,\
'samoyed':29,\
'scottish_terrier':30,\
'shiba_inu':31,\
'Siamese':32,\
'Sphyn':33,\
'staffordshire_bull_terrier':34,\
'wheaten':34,\
'yorkshire':36\
}
class SubDateset(data.Dataset):
    def __init__(self,path,transform=None,train=True,test=False):
        super(SubDateset,self).__init__()
        """获取训练数据"""
        self.test = test
        imgs=[os.path.join(path,img) for img in os.listdir(path)]

        # if self.test:
        #
        #     imgs=sorted(imgs,key=lambda x: (int(x.split('.')[-2].split('\\')[-1]),print("int(x.split('.')[-2].split('\\')[-1])",int(x.split('.')[-2].split('\\')[-1]))))
        # else :
        #     imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2]))
        imgs_num=len(imgs)
        #训练集，测试集验证集划分验证：训练=3:7
        if self.test:
            self.imgs=imgs
        elif train:
            self.imgs=imgs[:int(0.7*imgs_num)]
        else :
            self.imgs=imgs[int(0.7*imgs_num):]
        if transform is None:
            normalize=tv.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            if self.test or not train:
                self.transforms=tv.transforms.Compose([
                    tv.transforms.Scale(224),
                    tv.transforms.CenterCrop(224),
                    tv.transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = tv.transforms.Compose([
                    tv.transforms.Scale(256),
                    tv.transforms.RandomResizedCrop(224),
                    tv.transforms.RandomHorizontalFlip(),
                    tv.transforms.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        # 返回一张图片的数据
        img_path = self.imgs[index]
        if image.imread(img_path).shape[2] !=3:
            return None,None
        print("iamge path:",img_path)
        keyword =img_path.rsplit('/')[-1][:-4]

        if self.test:
            label = int(keyword.split('_')[2][:-4])

        else:
            for k, v in classes.items():
                if k in keyword:
                    label=v
            # label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        # 返回数据集中的所有图片张数
        return len(self.imgs)
datapath='D:/workspace/code/pytorch/animal/data/'
batch_size=8
num_workers=0
train_dataset=SubDateset(datapath,train=True)
train_dataloader=DataLoader(train_dataset,batch_size,shuffle=True,num_workers=num_workers)
for ii,(data,label) in enumerate(train_dataloader):
    print('label :{}\n'.format(label))





