# CreateTime：2020-11-10
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
# from MyDataSet import Slidedataset

# root_dir = r'data'
# img_path_ = root_dir + '/img_split'
# print(os.listdir(img_path_))

# img_path = os.listdir(root_dir + '/img_split')
# print(img_path)
# label_path = os.listdir(root_dir + '/label_split')
# indx = 10

# print(os.path.join(root_dir + '\img_split', img_path[indx]))
# images = Image.open(os.path.join(root_dir + '/img_split', img_path[indx]))
# labels = Image.open(os.path.join(root_dir + '/label_split', label_path[indx]))


import torch
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from U_net import Unet
from MyDataSet import Slidedataset


batch_size = 1

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

label_transforms = transforms.ToTensor()

test_dataset = Slidedataset('data', image_transforms, label_transforms, train=False)
test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = Unet(3, 3)
print(test)

# plt.ion()
# with torch.no_grad():
# image = Image.open('image.png')
# plt.imshow(image)

for num, (x, y) in enumerate(test):
    # y = y.squeeze().numpy()
    # print(type(y))
    # print(y.shape)
    # y = np.transpose(y, (1, 2, 0))
    # print(y.shape)

    out = model(x)
    out=out.squeeze().detach().numpy()
    print(type(out))
    print(out.shape)
    out = np.transpose(out, (1, 2, 0))

    x = x.squeeze().numpy()
    x = np.transpose(x, (1, 2, 0))
    # print(out.shape)
    # out = torch.squeeze(y).numpy()
    # print(out.shape)
    plt.imshow(x)
    # plt.imshow(out)
    break
    #     y = model(x).sigmoid()
    #     img_y = torch.squeeze(y).numpy()
    #     plt.imshow(img_y)
    #     plt.pause(0.01)
plt.show()




# if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    #
    # image_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])
    #
    # label_transforms = transforms.ToTensor()
    #
    # slides_dataset = Slidedataset(r'data', image_transforms, label_transforms)
    # batch_size = 60
    # dataloaders = DataLoader(slides_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # print(dataloaders)
    # dt_size = len(dataloaders.dataset)
    # print(dataloaders.dataset)
    # print(dt_size)
    # for i, (x, y) in enumerate(dataloaders):
    #     print('i',i)
    #     print('_____________')
    #     print('x:', x.shape)
    #     print('____________________')
    #     print('y:', y.shape)

    # image, label = iter(dataloaders).next()
    # sample = image[0].squeeze()
    # sample = sample.permute((1, 2, 0)).numpy()
    # sample *= [0.229, 0.224, 0.225]
    # sample += [0.485, 0.456, 0.406]
    # sample = np.clip(sample, 0, 1)
    # plt.imshow(sample)
    # plt.show()
    # root = 'data'
    # img_path = os.listdir(root + '/img_split')
    #
    # print(img_path[:int(0.8*len(img_path))])
    #
    # print(img_path[int(0.8*len(img_path)):])

