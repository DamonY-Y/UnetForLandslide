from torch.utils.data import Dataset
import PIL.Image as Image
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class Slidedataset(Dataset):
    def __init__(self, root_dir, image_transform=None, label_transform=None, train=True):
        self.root_dir = root_dir
        self.img_path = os.listdir(self.root_dir+'/img_split')
        self.label_path = os.listdir(self.root_dir+'/label_split')

        self.image_transform = image_transform
        self.label_transform = label_transform

        self.mode = train

        if self.mode:
            self.img_path = self.img_path[:int(0.9*len(self.img_path))]
            self.label_path = self.label_path[:int(0.9 * len(self.label_path))]
        else:
            self.img_path = self.img_path[int(0.9 * len(self.img_path)):]
            self.label_path = self.label_path[int(0.9 * len(self.label_path)):]

    def __getitem__(self, indx):
        images = Image.open(os.path.join(self.root_dir+'/img_split', self.img_path[indx]))
        labels = Image.open(os.path.join(self.root_dir+'/label_split', self.label_path[indx]))

        if self.image_transform:
            images = self.image_transform(images)
        if self.label_transform:
            labels = self.label_transform(labels)
        return images, labels

    def __len__(self):
        return len(self.img_path)

# if __name__ == '__main__':
#     image_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#
#     label_transforms = transforms.ToTensor()
#     batch_size = 128
#     train_dataset = Slidedataset('data', image_transforms, label_transforms, train=True)
#     train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#     print(len(train.dataset))
#
#     test_dataset = Slidedataset('data', image_transforms, label_transforms,train=False)
#     test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     print(len(test.dataset))
