import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from U_net import Unet
from MyDataSet import Slidedataset


if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    print(device)

    batch_size = 64

    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    label_transforms = transforms.ToTensor()

    train_dataset = Slidedataset('data', image_transforms, label_transforms, train=True)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #  [64,3,80,80]

    test_dataset = Slidedataset('data', image_transforms, label_transforms,train=False)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    model = Unet(3, 3).to(device)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 1
    if __name__ == '__main__':
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dt_size = len(train.dataset)
            epoch_loss = 0

            for step, (x, y) in enumerate(train):
                inputs = x.to(device)
                labels = y.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train.batch_size + 1, loss.item()))

            print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

        # torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
