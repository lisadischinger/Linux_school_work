from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter  # for pytorch below 1.14
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
# from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc1_bn = nn.BatchNorm1d(512)           # added for step 1
        self.fc15 = nn.Linear(512, 512)             # added for step 2
        self.fc2 = nn.Linear(512, 10)

        # load old parameters into the system just for initialization
        self.load_older_state_dict()                # added for step 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_bn(self.fc1(x)))            # added for step 1
        x = self.fc15(x)                                # added for step 2
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def load_older_state_dict(self):
        pretrained_dict = torch.load('Batch_Norm.pth')
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train() # Why would I do this?
    return total_loss / total, correct.float() / total



#
# def lmd_plot(x, y, x_label, y_label, title):  # save_plot, path):
#     # this function just plots the basics and includes labels and such
#     plt.plot(x, y)
#     plt.ylabel(y_label)
#     plt.xlabel(x_label)
#     plt.title(title)
#     # if save_plot:
#     #     plt.savefig(path + title + ".png")
#     # plt.close()
#     plt.show(block=True)


def lmd_overlay_plot(x_data, y1_data, y2_data, x_label, y_label, labels, plt_title):

    plt.plot(x_data, y1_data, 'r', label=labels[0])
    plt.plot(x_data, y2_data, 'b', label=labels[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

    # plt.show()


if __name__ == "__main__":
    BATCH_SIZE = 32                     # mini_batch size
    MAX_EPOCH = 10                      # maximum epoch to train

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    net = Net().cuda()
    net.train()                                         # Why would I do this?

    writer = SummaryWriter(log_dir='./log')
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)        # use up to step 2
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)                 # added for step 3

    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_loss_list = []

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data                               # get the inputs

            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()       # wrap them in Variable

            optimizer.zero_grad()                       # zero the parameter gradients

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

        # writer.add_scalar('train_acc', train_acc)
        # writer.add_scalar('test_acc', test_acc)
        # writer.add_scalar('train_loss', train_loss)
        # writer.add_scalar('test_loss', test_loss)

        writer.add_scalars('Accuracy', {'train_acc':100*train_acc, 'test_acc':100*test_acc}, epoch)
        writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)

        # train_accuracy_list.append(100. * train_acc)
        # train_loss_list.append(train_loss)
        # test_accuracy_list.append(100. * test_acc)
        # test_loss_list.append(test_loss)



    # lmd_plot(range(len(train_accuracy_list)), train_accuracy_list, "Epochs", "Accuracy [%] ", "Train Accuracy")
    # plot_buf = lmd_overlay_plot(range(len(train_accuracy_list)), train_accuracy_list, test_accuracy_list,
    #                  "Epochs", "Accuracy", ["Training", "Test"], "Accuracy with Batch Norm")
    #
    # image = PIL.Image.open(plot_buf)
    # image = ToTensor()(image).unsqueeze(0)
    #
    # writer.add_image('Accuracy', image)

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'Adam_training.pth')


