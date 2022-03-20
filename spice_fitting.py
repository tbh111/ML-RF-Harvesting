# Python script for training, given multi parameters: input power, load resistance and working frequency,
# to predict transmission efficiency
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as var
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


use_pretrained_model = False
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
writer = SummaryWriter('./log')  # tensorboard --logdir=./log --port 8123


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(14, 400)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(400, 600)
        self.relu1 = nn.ReLU()
        self.linear3 = nn.Linear(600, 1)

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu1(y)
        y = self.linear3(y)
        return y


def split_data():
    mat = np.loadtxt(open('./spice_data/spice_reformat_dataset_mass_f=2.3-2.6.csv', 'r'), delimiter=',')
    param = mat.astype(np.float32)

    scales_list = []
    minmax_array = np.empty(shape=[param.shape[0], 0])  # minmax data = dataset size(row, 13)
    for i in range(param.shape[1]):  # normalize param in each column
        temp_data = param[:, i].reshape(1, -1)
        temp_data = np.transpose(temp_data)
        temp_scales = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(temp_data)
        temp_minmax = temp_scales.transform(temp_data)
        minmax_array = np.hstack((minmax_array, temp_minmax))
        scales_list.append(temp_scales)

    param_train, param_test = train_test_split(
        minmax_array, test_size=0.3, random_state=1
    )

    print(param_train.shape)  # (50408, 14)
    print(param_test.shape)   # (21604, 14)

    return param_train, param_test, scales_list, minmax_array


def train(param_train):

    net_input = param_train[:, 1:15]
    net_target = param_train[:, 0]

    net_input = var(torch.Tensor(net_input))
    net_target = var(torch.Tensor(net_target.reshape(-1,1)))
    net_input = net_input.cuda()
    net_target = net_target.cuda()

    if use_pretrained_model:
        model = torch.load('model_spice_mass_with_f.pth').to(device)
        print('use pretrained model')
    else:
        model = NeuralNetwork().to(device)
        print('use new model')
    print(model)

    num_epochs = 500000
    learning_rate = 3e-4
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9, last_epoch=-1)

    for epoch in range(num_epochs):
        net_output = model(net_input)
        loss = criterion(net_output, net_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar('loss', loss.item(), epoch, new_style=True)
        # writer.add_scalar('lr', optimizer.param_groups[-1]['lr'])
        if (epoch+1) % 200 == 0:
            print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, loss.item()))
        if (epoch+1) % 1000 == 0:
            print('Saving checkpoint')
            torch.save(model.state_dict(), './checkpoint/epoch='+str(epoch+1)+'.pth')
        if loss.item() < 2.5e-5:
            print('Current Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, loss.item()))
            print('Goal Reached, Stop Training')
            break

    torch.save(model.state_dict(), 'model_spice_mass_with_f.pth')
    print('Model Saved')


def test(param_test, scales_list, minmax_array):
    net_input = param_test[:, 1:15]
    net_target = param_test[:, 0]
    net_input = var(torch.Tensor(net_input))
    # net_target = var(torch.Tensor(net_target))

    model = NeuralNetwork()

    model.load_state_dict(torch.load('model_spice_mass_with_f.pth'))
    net_predict = model(net_input).detach().numpy()
    net_predict = scales[0].inverse_transform(net_predict)

    eff_test = param_test[:, 0]
    eff_test = scales[0].inverse_transform(eff_test.reshape(-1, 1))
    error = (eff_test - net_predict)
    print(error)
    pin_test = param_test[:, 3]
    pin_test = scales[3].inverse_transform(pin_test.reshape(-1, 1))
    scatter1 = plt.scatter(pin_test, net_predict*100, c='b', s=1.2)
    scatter2 = plt.scatter(pin_test, eff_test*100, c='r', s=1.2)
    scatter3 = plt.scatter(pin_test, error * 100, c='g', s=1.2)
    plt.xlabel('Pin/dBm')
    plt.ylabel('Eff/%')
    plt.legend([scatter1, scatter2, scatter3], ['Network Output', 'Actual Efficiency', 'Error'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    param_train, param_test, scales, minmax = split_data()
    # train(param_train)
    test(param_test, scales, minmax)
    writer.close()