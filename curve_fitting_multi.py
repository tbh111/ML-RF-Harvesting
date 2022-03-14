# Python script for training, given multi parameters: input power, load resistance and working frequency,
# to predict transmission efficiency
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as var
import torch
from torch import nn
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(3, 200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200, 400)
        self.relu1 = nn.ReLU()
        self.linear3 = nn.Linear(400, 1)
        # self.relu2 = nn.ReLU()
        # self.linear4 = nn.Linear(600, 400)
        # self.relu3 = nn.ReLU()
        # self.linear5 = nn.Linear(400, 1)

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu1(y)
        y = self.linear3(y)
        # y = self.relu2(y)
        # y = self.linear4(y)
        # y = self.relu3(y)
        # y = self.linear5(y)
        return y


def split_data():
    mat = io.loadmat('2820_multi_param_data.mat')
    eff = mat['Eff'].astype(np.float32)  # (368010,1)
    pin = mat['Pin'].astype(np.float32)
    f0 = mat['F0'].astype(np.float32)
    rl = mat['RL'].astype(np.float32)
    eff_scales = preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(eff)
    eff_minmax = eff_scales.transform(eff)
    pin_scales = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(pin)
    pin_minmax = pin_scales.transform(pin)
    f0_scales = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(f0)
    f0_minmax = f0_scales.transform(f0)
    rl_scales = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(rl)
    rl_minmax = rl_scales.transform(rl)
    scales_list = [eff_scales, pin_scales, f0_scales, rl_scales]

    eff_train, eff_test, pin_train, pin_test, f0_train, f0_test, rl_train, rl_test = train_test_split(
        eff_minmax, pin_minmax, f0_minmax, rl_minmax,
        test_size=0.3, random_state=1
    )
    param_train = np.hstack((eff_train, pin_train, f0_train, rl_train))
    param_test = np.hstack((eff_test, pin_test, f0_test, rl_test))
    print(param_train.shape)  # (294408, 4)
    print(param_test.shape)   # (732602, 4)

    return param_train, param_test, scales_list


def train(param_train):

    net_input = param_train[:, 1:4]
    net_target = param_train[:, 0]

    net_input = var(torch.Tensor(net_input))
    net_target = var(torch.Tensor(net_target.reshape(-1,1)))
    net_input = net_input.cuda()
    net_target = net_target.cuda()

    model = NeuralNetwork().to(device)
    print(model)

    num_epochs = 100000
    learning_rate = 3e-4
    criterion = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        net_output = model(net_input)
        loss = criterion(net_output, net_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 200 == 0:
            print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, loss.item()))
        if loss.item() < 5e-6:
            print('Current Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, loss.item()))
            print('Goal Reached, Stop Training')
            break

    torch.save(model.state_dict(), 'model_multi.pth')
    print('Model Saved')


def test(param_test, scales):
    net_input = param_test[:, 1:4]
    net_target = param_test[:, 0]
    net_input = var(torch.Tensor(net_input))
    # net_target = var(torch.Tensor(net_target))

    model = NeuralNetwork()

    model.load_state_dict(torch.load('model_multi.pth'))
    net_predict = model(net_input).detach().numpy()
    net_predict = scales[0].inverse_transform(net_predict)

    eff_test = param_test[:, 0]
    eff_test = scales[0].inverse_transform(eff_test.reshape(-1, 1))
    error = eff_test - net_predict
    pin_test = param_test[:, 1]
    pin_test = scales[1].inverse_transform(pin_test.reshape(-1, 1))
    scatter1 = plt.scatter(pin_test, net_predict*100, c='b', s=1.2)
    scatter2 = plt.scatter(pin_test, eff_test*100, c='r', s=1.2)
    scatter3 = plt.scatter(pin_test, error * 100, c='g', s=1.2)
    plt.xlabel('Pin/dBm')
    plt.ylabel('Eff/%')
    plt.legend([scatter1, scatter2, scatter3], ['Network Output', 'Actual Efficiency', 'Error'], loc='upper left')
    #
    plt.show()


if __name__ == "__main__":
    param_train, param_test, scales = split_data()
    # train(param_train)
    test(param_test, scales)