import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as var
import torch
from scipy import io

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

mat = io.loadmat('2820.mat')
Eff = mat['Eff']
Pin = mat['Pin']
print(Eff.shape)
print(Pin.shape)

def get_data(x, w, b, d):
    c, r = x.shape
    y = (w * x * x + b * x + d) + (0.1 * (2 * np.random.rand(c, r) - 1))
    return (y)


# xs = np.arange(0, 3, 0.01).reshape(-1, 1)
# ys = get_data(xs, 1, -2, 3)
xs = Pin
ys = Eff


xs = var(torch.Tensor(xs))
ys = var(torch.Tensor(ys))
xs = xs.cuda()
ys = ys.cuda()

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(1, 200)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 800)
        self.relu = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(800, 1)

        self.criterion = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, input):
        y = self.linear1(input)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.relu(y)
        y = self.linear3(y)
        return y



model = NeuralNetwork().to(device)
print(model)
for e in range(1000001):
    y_pre = model(xs)

    loss = model.criterion(y_pre, ys)
    if (e % 200 == 0):
        print(e, loss.data)

    # Zero gradients
    model.opt.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    model.opt.step()

ys_pre = model(xs)
xs = xs.cpu()
ys = ys.cpu()
ys_pre = ys_pre.cpu()
plt.title("curve")
plt.plot(xs.data.numpy(), ys.data.numpy())
plt.plot(xs.data.numpy(), ys_pre.data.numpy())
plt.legend("ys", "ys_pre")
plt.show()

torch.save(model.state_dict(), "model.pth")