import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import nn

batch_size = 256
trans = transforms.ToTensor()

train_set = dataset.MNIST(root="./data", train=True, transform=trans, download=False)
test_set = dataset.MNIST(root="./data", train=False, transform=trans, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                          batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                         batch_size=batch_size,
                                         shuffle=True)
                                        
class Net(nn.Module):
    def __init__(self, train=True):
        super(Net, self).__init__()
        
        self.is_train = train
        self.lin1 = nn.Linear(784, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128,10)
        self.relu = nn.ReLU()
    def forward(self, X):
        x = self.relu(self.lin1(X.reshape(-1,784)))
        if self.is_train == True:
            x = nn.Dropout(p=0.5)(x)
        x = self.relu(self.lin2(x))
        if self.is_train == True:
            x = nn.Dropout(p=0.5)(x)
        x = self.lin3(x)
        return x
net = Net()

criterion = nn.CrossEntropyLoss()

n_epochs, lr = 20, 0.1
opt = torch.optim.SGD(net.parameters(),lr=lr)

def train_batch(X, y, opt, net, criterion):
  opt.zero_grad()
  y_hat = net(X)
  loss = criterion(y_hat, y)
  loss.backward()
  opt.step()
  return loss.data

for epoch in range(n_epochs):
  av_loss = 0
  net.train()
  for batch_idx,(X, y) in enumerate(train_loader):
    av_loss += train_batch(X, y, opt, net, criterion)
  print("epoch {}/{}, average loss : {:.5f}".format(epoch, n_epochs, av_loss))

acc = 0
for _, (X,y) in enumerate(test_loader):
  corr = torch.sum(torch.argmax(net(X),dim=1) == y)
  acc += corr/len(X)
print("Précision sur le jeu de test : ", acc/len(test_loader))
