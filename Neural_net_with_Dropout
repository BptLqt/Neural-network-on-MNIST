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
optimizer = torch.optim.SGD(net.parameters(),lr=lr)

for epoch in range(n_epochs):
    av_loss = 0
    for batch_idx, (X,y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = net(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        av_loss += loss.data
        if (batch_idx+1) == len(train_loader):
            print("epoch {}, loss : {:.5f}".format(epoch+1, av_loss/len(train_loader)))

acc = 0
for _, (X,y) in enumerate(test_loader):
    corr = 0
    for idx in range(len(X)):
        if torch.argmax(net(X[idx])) == y[idx]:
            corr += 1
    acc += corr/len(X)
print("Précision sur le jeu de test : ", acc/len(test_loader))
