W1 = nn.Parameter(torch.randn(784, 256, requires_grad=True)*0.01)
b1 = nn.Parameter(torch.zeros(256, requires_grad=True))
W2 = nn.Parameter(torch.randn(256,10, requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(10, requires_grad=True))

params = [W1,b1,W2,b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

def net(X):
    X = X.reshape((-1,784))
    H = relu(X@W1+b1)
    return (H@W2+b2)

criterion = nn.CrossEntropyLoss()

n_epochs, lr = 20, 0.1
opt = torch.optim.SGD(params,lr=lr)

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
