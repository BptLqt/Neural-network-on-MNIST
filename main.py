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
optimizer = torch.optim.SGD(params,lr=lr)

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
