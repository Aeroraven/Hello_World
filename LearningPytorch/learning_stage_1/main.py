import torch
import torchvision

ds = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
testds = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
train_dl = torch.utils.data.DataLoader(ds, batch_size=64)
test_dl = torch.utils.data.DataLoader(testds, batch_size=64)


class MnistNN(torch.nn.Module):
    def __init__(self):
        super(MnistNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


model = MnistNN().to("cpu")

FX= None

def train(dataloader,model,loss_func,opt_func):
    global FX
    size = len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        X=X.to("cpu")
        FX=X
        y=y.to("cpu")
        pred=model(X)
        loss=loss_func(pred,y)
        opt_func.zero_grad()
        loss.backward()
        opt_func.step()
        if batch%100==0:
            print("Train batch:"+str(batch)+" Loss="+str(loss.item()))

def test(dataloader,model,loss_func):
    size=len(dataloader.dataset)
    nbat = len(dataloader)
    model.eval()
    test_loss,correct=0,0
    with torch.no_grad():
        for X,y in dataloader:
            X=X.to("cpu")
            y=y.to("cpu")
            pred=model(X)
            test_loss+=loss_func(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print("acc:"+str(correct/size))

epoch = 5
for t in range(epoch):
    print("Epoch "+str(t))
    train(train_dl,model,torch.nn.CrossEntropyLoss(),torch.optim.SGD(model.parameters(), lr=1e-3))
    test(test_dl,model,torch.nn.CrossEntropyLoss())
print("Done!")