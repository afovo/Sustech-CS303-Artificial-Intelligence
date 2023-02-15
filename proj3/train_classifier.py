from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.utils.data as Data
import hiddenlayer as hl

data = torch.load("data.pth")
label = data["label"]
feature = data["feature"]

X_train = feature[:59500]
X_test = feature[59500:]
y_train = label[:59500]
y_test = label[59500:]


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self)
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=160, bias=True),
            nn.ReLU())
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=160, out_features=80, bias=True),
            nn.ReLU())
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=80, out_features=10, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.hidden3(fc2)
        return fc1, fc2, output


if __name__ == '__main__':

    model = FCNN()

    train_data = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=500, shuffle=True, num_workers=0)
    for step, (batch_x, batch_y) in enumerate(train_loader):
        if step > 0:
            break
        print(step, batch_x.shape, batch_y.shape)

    optomizerAdam = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFunc = nn.CrossEntropyLoss()
    history1 = hl.History()
    canvas1 = hl.Canvas()
    logStep = 25
    for epoch in range(15):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            _, _, output = model(batch_x)
            train_loss = lossFunc(output, batch_y)
            optomizerAdam.zero_grad()
            train_loss.backward()
            optomizerAdam.step()

            niter = epoch * len(train_loader) + step + 1
            if niter % logStep == 0:
                _, _, output = model(X_test)
                _, pre_lab = torch.max(output, 1)
                test_accuracy = accuracy_score(y_test, pre_lab)
                history1.log(niter, train_loss=train_loss, test_accuracy=test_accuracy)
                with canvas1:
                    canvas1.draw_plot(history1['train_loss'])
                    canvas1.draw_plot(history1['test_accuracy'])

    torch.save(model.state_dict(), 'D:/Grade3/CS303 AI/Proj/proj3/classifier.pth')
