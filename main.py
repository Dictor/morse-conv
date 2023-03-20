import torch
from load_data import load_data
from model import MorseCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print("현재 device : {}".format(device))

learning_rate = 0.001
training_epochs = 30
batch_size = 100

xtr, ytr, xva, yva, xte, yte = load_data("jeonghyun.npz")
model = MorseCNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_train_data = len(ytr)
total_batch = int(total_train_data / batch_size)
print("총 학습 데이터 {0}개, 총 배치 수 {1}개".format(total_train_data, total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for current_set in range(batch_size):
        idx_from = batch_size * current_set
        idx_to = batch_size * (current_set + 1)
        x = torch.from_numpy(xtr[idx_from : idx_to]).to(device).float()
        x = x.unsqueeze(1)
        y = torch.from_numpy(ytr[idx_from : idx_to]).to(device).float()
        _, y = y.max(dim=1) # parse one hot to label
        y = torch.where(y < 36, y, 36)
        optimizer.zero_grad()
        hypothesis = model(x)
        cost = criterion(hypothesis, y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

with torch.no_grad():
    model.eval()
    corr = 0
    running_loss = 0
    test_set_length = len(xva)

    for current_set in range(test_set_length):
        x = torch.from_numpy(xva[current_set]).to(device).float()
        x = x.unsqueeze(0).unsqueeze(0) # make dummy 3D tensor width batch size 1
        y = torch.from_numpy(yva[current_set]).to(device).float()
        output = model(x)
        _, predict = output.max(dim=1)
        corr += torch.sum(predict.eq(y)).item()
        
    acc = corr / test_set_length
    print("Accuracy: {}".format(acc))

torch.save(model.state_dict(), "model")