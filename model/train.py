import torch

# from metrics import metric

# # from data_gen import *

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def train(train_loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    correct = 0.0
    total = 0.0
    # lambda_l1 = 0.000001

    for i, (train_data, label) in enumerate(train_loader):
        label = label.to(device)
        train_data = train_data.to(device)
        optimizer.zero_grad()
        out = model(train_data)
        predict = out.max(dim=1).indices
        loss = criterion(out, label)
        # for param in model.parameters():
        #     loss += lambda_l1 * torch.norm(param, 1)
        loss.backward()
        optimizer.step()
        total += label.size(0)
        correct += (predict == label).sum().item()
        epoch_loss += loss.item()
        print(f"batch {i}, loss: {loss.item():.4f}, acc: {correct / total * 100:.2f}%")

    # print('train_loss:{}'.format(l.item()))
    accuracy = correct / total * 100.0
    average_epoch_loss = epoch_loss / len(train_loader)
    return accuracy, average_epoch_loss


def test(data_loader, model, criterion):
    total = 0.0
    correct = 0.0
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            predict = out.max(dim=1).indices
            running_loss += loss.item()
            total += y.size(0)
            correct += (predict == y).sum().item()
        # print('eval_num:{},eval_acc:{:.3f}%,eval_loss:{}'.format(counter,test_acc * 100,l.item()))
    accuracy = correct / total * 100.0
    loss = running_loss / len(data_loader)
    return accuracy, loss


def predict_test(data_loader, model, batch_size):
    model.eval()
    preds = torch.tensor([0]).to(device)
    trues = torch.tensor([0]).to(device)
    with torch.no_grad():
        counter = 0.0
        right = 0.0
        acc = []
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            counter += batch_size
            predict = out.max(dim=1).indices
            # right += (predict == y).sum().item()

            preds = torch.cat((preds, predict))
            trues = torch.cat((trues, y))

    preds = preds.cpu().numpy()
    trues = trues.cpu().numpy()
    mae, mse, rmse, mape, mspe = metric(preds[1:], trues[1:])  # 全部结果算
    # test_acc = (right / counter)
    # print('mae:{:.3f},mse:{:.3f},rmse:{:.3f},mape:{:.3f},mspe:{:.3f}'.format(mae, mse, rmse, mape, mspe))
    # print('test_num:{},test_acc:{:.3f}%'.format(counter,test_acc * 100))
    # acc.append(test_acc)
    return mae, mse, rmse, mape, mspe
