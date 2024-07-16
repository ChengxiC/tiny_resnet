# Chengxi Chu, Universiti Malaya
import torch


def test(model, device, loader_test):

    model = model.to(device)
    correct = 0  # record the number of corrects
    total = 0  # the total number

    # testing
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(loader_test):
            total += x.size(0)
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)  # [b, 10]
            # get the biggest value. [b, 10] => [b]
            predictions = torch.argmax(logits, dim=1)
            correct += torch.eq(predictions, labels).cpu().float().sum()

    return correct/total







