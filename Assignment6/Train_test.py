from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self):
        self.train_losses = []
        
        self.train_acc = []
        

    def train(self, model, device, train_loader, optimizer, criterion, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = criterion(y_pred, target)
            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
            
class Test:
    def __init__(self):
        self.test_losses = []
        self.test_acc = []
        self.misclassified_images = []
        
    def test(self, model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                
                # compare predictions with true label
                for i, (p, t) in enumerate(zip(pred, target)):
                    if p != t:
                        self.misclassified_images.append((data[i], p, t))
                
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))
        return self.misclassified_images

