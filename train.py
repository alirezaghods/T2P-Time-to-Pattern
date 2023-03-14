
import torch
from torch import Tensor
from utils.plotting_utils import plot_line
from module.loss import loss
from utils.util import compute_sparsity
import torch.optim as optim
import time

def train_one_epoch(epoch_index,data_loader,model,optimizer,a, lambda_1, lambda_2, device):
    running_loss = 0.
    running_sparsity = 0.

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output, 4)
        output = torch.squeeze(output, 2)
        elbo = loss(output,data,model.alpha,a, lambda_1, lambda_2, device)
        sparsity = compute_sparsity(zs=model.z, norm=True)
        elbo.backward()
        optimizer.step()
        running_loss += elbo.item()
        running_sparsity += sparsity

    return running_loss / len(data_loader), running_sparsity / len(data_loader)

def train(epochs,model,data_loader,learning_rate,a, lambda_1, lambda_2, device):
    print('#'*10+'Start trining'+'#'*10)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elbo = list()
    sparsity = list()
    device = torch.device(device)
    start_time = time.time()
    for epoch in range(epochs):
        model.train(True)
        
        avg_loss, avg_sparsity = train_one_epoch(epoch,data_loader,model, optimizer, a, lambda_1, lambda_2, device)
        elbo.append(avg_loss)
        sparsity.append(avg_sparsity.item())

        print(f'Epoch: {epoch:06} | ELBO: {avg_loss:.3f} | Sparsity: {avg_sparsity:.3f}'.format(epoch, avg_loss, avg_sparsity))
    print('#'*10+'End trining'+'#'*10)
    print("--- Training time in seconds: %s seconds ---" % (time.time() - start_time)) 
    

    return elbo, sparsity