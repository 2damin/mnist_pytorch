import torch
import numpy as np
import PIL
import sys
import argparse
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torchsummary as summary
from torch.utils.tensorboard import SummaryWriter
import torchvision
from model import models
from loss import loss

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--mode', dest='mode', help="Train / Test",
                        default='test', type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args
    

def get_data(my_transform):
    print("get_data")
    download_root = './mnist_dataset'
    train_dataset = MNIST(download_root, transform=my_transform, train=True, download=False)
    eval_dataset = MNIST(download_root, transform=my_transform, train=False, download=False)
    test_dataset = MNIST(download_root, transform=my_transform, train=False, download=False)
    return train_dataset,eval_dataset,test_dataset

def main():
    print(torch.__version__)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    #Get Dataset
    mnist_transform = transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(1.0,))
    ])
    train_dataset, eval_dataset, test_dataset = get_data(mnist_transform)
    
    #Make Dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, pin_memory=True, drop_last=True, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=1, pin_memory=True, drop_last=True, shuffle=True)
    test_loader = DataLoader(eval_dataset, batch_size=1, num_workers=1, pin_memory=True, drop_last=True, shuffle=True)
    
    #Get model
    _model = models.get_model("lenet5")

    #summary.summary(model,(1,32,32))    
    
    if args.mode == "train":
        model = _model(batch = 8, n_classes=10, in_channel=1, in_width=32, in_height=32, is_train=True)
        model.to(device=device)
        model.train()
        writer = SummaryWriter("runs")
        # dataiter = iter(train_loader)
        # images, labels = dataiter.next()
        # img_grid = torchvision.utils.make_grid(images)
        # writer.add_image('four_mnist_images', img_grid)
        
        #optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        epoch = 2
        criterion = loss.get_criterion(crit="mnist")
        for e in range(epoch):
            total_loss = 0
            i = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                img = batch[0]
                gt = batch[1]
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                out = model(img)
                loss_val = criterion(out, gt)
                loss_val.backward()
                optimizer.step()
                total_loss += loss_val.item()
                writer.add_scalar("Loss/train", loss_val, e)
                i += 1
                if i % 100 == 0:
                    print("{}th iter loss : {}".format(i, loss_val.item()))
            total_loss = total_loss / i
            print("--{} epoch loss : {}".format(e, total_loss))
            scheduler.step()
        torch.save(model.state_dict(), "./model.pt")
        writer.flush()
        print("Train end")

    elif args.mode == "test":
        model = _model(batch = 1, n_classes=10, in_channel=1, in_width=32, in_height=32)
        checkpoint = torch.load("./model.pt")
        model.load_state_dict(checkpoint)
        model.to(device=device)
        model.eval()
        acc = 0
        num_eval = 0
        for i, batch in enumerate(eval_loader):
            img = batch[0]
            gt = batch[1]
            img = img.cuda(non_blocking=True)
            out = model(img)
            out = out.cpu()
            if out == gt:
                acc += 1
            num_eval += 1
        
        print("Eval : {} / {}".format(acc, num_eval))
    

if __name__ == "__main__":
    args = parse_args()
    main()
    