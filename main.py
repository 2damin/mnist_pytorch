import torch
import os
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
from util.tools import *

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--mode', dest='mode', help="Train / Test",
                        default='test', type=str)
    parser.add_argument('--download', dest='download', help="Whether to download MNIST dataset",
                        default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', help="output directory",
                        default="./output", type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint model',
                        default=None, type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args
    

def get_data(my_transform):
    print("get_data")
    download_root = './mnist_dataset'
    train_dataset = MNIST(download_root, transform=my_transform, train=True, download=args.download)
    eval_dataset = MNIST(download_root, transform=my_transform, train=False, download=args.download)
    test_dataset = MNIST(download_root, transform=my_transform, train=False, download=args.download)
    return train_dataset,eval_dataset,test_dataset

def main():
    print(torch.__version__)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

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
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=1, pin_memory=True, drop_last=False, shuffle=False)
    test_loader = DataLoader(eval_dataset, batch_size=1, num_workers=1, pin_memory=True, drop_last=False, shuffle=False)
    
    #Get model
    _model = models.get_model("lenet5")

    #summary.summary(model,(1,32,32))    
    
    if args.mode == "train":
        torch_writer = SummaryWriter(args.output_dir)
        model = _model(batch = 8, n_classes=10, in_channel=1, in_width=32, in_height=32, is_train=True)
        model.to(device)
        model.train()
        # writer = SummaryWriter("runs")
        # dataiter = iter(train_loader)
        # images, labels = dataiter.next()
        # img_grid = torchvision.utils.make_grid(images)
        # writer.add_image('four_mnist_images', img_grid)
        
        #optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        epoch = 15
        criterion = loss.get_criterion(crit="mnist", device=device)
        iter = 0
        for e in range(epoch):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]
                img = img.to(device)
                gt = gt.to(device)
                out = model(img)
                loss_val = criterion(out, gt)
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss_val.item()
                # writer.add_scalar("Loss/train", loss_val, e)
                iter += 1
                if iter % 100 == 0:
                    print("{}epoch {}th iter loss : {}".format(e, i, loss_val.item()))
                    torch_writer.add_scalar("lr", get_lr(optimizer), iter)
                    # torch_writer.add_scalar('example/sec', latency, iter)
                    torch_writer.add_scalar("loss", loss_val.item(), iter)
            total_loss = total_loss / i
            scheduler.step()
            print("-----{} epoch loss : {}".format(e, total_loss))
            torch.save(model.state_dict(), args.output_dir +"/model_epoch"+str(e)+".pt")
        # writer.flush()
        print("Train end")

    elif args.mode == "eval":
        model = _model(batch = 1, n_classes=10, in_channel=1, in_width=32, in_height=32)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        acc = 0
        num_eval = 0
        for i, batch in enumerate(eval_loader):
            img = batch[0]
            gt = batch[1]
            img = img.to(device)
            out = model(img)
            out = out.cpu()
            if out == gt:
                acc += 1
            num_eval += 1
        
        print("Evaluation score : {} / {}".format(acc, num_eval))
    elif args.mode == "test":
        model = _model(batch = 1, n_classes=10, in_channel=1, in_width=32, in_height=32)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        for i, batch in enumerate(test_loader):
            img = batch[0]
            img = img.to(device)
            out = model(img)
            out = out.cpu()
            print(out)
            #show input_image
            show_img(img[0].numpy(), str(out.item()))
    

if __name__ == "__main__":
    args = parse_args()
    main()
    