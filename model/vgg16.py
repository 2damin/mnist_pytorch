import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train = False):
        super().__init__()
        self.batch = batch
        self.n_class = n_classes
        self.in_c, self.in_w, self.in_h = in_channel, in_width, in_height
        self.is_train = is_train
        self.module_cfg = []
        self.module_list = nn.Sequential()
        #layer1 nn.Conv2d(self.in_channel, 6, kernel_size=5, stride= 1, padding=0)
        layer0 = nn.Sequential(nn.Conv2d(self.in_c,64,kernel_size=3, padding=1, stride=1),
                               nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1),
                               nn.MaxPool2d(kernel_size=2))
        self.module_list.add_module('layer0', layer0)
        self.module_cfg.append('layer0')
        layer1 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3, padding=1, stride=1),
                               nn.Conv2d(128,128,kernel_size=3,padding=1,stride=1),
                               nn.MaxPool2d(kernel_size=2))
        self.module_list.add_module('layer1', layer1)
        self.module_cfg.append('layer1')
        layer2 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3, padding=1, stride=1),
                               nn.Conv2d(256,256,kernel_size=3,padding=1,stride=1),
                               nn.MaxPool2d(kernel_size=2))
        self.module_list.add_module('layer2', layer2)
        self.module_cfg.append('layer2')
        layer3 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3, padding=1, stride=1),
                               nn.Conv2d(512,512,kernel_size=3,padding=1,stride=1),
                               nn.Conv2d(512,512,kernel_size=3,padding=1,stride=1),
                               nn.MaxPool2d(kernel_size=2))
        self.module_list.add_module('layer3', layer3)
        self.module_cfg.append('layer3')
        layer4 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3, padding=1, stride=1),
                               nn.Conv2d(512,512,kernel_size=3,padding=1,stride=1),
                               nn.Conv2d(512,512,kernel_size=3,padding=1,stride=1),
                               nn.MaxPool2d(kernel_size=2))
        self.module_list.add_module('layer4', layer4)
        self.module_cfg.append('layer4')
        fc5 = nn.Linear(self.in_w//32 * self.in_h//32 * 512, 4096)
        fc6 = nn.Linear(4096, 4096)
        fc7 = nn.Linear(4096, 10)
        self.module_list.add_module('layer5', fc5)
        self.module_cfg.append('layer5')
        self.module_list.add_module('layer6', fc6)
        self.module_cfg.append('layer6')
        self.module_list.add_module('layer7', fc7)
        self.module_cfg.append('layer7')

    def forward(self, x):
        for module_name, module in zip (self.module_cfg, self.module_list):
            if module_name == "layer5":
                x = x.view(self.batch,-1)
            x = module(x)
        x = nn.functional.softmax(x, dim=1)
        if self.is_train is False:
            x = torch.argmax(x, dim=1)
        return x
        

