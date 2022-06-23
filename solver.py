import os
import time
import numpy as np
import torch
import torch.nn as nn
from model import Net
from data import generate_loader

class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        self.net = Net(opt).to(self.dev)
        if opt.multigpu: # if you want to use only some gpus, nn.DataParallel(, device_ids = [0, 1])
            self.net = nn.DataParallel(self.net).to(self.dev)
        
        print("# params:", sum(map(lambda x: x.numel(), self.net.parameters())))
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.dev)
        self.optim = torch.optim.RMSprop(self.net.parameters(), opt.lr, weight_decay=opt.weight_decay, 
                                        alpha=0.9, eps=0.001, momentum=0.9)
        
        self.train_loader = generate_loader('train', opt)
        print("train set ready")
        self.val_loader = generate_loader('val', opt)
        print("validation set ready") 
        self.t1, self.t2 = None, None
        self.best_acc, self.best_epoch = 0, 0

    def fit(self):
        opt = self.opt
        self.t1 = time.time()
        print("let's stat training")            
        for epoch in range(opt.n_epoch):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):
                images = inputs[0].to(self.dev)
                labels = inputs[1].to(self.dev)
                preds = self.net(images)
                loss = self.loss_fn(preds, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            if (epoch + 1) % opt.eval_epoch == 0:
                
                val_acc = self.eval(self.val_loader) 
                self.t2 = time.time()
                eta = (self.t2-self.t1) * (self.opt.n_epoch - epoch) / 3600
                if val_acc >= self.best_acc:
                    self.best_acc, self.best_epoch = val_acc, epoch
    
                self.save(epoch + 1)
                print("Epoch [{}/{}] Loss: {:.3f}, Test Acc: {:.3f}".
                    format(epoch+1, opt.n_epoch, loss.item(), val_acc))
                print("Best: {:.2f} @ {}, ETA: {:.1f}".
                    format(self.best_acc, self.best_epoch + 1, eta)) 
                self.t1 = time.time()

    @torch.no_grad()
    def eval(self, data_loader):
        opt = self.opt
        loader = data_loader
        self.net.eval()
        num_correct, num_total = 0, 0
        
        for inputs in loader:
            images = inputs[0].to(self.dev)
            labels = inputs[1].to(self.dev)
            outputs = self.net(images)
            _, preds = torch.max(outputs.detach(), 1)
            
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
            
        return num_correct / num_total
    
    def save(self, epoch):
        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name, self.opt.model_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, self.opt.model_name, str(epoch)+".pt")
        torch.save(self.net.state_dict, save_path)
