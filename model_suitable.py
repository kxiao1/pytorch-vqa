import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

import config
import data
import utils
import colors
from model_baseline import update_learning_rate, total_iterations

# Simple neural network to classify colors given resnet feature maps
class QualityNet(nn.Module):
    # conv2d, max pool, min pool, 1 linear layers 
    # inspired by https://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf
    def __init__(self):
        super(QualityNet, self).__init__()
        self.conv = nn.Conv2d(config.output_features, 256, 1)
        self.maxPool = nn.MaxPool2d(14)
        # pytorch has no built-in minPool self.minPool 
        self.flattener = nn.Flatten()
        self.lin1 = nn.Linear(2048 * 2, 100)
        self.FC = nn.Linear(100, 100)
        self.lin2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.05)

        self.convTest = nn.Conv2d(3, 50, 1)
        self.maxPoolTest = nn.MaxPool2d(2)
        self.flattenerTest = nn.Flatten()
        self.lin1Test = nn.Linear(50 * 1 * 1, 100)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        self.linTestSimp = nn.Linear(2048 * 14 * 14, 1)
        self.linTestInter = nn.Linear(2048 * 7 * 7, 256) 
        self.FCInter = nn.Linear(256, 256)
        self.lin2Inter = nn.Linear(256, 1)

    def forward(self, v):
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
       
        # 'comp'
        # o1 = self.conv(v)
        # print(o1.size())
        maxPool = self.maxPool(v) 
        minPool = -1 * self.maxPool(-1 * v)
        flattenedMax = self.flattener(maxPool)
        flattenedMin = self.flattener(minPool)
        flattened = torch.cat([flattenedMax, flattenedMin], 1)
        FC1 = self.lin1(flattened)
        FC2 = self.FC(self.relu(FC1))
        res = self.lin2(self.drop(self.relu(FC2)))
        return self.sigmoid(res)

        # 'simp'
        # flattened = self.flattener(v)
        # res = self.linTestSimp(flattened)
        # return self.sigmoid(res)

        # 'inter'
        # maxPoolTest = self.maxPoolTest(v)
        # flattened = self.flattener(maxPoolTest)
        # FC1 = self.linTestInter(flattened)
        # FC2 = self.FCInter(self.relu(FC1))
        # res = self.lin2Inter(self.sigmoid(FC2))
        # return self.sigmoid(res)

def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    # log_softmax = nn.LogSoftmax().to("cuda:0" if torch.cuda.is_available() else "cpu")
    for v, q, a, idx, q_len in tq:
        var_params = {
            'requires_grad': False,
        }
        with torch.set_grad_enabled(train):
            v = Variable(v.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            q = Variable(q.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            a = Variable(a.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            q_len = Variable(q_len.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)

        out = net(v)
        # assert(False)
        # print(a.float().unsqueeze(-1))
        # assert(False)
        # nll = -log_softmax(out)
        # loss = ((out - a)**2).sum(dim=1).mean()
        
        # binary cross entropy
        # weights = -(a.data/2 - 1).unsqueeze(-1)
        # assert(False)
        loss = nn.BCELoss()(out, a.float().unsqueeze(-1))
        chosen = [1 if x > 0.5 else 0 for x in out.data.cpu()]
        acc = [1.0 if chosen[i] == a.data[i] else 0.0 for i in range(len(chosen))]

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
        else:
            # store information about evaluation of this minibatch
            answ.append(out.data.cpu())
            for a in acc:
                accs.append(a)
            idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.data.item())
        for a in acc:
            acc_tracker.append(a)
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
    # print(out.data)

    if not train:
        # answ = list(torch.cat(answ, dim=0))
        # accs = list(torch.cat(accs, dim=0))
        # idxs = list(torch.cat(idxs, dim=0))
        # print(answ, accs, idxs)
        return answ, accs, idxs


def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs_karl', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True, check_suitable=True)
    val_loader = data.get_loader(val=True, check_suitable=True)

    net = nn.DataParallel(QualityNet()).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        _ = run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'vocab': train_loader.dataset.vocab,
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
