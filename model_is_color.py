# determine whether the question is asking about color
# initially, we use the text only.
# run the file to execute training (uses parameters in config.py)

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch import log
from tqdm import tqdm

import data
import config
import utils

class ColorNet(nn.Module):
    def __init__(self, embedding_tokens):
        super(ColorNet, self).__init__()
        self.question_features = 1024
        self.internal_features = 128
        vision_features = config.output_features

        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=self.question_features,
            drop=0.5,
        )

        self.hidden = nn.Linear(self.question_features, self.internal_features)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.05)

        # size of q: [batch size, question features]
        self.linear = nn.Linear(self.internal_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, q_len):
        q = self.text(q, list(q_len.data))
        h = self.drop(self.relu(self.hidden(q)))
        answer = self.linear(h)
        return self.sigmoid(answer)


class TextProcessor(nn.Module):
    def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
        super(TextProcessor, self).__init__()
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1)
        self.features = lstm_features

        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        return c.squeeze(0)

def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

total_iterations = 0

def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
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

    answ = []
    for v, q, c, idx, q_len in tq:

        var_params = {
            'requires_grad': False,
        }

        with torch.set_grad_enabled(train):
            q = Variable(q.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            q_len = Variable(q_len.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            c = Variable(c.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)

        # c = 1 if it's a color, 0 otherwise

        # print("q", q)
        # print("idx", idx)
        # print("q_len", q_len)
        # print("c", c)

        out = net(q, q_len)

        # print(c.size())
        # binary cross entropy
        loss = nn.BCELoss()(out, c.float().unsqueeze(-1))

        chosen = [1 if x > 0.5 else 0 for x in out.data]
        acc = [1.0 if chosen[i] == c.data[i] else 0.0 for i in range(len(chosen))]

        # print(out)

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iterations += 1

        else:
            # TODO fix this
            # _, answer = out.data.cpu().max(dim=1)
            # answ.append(out.data.cpu())

            for a in out.data.cpu():
                answ.append(a.item())

            for a in acc:
                accs.append(a)
            idxs.append(idx.view(-1).clone())
        
        loss_tracker.append(loss.data.item())
        # acc_tracker.append(acc.mean())
        for a in acc:
            acc_tracker.append(a)
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
    # print(out.data)
    
    if not train:
        # answ = list(torch.cat(answ, dim=0))
        # answ = []
        # accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
        return answ, accs, idxs

def main():
    from datetime import datetime
    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs_is_color', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    train_loader = data.get_loader(train=True, check_pertains_to_color=True)
    val_loader = data.get_loader(val=True, check_pertains_to_color=True)
    net = nn.DataParallel(ColorNet(train_loader.dataset.num_tokens)).cuda()
    tracker = utils.Tracker()

    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad], weight_decay=0.01)
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

if __name__ == "__main__":
    main()