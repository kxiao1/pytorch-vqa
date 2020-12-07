import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os.path
import math
import json


import config
import data
import utils
import config

# batch_size = 128 

class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2
        # print("embedding tokens:", embedding_tokens) = 2061 tokens in questions
        self.text = TextProcessor(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.5,
        )
        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=512,
            glimpses=2,
            drop=0.5,
        )
        # self.classifier = Classifier(
        #     in_features=glimpses * vision_features + question_features,
        #     mid_features=1024,
        #     out_features=config.max_answers,
        #     drop=0.5,
        # )

        self.classifier = Classifier(
            in_features=glimpses * vision_features + vision_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    # def forward(self, v, q, q_len):
    #     q = self.text(q, list(q_len.data))
    #     # print("after text layer", q.size()) = [128, 1024]
    #     v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
    #     # print("image vector size", v.size()) = [128, 2048, 14, 14]
    #     a = self.attention(v, q)
    #     # print("after attention layer", a.size()) = [128, 2, 14, 14], 2 glimpses
    #     v = apply_attention(v, a)
    #     # print("after applying attention", v.size()) = [128, 4096 + 1024]
    #     combined = torch.cat([v, q], dim=1)
    #     answer = self.classifier(combined)
    #     return answer

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        # print("after text layer", q.size()) = [128, 1024]
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        # print("image vector size", v.size()) = [128, 2048, 14, 14]
        a = self.attention(v, q)
        # print("after attention layer", a.size()) = [128, 2, 14, 14], 2 glimpses
        v_after_attention = apply_attention(v, a)
    
        temp = v.mean(dim = -1)
        v_squeezed = temp.mean(dim = -1)
        # print(v.size(), v_squeezed.size(), v_after_attention.size())
        combined = torch.cat([v_squeezed, v_after_attention], dim=1)
        # print("after concat", v.size()) = [128, 4096]
        answer = self.classifier(combined)
        return answer

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        # print(in_features, mid_features, out_features)
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


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


# Visualize feature maps
activation = {}

class Attention(nn.Module):
    # def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
    #     super(Attention, self).__init__()
    #     self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
    #     self.q_lin = nn.Linear(q_features, mid_features)
    #     self.x_conv = nn.Conv2d(mid_features, glimpses, 1) # 2 glimpses to look at

    #     self.drop = nn.Dropout(drop)
    #     self.relu = nn.ReLU(inplace=True)

    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        primary_glimpses = 8
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin1 = nn.Linear(q_features, mid_features)
        self.q_lin2 = nn.Linear(q_features, primary_glimpses)
        self.x_conv = nn.Conv2d(mid_features, primary_glimpses, 1) # 16 primary glimpses to look at
        self.y_conv= nn.Conv2d(primary_glimpses, glimpses, 1) # 2 final glimpses

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        
    # def forward(self, v, q):
    #     v = self.v_conv(self.drop(v)) # [128, 512, 14, 14]
    #     q = self.q_lin(self.drop(q)) #= [128, 512]
    #     q = tile_2d_over_nd(q, v)
    #     # print("after tiling", q.size()) = [128, 512, 14, 14]
    #     x = self.relu(v + q)
    #     x = self.x_conv(self.drop(x)) # [128, 2, 14, 14]
    #     return x

    def forward(self, v, q):
        v = self.v_conv(self.drop(v)) # [128, 512, 14, 14]

        # q = [128, 1024]
        q_lin1 = self.q_lin1(self.drop(q)) # = [128, 512]
        q_tile1 = tile_2d_over_nd(q_lin1, v)
        # print("after tiling", q.size()) = [128, 512, 14, 14]
        x = self.relu(v + q_tile1)
        x = self.x_conv(self.drop(x)) # [128, 8, 14, 14]

        q_lin2 = self.q_lin2(self.drop(q)) 
        q_tile2 = tile_2d_over_nd(q_lin2, x)
        y = self.relu(x + q_tile2)
        y = self.y_conv(self.drop(y))
        return y

def apply_attention(input, attention):
    """ Apply any number of attention maps over the input. 
    Softmax the attention layer, then take weighted average with image input.
    """
    # attention: [128, 2, 14, 14], 2 glimpses
    # input: [128, 2048, 14, 14]
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1) # [n, 1, c, s] = [128, 1, 2048, 196]
    attention = attention.view(n, glimpses, -1) # [128, 2, 196]
    attention = F.softmax(attention, dim=-1).unsqueeze(2) # [n, g, 1, s] = [128, 2, 1, 196]
    weighted = attention * input # [n, g, v, s] = [128, 2, 2048, 196]
    weighted_mean = weighted.sum(dim=-1) # [n, g, v] = [128, 2, 2048]
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2
    # print(n, c, spatial_size, feature_map.size()) = 128, 512, 2, [128, 512, 14, 14]
    # feature_vector.view(n, c, *([1] * spatial_size)).size() = [128, 512, 1, 1]
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map) #[128, 512, 14, 14]
    return tiled
def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0


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

    log_softmax = nn.LogSoftmax().to("cuda:0" if torch.cuda.is_available() else "cpu")
    for v, q, a, idx, q_len in tq:
        var_params = {
            'requires_grad': False,
        }
        with torch.set_grad_enabled(train):
            v = Variable(v.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            q = Variable(q.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            a = Variable(a.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)
            q_len = Variable(q_len.to("cuda:0" if torch.cuda.is_available() else "cpu"), **var_params)

        out = net(v, q, q_len)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1
            # fig, axarr = plt.subplots(act0.size(0))
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())
            acts = activation['attention'].squeeze()
            actList = [acts[:,x,:,:] for x in range(2)] 
            if epoch == config.epochs - 1:
                for num in range(2):
                    for i in range(actList[num].size(0)):
                        # print(act0[idx])
                        # axarr[idx].imshow(act0[idx])
                        # axarr[idx].set_title("dimension" + str(idx))
                        plt.imshow(actList[num][i].cpu())
                        plt.savefig("img_karl/img_" + str(idx[i].item())+ "_layer_" + str(num) + ".png")

        loss_tracker.append(loss.data.item())
        # acc_tracker.append(acc.mean())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        accs = list(torch.cat(accs, dim=0))
        idxs = list(torch.cat(idxs, dim=0))
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

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(Net(train_loader.dataset.num_tokens)).cuda()
    # net = Net(train_loader.dataset.num_tokens)
    print(net)
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for name, layer in net.named_modules():
        if name == "module.attention.y_conv":
            layer.register_forward_hook(get_activation('attention'))
    
    # net.attention.register_forward_hook(get_activation('attention'))

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
