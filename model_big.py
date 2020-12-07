import sys
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


import config
from model import Classifier, TextProcessor, Attention, apply_attention, tile_2d_over_nd
from train import run, total_iterations, update_learning_rate
import model_colors
import model_is_color
import model_suitable
import data
import utils

class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

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
        self.classifier = Classifier(
            in_features=glimpses * vision_features + question_features + 16 + 1 + 1,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.is_color_classifier = nn.DataParallel(model_is_color.ColorNet(embedding_tokens)).cuda()
        log = torch.load('logs_is_color/final.pth', map_location=torch.device('cpu'))
        self.is_color_classifier.load_state_dict(log['weights'])
        self.is_color_classifier_weights = log['weights']
        self.is_color_classifier.eval()

        self.color_classifier = nn.DataParallel(model_colors.color_net).cuda()
        log = torch.load('logs_color/color_with_0.01_weight_decay.pth', map_location=torch.device('cpu'))
        self.color_classifier.load_state_dict(log['weights'])
        self.color_classifier_weights = log['weights']
        self.color_classifier.eval()

        self.suitable_classifier = nn.DataParallel(model_suitable.QualityNet()).cuda()
        log = torch.load('logs_karl/suitable_comp_3.pth', map_location=torch.device('cpu'))
        self.suitable_classifier.load_state_dict(log['weights'])
        self.suitable_classifier_weights = log['weights']
        self.suitable_classifier.eval()

    def forward(self, v, q, q_len):
        with torch.set_grad_enabled(False):
            self.is_color_classifier.eval()
            is_color = self.is_color_classifier(q, q_len)
            self.color_classifier.eval()
            color = self.color_classifier(v)
            self.suitable_classifier.eval()
            suitable  = self.suitable_classifier(v)

        q = self.text(q, list(q_len.data))
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q, is_color, color, suitable], dim=1)
        answer = self.classifier(combined)
        # for k, v in self.is_color_classifier.state_dict().items():
        #     assert torch.all(torch.eq(v.cpu(), self.is_color_classifier_weights[k].cpu()))
        # for k, v in self.color_classifier.state_dict().items():
        #     assert torch.all(torch.eq(v.cpu(), self.color_classifier_weights[k].cpu()))
        # for k, v in self.suitable_classifier.state_dict().items():
        #     assert torch.all(torch.eq(v.cpu(), self.suitable_classifier_weights[k].cpu()))
        return answer

def main():
    print("running on", "cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs_big', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(Net(train_loader.dataset.num_tokens)).cuda()
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
