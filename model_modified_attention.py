import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

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
    #     v = self.v_conv(self.drop(v)) # [128, 2048, 14, 14]
    #     q = self.q_lin(self.drop(q)) #= [128, 512]
    #     q = tile_2d_over_nd(q, v)
    #     # print("after tiling", q.size()) = [128, 512, 14, 14]
    #     x = self.relu(v + q)
    #     x = self.x_conv(self.drop(x)) # [128, 2, 14, 14]
    #     return x

    def forward(self, v, q):
        v = self.v_conv(self.drop(v)) # [128, 2048, 14, 14]

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
