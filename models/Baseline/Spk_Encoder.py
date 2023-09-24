import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from .WavLM import *

import torch
import torch.nn as nn

class MHFA(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


class spk_extractor(nn.Module):
    def __init__(self,**kwargs):
        super(spk_extractor, self).__init__()
        # checkpoint = torch.load('/mnt/proj3/open-24-5/pengjy_new/WavLM/Pretrained_model/WavLM-Base+.pt')
        print("Pre-trained Model: {}".format(kwargs['pretrained_model_path']))
        checkpoint = torch.load(kwargs['pretrained_model_path'])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        self.backend = MHFA(head_nb=32)


    def forward(self,wav_and_flag):
        
        x = wav_and_flag[0]

        cnn_outs, layer_results =  self.model.extract_features(x, output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)

        out = self.backend(x)
        return out

    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


def MainModel(**kwargs):
    model = spk_extractor(**kwargs)
    return model