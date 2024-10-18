import torch
from transformers import GPT2Config, GPT2Model
import torch.nn as nn

def get_attention_mask(context_len):
    mask = torch.ones(context_len, context_len).tril()
    for i in range(context_len):
        if i % 2 == 1:
            mask[i, i-1] = 0
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


class GPT2Transformer(nn.Module):
    def __init__(self, n_inputs, n_outputs, args):
        super(GPT2Transformer, self).__init__()
        print('=> Initializing a GPT2 Transformer...')
        
        self.context_len = args.n_context
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.config = GPT2Config(
            n_positions= 2 * self.context_len,                 
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self._read_in = nn.Linear(n_inputs, args.n_embd)
        self._backbone = GPT2Model(self.config)
        self._read_out = nn.Linear(args.n_embd, n_outputs)
        self.attn_mask = get_attention_mask(2*self.context_len).cuda()
    
    def forward(self, x):
        bsz, d = x.shape
        n_seq = int(bsz / (2 * self.context_len))
        x = x.view(n_seq, 2*self.context_len, d)
        x = self._read_in(x)
        x = self._backbone(inputs_embeds=x, attention_mask=self.attn_mask).last_hidden_state
        x = self._read_out(x)
        x = x.view(bsz, -1)
        return x


class CIFAR_Network(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, projector_type='MLP', n_classes=10, proj_hidden_dim=2048, separate_proj=False, predict_idx=False, args=None):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.linear = nn.Linear(self.feature_dim, n_classes)
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, proj_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(proj_hidden_dim, projection_dim))

    def forward(self, x, eval_only=False):
        feature = self.enc(x)
        if not eval_only:
            projection = self.projector(feature)
        else:
            projection = None
        logits = self.linear(feature.detach()) # remove 'detach' when running verification.py
        return feature, projection, logits
        
        # return feature, projection


class ImageNet_Network(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, projector_type='MLP', n_classes=200, proj_hidden_dim=2048, separate_proj=False, predict_idx=False, args=None):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.enc.fc = nn.Identity()  # remove final fully connected layer.
        self.feature_dim = 512

        # projector
        sizes = [self.feature_dim, proj_hidden_dim, projection_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))

        self.projector = nn.Sequential(*layers)

        self.linear = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x, eval_only=False):
        feature = self.enc(x)
        if not eval_only:
            projection = self.projector(feature)
        else:
            projection = None
        logits = self.linear(feature.detach()) # remove 'detach' when running verification.py
        return feature, projection, logits
    
