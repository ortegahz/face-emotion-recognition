import sys

import torch
import torch.nn as nn

sys.path.append('/tmp/pycharm_project_605/recognition/arcface_torch')
# sys.path.append('/media/manu/kingstop/workspace/insightface/recognition/arcface_torch')
from backbones import get_model


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.num_features = 512
        self.num_class = 8
        self.path_backbone_ckpt = '/media/manu-pc/tmp/wf42m_pfc02_8gpus_r50_bs1k/model.pt'

        self.backbone = get_model('r50', dropout=0.0, fp16=False, num_features=self.num_features).to(device)
        self.backbone.load_state_dict(torch.load(self.path_backbone_ckpt, map_location=device))
        self.classifier = nn.Linear(in_features=self.num_features, out_features=self.num_class).to(device)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
