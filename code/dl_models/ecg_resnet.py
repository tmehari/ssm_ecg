import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
from .xresnet1d import xresnet1d50, xresnet1d101
from .basic_conv1d import bn_drop_lin

class ECGResNet(nn.Module):

    def __init__(self, base_model, out_dim, widen=1.0, big_input=False, use_meta_information_in_head=False):
        super(ECGResNet, self).__init__()
        self.resnet_dict = { "xresnet1d50": xresnet1d50(widen=widen),
                            "xresnet1d101": xresnet1d101(widen=widen)}

        resnet = self._get_basemodel(base_model)
    
        list_of_modules = list(resnet.children())
        
        self.features = nn.Sequential(*list_of_modules[:-1], list_of_modules[-1][0])
        num_ftrs = resnet[-1][-1].in_features
        if big_input:
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=25, stride=10, padding=10)
        else:
            resnet[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(p=0.5)
        if use_meta_information_in_head:
            self.l1 = nn.Linear(num_ftrs+64, out_dim)
            meta_modules = bn_drop_lin(7, 64, bn=False,actn=nn.ReLU()) +\
            bn_drop_lin(64, 64, bn=True, p=0.5, actn=nn.ReLU()) + bn_drop_lin(64, 64, bn=True, p=0.5, actn=nn.ReLU())
            self.meta_head = nn.Sequential(*meta_modules)
        else:
            self.l1 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.bn(h)
        x = self.drop(x)
        x = self.l1(x)
        return x
    
    def forward_with_meta(self, x, meta_feats):
        h = self.features(x)
        h = h.squeeze()
        x = self.bn(h)
        x = self.drop(x)
        meta_feats = self.meta_head(meta_feats)
        x = torch.cat([x, meta_feats], axis=1)
        x = self.l1(x)
        return x
