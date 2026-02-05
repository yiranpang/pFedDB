import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as func
from collections import OrderedDict

#####################################################
# DigitModel
#####################################################
class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)
        return x


class DigitModel_DB(nn.Module):
    """
    Dual-branch version of the simple Digits backbone.
    - Shared branch is aggregated across clients.
    - Local branch is kept private to each client.
    The two 2048-D feature vectors are fused additively before final
    classification, matching the scheme in AlexNet_DB.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ---------- convolutional backbones ----------
        conv_layers = [
            ('conv1', nn.Conv2d(3,  64, 5, 1, 2)),
            ('bn1',   nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2)),                # 28×28 → 14×14

            ('conv2', nn.Conv2d(64, 64, 5, 1, 2)),
            ('bn2',   nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2)),                # 14×14 → 7×7

            ('conv3', nn.Conv2d(64, 128, 5, 1, 2)),
            ('bn3',   nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        # identical shared / local copies
        self.shared_conv = nn.Sequential(OrderedDict(conv_layers))
        self.local_conv  = nn.Sequential(OrderedDict(conv_layers))

        # ---------- 2048-D projection heads ----------
        self.shared_proj = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 7 * 7, 2048)),
            ('bn4', nn.BatchNorm1d(2048)),
            ('relu4', nn.ReLU(inplace=True)),
        ]))

        self.local_proj = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 7 * 7, 2048)),
            ('bn4', nn.BatchNorm1d(2048)),
            ('relu4', nn.ReLU(inplace=True)),
        ]))

        # ---------- final classifier (shared across clients) ----------
        self.final_classifier = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(2048, 512)),
            ('bn5', nn.BatchNorm1d(512)),
            ('relu5', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(512, num_classes)),
        ]))

        self._init_weights()

    # -----------------------------------------------------------------
    def forward(self, x):
        # conv feature maps
        shared_feat = self.shared_conv(x)          # B × 128 × 7 × 7
        local_feat  = self.local_conv(x)

        # flatten
        shared_flat = shared_feat.view(x.size(0), -1)
        local_flat  = local_feat.view(x.size(0), -1)

        # 2048-D projections
        shared_vec = self.shared_proj(shared_flat)  # B × 2048
        local_vec  = self.local_proj(local_flat)    # B × 2048

        # fuse & classify
        fused_vec  = shared_vec + local_vec
        logits     = self.final_classifier(fused_vec)

        return logits


    # -----------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#####################################################
# AlexNet
#####################################################
class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
            
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet_DB(nn.Module):
    """
    AlexNet with Dual Branch for Federated Learning.
    Input size: [batch_size, 3, 28, 28]
    """
    def __init__(self, num_classes=10):
        super(AlexNet_DB, self).__init__()
        
        # shared branch
        self.shared_conv = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        
        # local branch
        self.local_conv = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.shared_classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
            ])
        )

        self.local_classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
            ])
        )

        self.final_classifier = nn.Sequential(
            OrderedDict([

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )
        
        self._init_weights()

    
    def forward(self, x):
        shared_feat = self.shared_conv(x)  # [batch_size, 256, H, W]
        local_feat  = self.local_conv(x)   # [batch_size, 256, H, W]
        
        pooled_feat = self.avgpool(shared_feat)  # [batch_size, 256, 6, 6]
        shared_flat = torch.flatten(pooled_feat, 1)  # [batch_size, 256*6*6]
        shared_feat = self.shared_classifier(shared_flat)

        pooled_feat = self.avgpool(local_feat)  # [batch_size, 256, 6, 6]
        local_flat = torch.flatten(pooled_feat, 1)
        local_feat = self.local_classifier(local_flat)

        fused_feat = shared_feat + local_feat  # adding shared and local features
        logits = self.final_classifier(fused_feat)
        
        return logits, shared_feat

    def global_conv_forward(self, x):
        return self.shared_conv(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#####################################################
# DenseNet
#####################################################
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
    
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module(f'denselayer{i + 1}', layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2,
                 in_channels=1):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def make_dense_features_split(in_channels=1, num_init_features=64, growth_rate=32,
                              block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0,
                              branch_num_blocks=3):
    assert 1 <= branch_num_blocks < len(block_config), "branch_num_blocks 必须小于总 block 数且至少为1"

    features_initial = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(in_channels, num_init_features,
                            kernel_size=7, stride=2, padding=3, bias=False)),
        ('norm0', nn.BatchNorm2d(num_init_features)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))
    
    branch_layers = nn.Sequential()
    num_features = num_init_features
    for i in range(branch_num_blocks):
        num_layers = block_config[i]
        block = _DenseBlock(num_layers=num_layers,
                            num_input_features=num_features,
                            bn_size=bn_size,
                            growth_rate=growth_rate,
                            drop_rate=drop_rate)
        branch_layers.add_module(f'denseblock{i+1}', block)
        num_features = num_features + num_layers * growth_rate

        if i != branch_num_blocks - 1:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            branch_layers.add_module(f'transition{i+1}', trans)
            num_features = num_features // 2

    remaining_layers = nn.Sequential()
    for i in range(branch_num_blocks, len(block_config)):
        num_layers = block_config[i]
        block = _DenseBlock(num_layers=num_layers,
                            num_input_features=num_features,
                            bn_size=bn_size,
                            growth_rate=growth_rate,
                            drop_rate=drop_rate)
        remaining_layers.add_module(f'denseblock{i+1}', block)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            remaining_layers.add_module(f'transition{i+1}', trans)
            num_features = num_features // 2

    remaining_layers.add_module('norm_final', nn.BatchNorm2d(num_features))

    return features_initial, branch_layers, remaining_layers, num_features


def make_dense_features(
    in_channels: int = 1,
    num_init_features: int = 64,
    growth_rate: int = 32,
    block_config=(6, 12, 24, 16),
    bn_size: int = 4,
    drop_rate: float = 0,
):
    """Build a standard DenseNet feature extractor.

    Returns:
      (features, out_channels)
    where `features` maps input images to the final DenseNet feature map and
    `out_channels` is the number of channels in that feature map.
    """
    features = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(in_channels, num_init_features,
                            kernel_size=7, stride=2, padding=3, bias=False)),
        ('norm0', nn.BatchNorm2d(num_init_features)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
        )
        features.add_module(f'denseblock{i + 1}', block)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            features.add_module(f'transition{i + 1}', trans)
            num_features = num_features // 2

    features.add_module('norm5', nn.BatchNorm2d(num_features))
    return features, num_features

class DenseNet_DB(nn.Module):
    def __init__(self, num_classes=2, in_channels=1,
                 num_init_features=64, growth_rate=32,
                 block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0,
                 rep_dim=1024, branch_num_blocks=3):
        super(DenseNet_DB, self).__init__()
        
        (features_initial,
         branch_layers,
         self.remaining_layers,
         remaining_channels) = make_dense_features_split(in_channels=in_channels,
                                                         num_init_features=num_init_features,
                                                         growth_rate=growth_rate,
                                                         block_config=block_config,
                                                         bn_size=bn_size,
                                                         drop_rate=drop_rate,
                                                         branch_num_blocks=branch_num_blocks)

        self.shared_init_features = copy.deepcopy(features_initial)
        self.local_init_features = features_initial
        
        self.shared_branch = copy.deepcopy(branch_layers)
        self.local_branch = branch_layers
        
        self.classifier = nn.Linear(remaining_channels, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self._init_weights()
        
    def forward(self, x):
        x_share = self.shared_init_features(x)
        x_local = self.local_init_features(x)
        
        feat_shared = self.shared_branch(x_share)
        feat_local = self.local_branch(x_local)
        
        fused_features = feat_shared + feat_local
        
        out = self.remaining_layers(fused_features)
        
        out = self.avgpool(out)        # [B, C, 1, 1]
        out = torch.flatten(out, 1)    # [B, C]
        logits = self.classifier(out)
        
        return logits
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# -------------------------------
# 构造 DenseNet 特征提取部分的工具函数
# -------------------------------
def make_dense_features(in_channels=1, num_init_features=64, growth_rate=32,
                        block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0):
    features = nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(in_channels, num_init_features,
                            kernel_size=7, stride=2, padding=3, bias=False)),
        ('norm0', nn.BatchNorm2d(num_init_features)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        block = _DenseBlock(num_layers=num_layers,
                            num_input_features=num_features,
                            bn_size=bn_size,
                            growth_rate=growth_rate,
                            drop_rate=drop_rate)
        features.add_module(f'denseblock{i + 1}', block)
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            features.add_module(f'transition{i + 1}', trans)
            num_features = num_features // 2

    features.add_module('norm_final', nn.BatchNorm2d(num_features))
    return features, num_features  # 返回特征提取部分和最终通道数


# -------------------------------
# Dual-branch DenseNet model for federated learning scenarios
# -------------------------------
class PLENet_DenseNet_ShareCNN(nn.Module):
    """
    Uses DenseNet as the backbone with two branches:
      - shared_features: global shared branch, uploaded to the server in federated learning
      - local_features: local personalized branch, trained and updated only on the client
    Both branches extract features, which are then adaptive average pooled,
    passed through their respective branch classifiers to map to intermediate representations, and then simply fused (added),
    finally passed through the final classifier to obtain class predictions.
    """
    def __init__(self, num_classes=2, in_channels=1,
                 num_init_features=64, growth_rate=32,
                 block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0,
                 rep_dim=1024):
        super(PLENet_DenseNet_ShareCNN, self).__init__()
        
        # Construct shared branch feature extraction module
        self.shared_features, shared_out_channels = make_dense_features(
            in_channels=in_channels,
            num_init_features=num_init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=bn_size,
            drop_rate=drop_rate
        )
        
        # Construct local branch feature extraction module (same structure as shared branch, can be made more lightweight)
        self.local_features, local_out_channels = make_dense_features(
            in_channels=in_channels,
            num_init_features=num_init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=bn_size,
            drop_rate=drop_rate
        )
        # Note: The output channels of the two branches above should be the same (usually the final number of channels of DenseNet)
        assert shared_out_channels == local_out_channels, "The output channels of the shared and local branches should be the same!"
        self.feature_channels = shared_out_channels
        
        # Classification layer
        self.classifier = nn.Linear(self.feature_channels, num_classes)
        
        # Define adaptive average pooling layer (converts convolutional features to fixed size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self._init_weights()
        
    def forward(self, x):
        # 1. 分支特征提取
        shared_feat = self.shared_features(x)
        local_feat = self.local_features(x)
        
        # 2. Global average pooling and flattening
        shared_feat = self.avgpool(shared_feat)  # [B, C, 1, 1]
        shared_feat = torch.flatten(shared_feat, 1)  # [B, C]
        
        local_feat = self.avgpool(local_feat)
        local_feat = torch.flatten(local_feat, 1)
              
        # 4. Feature fusion (simple addition)
        fused_rep = shared_feat + local_feat
        
        # 5. Final classification
        logits = self.classifier(fused_rep)
        
        return logits  # Also return logits and intermediate representation of shared branch

    def _init_weights(self):
        # Initialize weights for convolutional, linear, and BatchNorm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PLENetReluBeforePoolWrapper(nn.Module):
    """Apply relu to DenseNet feature maps before avgpool.

    This wrapper exists to reproduce historical evaluation behavior used in some
    checkpoints.
    """

    def __init__(self, base: PLENet_DenseNet_ShareCNN):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.base

        out_shared = m.shared_features(x)
        out_local = m.local_features(x)

        out_shared = torch.relu(out_shared)
        out_local = torch.relu(out_local)

        out_shared = m.avgpool(out_shared)
        out_local = m.avgpool(out_local)

        out_shared = torch.flatten(out_shared, 1)
        out_local = torch.flatten(out_local, 1)

        fused = out_shared + out_local
        return m.classifier(fused)
