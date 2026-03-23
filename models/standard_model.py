"""
Standard (Non-Evidential) IENet for baseline UQ comparisons.
Same backbone as ImprovedEvidentialIENet but with softmax classification head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.evidential_model import WaveFeatureExtractor, EnhancedResidualBlock, PositionalEncoding


class StandardIENet(nn.Module):
    """
    Standard IENet with softmax output for use with MC Dropout and Deep Ensembles.
    Architecture identical to ImprovedEvidentialIENet except the output head.
    """
    def __init__(self, num_classes=2, input_length=860, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        # Identical backbone to evidential model
        self.wave_features = WaveFeatureExtractor(1, 36)

        self.residual_1 = EnhancedResidualBlock(36, 64, input_length)
        self.residual_2 = EnhancedResidualBlock(64, 128, input_length // 2)
        self.residual_3 = EnhancedResidualBlock(128, 256, input_length // 4)
        self.residual_4 = EnhancedResidualBlock(256, 256, input_length // 8)

        self.pos_encoding = PositionalEncoding(256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(4)

        feature_dim = 256 + 256 * 4

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.wave_features(x)
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.residual_4(x)
        x = self.pos_encoding(x)
        x_t = x.transpose(1, 2)
        x_t = self.transformer(x_t)
        x = x_t.transpose(1, 2)
        global_features = self.global_pool(x).squeeze(-1)
        local_features = self.local_pool(x).flatten(1)
        combined = torch.cat([global_features, local_features], dim=1)
        logits = self.classifier(combined)
        return logits


def create_standard_model(input_length=860, num_classes=2, dropout=0.1):
    """Factory function with proper weight initialization."""
    model = StandardIENet(
        num_classes=num_classes,
        input_length=input_length,
        dropout=dropout
    )
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return model
