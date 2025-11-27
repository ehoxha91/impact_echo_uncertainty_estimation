"""
Evidential Deep Learning Model for Impact Echo Signal Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveFeatureExtractor(nn.Module):
    """Multi-scale wave feature extraction with dilated convolutions"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[200, 100, 50, 25, 13, 7]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                         kernel_size=k, padding='same', dilation=1),
                nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        
    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return torch.cat(features, dim=1)


class EnhancedResidualBlock(nn.Module):
    """Residual block with SE attention and better gradient flow"""
    def __init__(self, in_channels, out_channels, seq_len, reduction=4):
        super().__init__()
        self.downsample = nn.AvgPool1d(2) if seq_len > 1 else nn.Identity()
        mid_channels = out_channels
        
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Squeeze-and-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        att = self.se(out)
        out = out * att
        
        out = out + identity
        out = self.activation(out)
        out = self.downsample(out)
        
        return out


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for wave signals"""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, d_model, max_len) * 0.02)
        
    def forward(self, x):
        return x + self.pe[:, :, :x.size(2)]


class ImprovedEvidentialIENet(nn.Module):
    """
    Enhanced Evidential IENet for 1D wave signal defect detection
    """
    def __init__(self, num_classes=2, input_length=200, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        # Multi-scale initial feature extraction
        self.wave_features = WaveFeatureExtractor(1, 36)
        
        # Progressive feature refinement with better gradient flow
        self.residual_1 = EnhancedResidualBlock(36, 64, input_length)
        self.residual_2 = EnhancedResidualBlock(64, 128, input_length // 2)
        self.residual_3 = EnhancedResidualBlock(128, 256, input_length // 4)
        self.residual_4 = EnhancedResidualBlock(256, 256, input_length // 8)
        
        # Positional encoding for transformer
        self.pos_encoding = PositionalEncoding(256)
        
        # Efficient transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global and local feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(4)
        
        feature_dim = 256 + 256 * 4
        
        # Evidence pathway with uncertainty decomposition
        self.evidence_pathway = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Separate evidence branches
        self.evidence_layer = nn.Linear(256, num_classes)
        self.uncertainty_layer = nn.Linear(256, num_classes)
        
        # Auxiliary defect characteristic predictor
        self.defect_features = nn.Linear(256, 16)
        
    def forward(self, x):
        # Multi-scale wave feature extraction
        x = self.wave_features(x)
        
        # Hierarchical feature extraction
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.residual_4(x)
        
        # Add positional encoding for transformer
        x = self.pos_encoding(x)
        
        # Transformer processing
        x_t = x.transpose(1, 2)
        x_t = self.transformer(x_t)
        x = x_t.transpose(1, 2)
        
        # Multi-level pooling
        global_features = self.global_pool(x).squeeze(-1)
        local_features = self.local_pool(x).flatten(1)
        
        # Combine global and local features
        combined = torch.cat([global_features, local_features], dim=1)
        
        # Evidence pathway
        features = self.evidence_pathway(combined)
        
        # Evidence outputs (Dirichlet parameters)
        evidence = F.softplus(self.evidence_layer(features)) + 1e-6
        uncertainty = F.softplus(self.uncertainty_layer(features)) + 1e-6
        
        # Defect characteristics
        defect_features = self.defect_features(features)
        
        return evidence, defect_features

    def predict_with_uncertainty(self, x):
        """Get predictions with evidential uncertainty measures"""
        with torch.no_grad():
            evidence, _ = self.forward(x)
            
            # Dirichlet parameters
            alphas = evidence + 1.0
            alpha_sum = torch.sum(alphas, dim=-1, keepdim=True)
            
            # Expected probabilities
            prob = alphas / alpha_sum
            
            # Uncertainty measures
            aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty = ImprovedEvidentialIENet.compute_uncertainties(alphas)
            
            # Confidence
            confidence = torch.max(prob, dim=-1)[0]
            
            return prob, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, confidence, alpha_sum

    @staticmethod
    def compute_uncertainties(alpha):
        S = torch.sum(alpha, dim=-1, keepdim=True)
        mean = alpha / S
        aleatoric = -torch.sum(mean * (torch.digamma(alpha + 1) - torch.digamma(S + 1)), dim=-1)
        total = -torch.sum(mean * torch.log(mean + 1e-10), dim=-1)
        epistemic = total - aleatoric
        return aleatoric, epistemic, total


def create_model(input_length=860, num_classes=2, dropout=0.1):
    """Factory function to create the model with proper initialization"""
    model = ImprovedEvidentialIENet(
        num_classes=num_classes,
        input_length=input_length,
        dropout=dropout
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model
