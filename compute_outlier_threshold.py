"""
Compute high-uncertainty fractions under standard outlier criteria on DS1 total
uncertainty (Evidential IENet, single forward pass).

Reports:
  - mu + 1.5*sigma  (previous, non-standard in the paper)
  - mu + 3*sigma    (standard Gaussian outlier rule)
  - Q3 + 1.5*IQR    (Tukey, standard)
  - top 10% percentile (for reference)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from dataloaders.dataloader import ImpactEchoDatasetClassifier
from models.evidential_model import create_model


def load_evidential_model(path, device):
    model = create_model().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    return model


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_evidential_model('weights/evidential_transformer.pth', device)

    ds1 = ImpactEchoDatasetClassifier(['data/X_test_860.npy'],
                                       y_path=['data/y_test.npy'], array_size=860)
    X = torch.stack([ds1[i][0] for i in range(len(ds1))]).unsqueeze(1).float().to(device)

    with torch.no_grad():
        _, _, _, total_unc, _, _ = model.predict_with_uncertainty(X)
    u = total_unc.squeeze().cpu().numpy()

    mu, sigma = u.mean(), u.std()
    q1, q3 = np.percentile(u, [25, 75])
    iqr = q3 - q1

    t_15s = mu + 1.5 * sigma
    t_3s  = mu + 3.0 * sigma
    t_iqr = q3 + 1.5 * iqr
    t_p90 = np.percentile(u, 90)

    n = len(u)
    p15 = (u > t_15s).sum() / n * 100
    p3  = (u > t_3s).sum()  / n * 100
    pIQ = (u > t_iqr).sum() / n * 100
    p90 = (u > t_p90).sum() / n * 100

    print(f"DS1 total-uncertainty  n={n}")
    print(f"  mu={mu:.4f}  sigma={sigma:.4f}  Q1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
    print()
    print(f"  mu + 1.5*sigma = {t_15s:.4f}   --> {p15:5.2f}% above")
    print(f"  mu + 3.0*sigma = {t_3s:.4f}    --> {p3:5.2f}% above")
    print(f"  Q3 + 1.5*IQR   = {t_iqr:.4f}   --> {pIQ:5.2f}% above (Tukey)")
    print(f"  90th percentile= {t_p90:.4f}   --> {p90:5.2f}% above")


if __name__ == '__main__':
    main()
