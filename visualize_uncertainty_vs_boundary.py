"""
Qualitative A.10: overlay approximate defect footprints (from the CCNY slab
schematic, 10ft x 8ft = 120in x 96in) on DS2 May / DS2 June / DS4
total-uncertainty maps, and report a simple inside-vs-outside uncertainty
contrast.

Per-dataset orientation handling:
  - DS2-May  (31, 38): rows=Y=96in, cols=X=120in  (landscape)
  - DS2-June (19, 34): rows=Y=96in, cols=X=120in  (landscape, anisotropic)
  - DS4      (44, 34): rows=X=120in, cols=Y=96in  (slab rotated 90 deg)

Not pixel-accurate. Bounding boxes are hand-coded from the schematic and are
intended only to illustrate that high-uncertainty regions cluster around known
defect footprints, without claiming a precise GT mask.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from models.evidential_model import create_model
from utils.utils import (
    load_ccny_sep2022_data_into_torch_tensor,
    load_ccny_nov2023_data_into_torch_tensor2,
)


# Slab physical size (CCNY), origin top-left when viewed landscape.
SLAB_W_IN = 120.0   # 10 ft
SLAB_H_IN = 96.0    # 8 ft

# Approximate defect bounding boxes in inches in the landscape slab frame,
# origin top-left, (x0, y0, x1, y1). Read off the CCNY schematic; not
# pixel-accurate.
DEFECT_BOXES_IN = [
    ("Delam-1",    16.0,  9.0,  28.0, 21.0),
    ("Honey",      44.0, 11.0,  53.0, 16.5),
    ("Honey",      65.0, 11.0,  74.0, 16.5),
    ("Void",       86.0,  7.0,  92.0, 21.0),
    ("F-metal",    96.0,  9.0, 114.0, 21.0),
    ("Delam-D",    40.0, 20.0,  50.5, 30.5),
    ("Void",       66.0, 32.0,  75.0, 37.5),
    ("T-metal",    24.0, 75.0,  40.0, 87.0),
    ("Honey",      52.0, 80.0,  61.0, 86.0),
    ("Honey",      74.0, 79.0,  84.0, 86.0),
]

# Band around each defect (inches) considered boundary-adjacent.
BOUNDARY_BAND_IN = 4.0


def load_evidential(path, device):
    model = create_model().to(device)
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.eval()
    return model


def build_masks_landscape(shape_rc, boxes_in, band_in):
    """rows=Y=96in, cols=X=120in. Returns (defect_mask, band_mask)."""
    R, C = shape_rc
    py_per_in = R / SLAB_H_IN
    px_per_in = C / SLAB_W_IN
    defect = np.zeros((R, C), dtype=bool)
    band = np.zeros((R, C), dtype=bool)
    for _, x0, y0, x1, y1 in boxes_in:
        c0 = max(0, int(np.floor(x0 * px_per_in)))
        c1 = min(C, int(np.ceil(x1 * px_per_in)))
        r0 = max(0, int(np.floor(y0 * py_per_in)))
        r1 = min(R, int(np.ceil(y1 * py_per_in)))
        defect[r0:r1, c0:c1] = True
        bc0 = max(0, int(np.floor((x0 - band_in) * px_per_in)))
        bc1 = min(C, int(np.ceil((x1 + band_in) * px_per_in)))
        br0 = max(0, int(np.floor((y0 - band_in) * py_per_in)))
        br1 = min(R, int(np.ceil((y1 + band_in) * py_per_in)))
        band[br0:br1, bc0:bc1] = True
    band = band & ~defect
    return defect, band


def build_masks_portrait(shape_rc, boxes_in, band_in):
    """rows=X=120in, cols=Y=96in (slab rotated 90 deg CCW)."""
    R, C = shape_rc
    px_per_in_x = R / SLAB_W_IN   # rows index slab X
    px_per_in_y = C / SLAB_H_IN   # cols index slab Y
    defect = np.zeros((R, C), dtype=bool)
    band = np.zeros((R, C), dtype=bool)
    for _, x0, y0, x1, y1 in boxes_in:
        r0 = max(0, int(np.floor(x0 * px_per_in_x)))
        r1 = min(R, int(np.ceil(x1 * px_per_in_x)))
        c0 = max(0, int(np.floor(y0 * px_per_in_y)))
        c1 = min(C, int(np.ceil(y1 * px_per_in_y)))
        defect[r0:r1, c0:c1] = True
        br0 = max(0, int(np.floor((x0 - band_in) * px_per_in_x)))
        br1 = min(R, int(np.ceil((x1 + band_in) * px_per_in_x)))
        bc0 = max(0, int(np.floor((y0 - band_in) * px_per_in_y)))
        bc1 = min(C, int(np.ceil((y1 + band_in) * px_per_in_y)))
        band[br0:br1, bc0:bc1] = True
    band = band & ~defect
    return defect, band


def run_dataset(model, X, shape_rc, orientation, title, outfile, device):
    X = X.to(device)
    with torch.no_grad():
        _, _, _, total_unc, _, _ = model.predict_with_uncertainty(X)
    u = total_unc.squeeze().cpu().numpy()
    R, C = shape_rc
    if u.size != R * C:
        raise ValueError(f"Sample count {u.size} does not match {R}x{C}")
    u_map = u.reshape(R, C)

    if orientation == "landscape":
        defect_mask, band_mask = build_masks_landscape(shape_rc, DEFECT_BOXES_IN, BOUNDARY_BAND_IN)
        extent = [0, SLAB_W_IN, SLAB_H_IN, 0]   # x0, x1, y1, y0
        boxes_for_plot = [(x0, y0, x1 - x0, y1 - y0) for _, x0, y0, x1, y1 in DEFECT_BOXES_IN]
        xlabel = "Slab X (in, 120in long side)"
        ylabel = "Slab Y (in, 96in short side)"
    elif orientation == "portrait":
        defect_mask, band_mask = build_masks_portrait(shape_rc, DEFECT_BOXES_IN, BOUNDARY_BAND_IN)
        # plot axes: horizontal = slab Y (96in), vertical = slab X (120in, running downward)
        extent = [0, SLAB_H_IN, SLAB_W_IN, 0]
        boxes_for_plot = [(y0, x0, y1 - y0, x1 - x0) for _, x0, y0, x1, y1 in DEFECT_BOXES_IN]
        xlabel = "Slab Y (in, 96in short side)"
        ylabel = "Slab X (in, 120in long side)"
    else:
        raise ValueError(orientation)

    interior = u_map[defect_mask].mean() if defect_mask.any() else float("nan")
    boundary = u_map[band_mask].mean() if band_mask.any() else float("nan")
    outside = u_map[~(defect_mask | band_mask)].mean()

    print(f"\n{title}  shape={shape_rc}  orientation={orientation}  n={u.size}")
    print(f"  Mean uncertainty inside defect boxes     : {interior:.4f}")
    print(f"  Mean uncertainty in {BOUNDARY_BAND_IN:.0f} in boundary band  : {boundary:.4f}")
    print(f"  Mean uncertainty outside (background)    : {outside:.4f}")
    if not np.isnan(boundary):
        print(f"  (interior+boundary) / outside            : {(u_map[defect_mask|band_mask].mean()/outside):.3f}x")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        u_map,
        cmap="plasma",
        extent=extent,
        aspect="equal",
        interpolation="bilinear",
    )
    for x, y, w, h in boxes_for_plot:
        ax.add_patch(mpatches.Rectangle(
            (x, y), w, h,
            linewidth=1.6, edgecolor="#00ff7f", facecolor="none",
        ))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}: total uncertainty with approximate defect footprints.\n"
        f"Inside={interior:.3f}, boundary({BOUNDARY_BAND_IN:.0f}in)="
        f"{boundary:.3f}, outside={outside:.3f}"
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Total uncertainty")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {outfile}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_evidential("weights/evidential_transformer.pth", device)

    X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(device)
    X_nov = load_ccny_nov2023_data_into_torch_tensor2(device=device)

    run_dataset(
        model, X_may, (31, 38), "landscape",
        "DS2 May 2022",
        "figures/ds2_may_uncertainty_vs_boundary.pdf",
        device,
    )
    run_dataset(
        model, X_june, (19, 34), "landscape",
        "DS2 June 2022",
        "figures/ds2_june_uncertainty_vs_boundary.pdf",
        device,
    )
    run_dataset(
        model, X_nov, (44, 34), "portrait",
        "DS4 November 2023",
        "figures/ds4_nov2023_uncertainty_vs_boundary.pdf",
        device,
    )


if __name__ == "__main__":
    main()
