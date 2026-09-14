 
import os
import re
import numpy as np
import pydicom
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
 
ROOT = r"C:\Users\XYZ\Desktop\Principi dubokog ucenja\Projekat\INBreast\INBreast"
DICOM_DIR = os.path.join(ROOT, "AllDICOMs")
MASK_DIR = os.path.join(ROOT, "extras", "MassSegmentationMasks")
 
def build_index(folder, pattern, ext):
    index = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(ext):
            continue
        m = re.match(pattern, fname)
        if m:
            index[m.group(1)] = os.path.join(folder, fname)
    return index
 
def load_mask_bool(path):
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127
 
def check_pair(dcm_path, mask_path):
    ds = pydicom.dcmread(dcm_path)
    image = ds.pixel_array.astype(np.float32)
    mask = load_mask_bool(mask_path)
 
    if image.shape != mask.shape:
        return {"ok": False, "reason": f"dimenzije se NE poklapaju: slika {image.shape} vs maska {mask.shape}"}
 
    if mask.sum() == 0:
        return {"ok": False, "reason": "maska je prazna (sve nule)"}
 
    inside_mean = float(image[mask].mean())
 
    dilated = ndimage.binary_dilation(mask, iterations=15)
 
    ring = dilated & (~mask)
    ring_mean = float(image[ring].mean()) if ring.sum() > 0 else float("nan")
 
    background_frac = float((image[mask] < 5).mean())
 
    return {
        "ok": True,
        "inside_mean": inside_mean,
        "ring_mean": ring_mean,
        "background_frac": background_frac,
        "mask_area_px": int(mask.sum()),
    }
 
def main():
    dicom_index = build_index(DICOM_DIR, r"(\d+)_", ".dcm")
    mask_index = build_index(MASK_DIR, r"(\d+)_mask", ".png")
 
    common_ids = sorted(set(dicom_index) & set(mask_index))
    print(f"Proveravam {len(common_ids)} parova slika/maska...\n")
 
    suspicious = []
    results = []
    for img_id in common_ids:
        res = check_pair(dicom_index[img_id], mask_index[img_id])
        res["id"] = img_id
        results.append(res)
 
        if not res["ok"]:
            suspicious.append((img_id, res["reason"]))
            continue
 
        if res["background_frac"] > 0.15 or res["inside_mean"] <= res["ring_mean"]:
            suspicious.append((
                img_id,
                f"bg_frac={res['background_frac']:.2f}, "
                f"inside_mean={res['inside_mean']:.1f}, ring_mean={res['ring_mean']:.1f}",
            ))
 
    ok_count = sum(1 for r in results if r["ok"])
    print(f"Uspesno ucitano i proverljivo: {ok_count}/{len(results)}")
    print(f"Broj SUMNJIVIH slucajeva: {len(suspicious)}\n")
    if suspicious:
        print("Lista sumnjivih ID-jeva (proveri ih rucno):")
        for img_id, reason in suspicious:
            print(f"  ID {img_id}: {reason}")
    else:
        print("Nijedan slucaj nije oznacen kao sumnjiv - dobar znak!")
 
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
 
    for i, img_id in enumerate(common_ids[:6]):
        ds = pydicom.dcmread(dicom_index[img_id])
        image = ds.pixel_array.astype(np.float32)
        mask = load_mask_bool(mask_index[img_id])
 
        ys, xs = np.where(mask)
 
        pad = 80
        y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad)
 
        crop_img = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
 
        axes[i].imshow(crop_img, cmap="gray")
        axes[i].contour(crop_mask, levels=[0.5], colors="red", linewidths=2)
        axes[i].set_title(f"ID {img_id}")
        axes[i].axis("off")
 
    plt.tight_layout()
    plt.savefig("provera_bbox_zoom.png", dpi=150)
    print("\nSacuvano: provera_bbox_zoom.png")
 
if __name__ == "__main__":
    main()
