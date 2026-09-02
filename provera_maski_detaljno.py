"""
provera_maski_detaljno.py  (komentarisana verzija - vidi objasnjenje u chatu)
 
Cilj: za SVIH 107 parova (DICOM slika, Mass maska) automatski proveriti da li je
maska na pravom mestu - bez rucnog gledanja, jer to nije skalabilno/pouzdano.
"""
 
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
    """Pretvara folder pun fajlova u RECNIK: {id_slike: puna_putanja}.
 
    Ovo pravimo posebnom funkcijom (a ne inline kodom) jer je ISTA logika
    potrebna i za DICOM folder i za mask folder - samo se regex razlikuje.
    Umesto da kopiramo iste 5 linije dvaput, napisemo ih jednom i pozovemo
    dvaput sa razlicitim argumentima (DRY princip - "Don't Repeat Yourself").
    """
    index = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(ext):
            continue
        m = re.match(pattern, fname)
        if m:
            index[m.group(1)] = os.path.join(folder, fname)
    return index
 
 
def load_mask_bool(path):
    """Ucitaj PNG masku i vrati je kao 2D niz TRUE/FALSE vrednosti.
 
    Zasto konvertujemo u bool, a ne ostavljamo kao 0/255:
    Kasnije u kodu pisemo stvari tipa `image[mask]` - to je NAPREDNA
    numpy tehnika zvana "boolean indexing/masking". Kada mask sadrzi
    True/False vrednosti, `image[mask]` vraca SAMO one piksele slike
    gde je mask==True, spljostene u 1D niz. To nam omogucava da
    izracunamo npr. prosek SAMO unutar maske, bez petlje (for) kroz
    svaki piksel rucno - mnogo brze i kod je citljiviji.
 
    Zasto prag >127 a ne >0: maska ima vrednosti tacno 0 i 255 (videli
    smo to u prethodnom izlazu), pa je 127 "sredina" - siguran prag koji
    ce ispravno razdvojiti 0 od 255 cak i da ima blagog zamucenja na
    ivicama (kompresija slike ponekad malo "zamuti" ostre ivice).
    """
    arr = np.array(Image.open(path).convert("L")) #konvertuje sliku u "L"luminance (grayscale) mod, i pretvara u numpy niz
    return arr > 127
 
 
def check_pair(dcm_path, mask_path):
    """Ovo je SRZ citavog algoritma - vraca statistiku za JEDAN par (slika, maska).
 
    Logika (ljudskim jezikom): masa na mamogramu je gusce tkivo, sto znaci
    da je SVETLIJA (visi piksel intenzitet) od tkiva oko sebe. Ako je maska
    na pravom mestu, ocekujemo da vazi:
 
        prosecan intenzitet UNUTAR maske  >  prosecan intenzitet OKO maske
 
    Ako to ne vazi (unutrasnjost je tamnija ili ista kao okolina), nesto je
    sumnjivo - ili je maska pogresno pozicionirana, ili je slika neuobicajena.
    """
    ds = pydicom.dcmread(dcm_path)
    image = ds.pixel_array.astype(np.float32) #float zbog preciznijih racunanja, da ne bi bilo "prelivanja" (overflow) kod sabiranja i deljenja
    mask = load_mask_bool(mask_path)
 
    # --- Provera 1: da li se dimenzije uopste poklapaju ---
    # Ako slika i maska nisu iste velicine, `image[mask]` ce ili baciti
    # gresku ili (gore) tiho dati pogresan rezultat. Zato ovo proveravamo
    # PRE nego sto pokusamo bilo sta drugo sa njima.
    if image.shape != mask.shape:
        return {"ok": False, "reason": f"dimenzije se NE poklapaju: slika {image.shape} vs maska {mask.shape}"}
 
    if mask.sum() == 0:
        # mask.sum() sabira sve True vrednosti (True se racuna kao 1) -
        # ako je 0, maska je prazna (nema bele oblasti uopste)
        return {"ok": False, "reason": "maska je prazna (sve nule)"}
 
    # --- Prosek piksela UNUTAR maske ---
    # image[mask] uzima samo piksele gde je mask==True (boolean indexing,
    # objasnjeno gore) i vraca ih kao ravan (1D) niz. .mean() na tome
    # racuna prosek TIH piksela - dakle prosecnu "svetlinu" mase.
    inside_mean = float(image[mask].mean())
 
    # --- Prosek piksela u "PRSTENU" OKO maske ---
    # Ovde je kljucni trik: ne poredimo masku sa CELOM slikom (to bi bilo
    # nefer - cela slika ukljucuje i crnu pozadinu i svetlu kozu, sto bi
    # iskrivilo prosek), nego SAMO sa tkivom neposredno OKO mase.
    #
    # ndimage.binary_dilation "naduva" (prosiri) masku za 15 piksela u
    # svim pravcima - zamisli da originalnu belu mrlju obojis "deblјom
    # olovkom" tako da postane malo veca u svim pravcima.
    dilated = ndimage.binary_dilation(mask, iterations=15)
 
    # "prsten" = naduvana maska MINUS originalna maska (~ je NOT operator
    # za bool nizove). Rezultat: True samo u tankom pojasu OKO originalne
    # mase, a False i unutar mase i daleko od nje.
    ring = dilated & (~mask)
    ring_mean = float(image[ring].mean()) if ring.sum() > 0 else float("nan")
 
    # --- Provera: koliko % maske "upada" u crnu pozadinu (van dojke) ---
    # Ako je pravilno pozicionirana, maska treba da bude SKROZ unutar
    # tkiva dojke (svetlo-sivi pikseli), NE u crnom regionu oko dojke
    # (koji ima piksel vrednosti blizu 0). Ako veliki % piksela maske ima
    # vrednost < 5 (skoro crno), to je znak da maska "curi" van dojke.
    #
    # (image[mask] < 5) pravi NOVI bool niz (True gde je piksel taman),
    # a .mean() nad bool nizom daje PROCENAT True vrednosti (jer se True
    # racuna kao 1, False kao 0 - prosek 1 i 0 vrednosti = udeo True-ova).
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
 
    # presek (&) dva skupa kljuceva - ID-jevi koji postoje U OBA recnika
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
 
        # Slucaj oznacavamo sumnjivim ako VAZI BILO KOJI od ova dva uslova:
        #   1) vise od 15% maske je na crnoj pozadini, ILI
        #   2) unutrasnjost maske NIJE svetlija od okoline (neuobicajeno
        #      za masu - moglo bi da znaci da je maska "promasila" pravu
        #      leziju i pala na susedno, obicno tkivo)
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
 
    # --- Precizan (bounding-box) vizuelni prikaz prvih 6 primera ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()  # pretvara 2D niz osa (2 reda x 3 kolone) u 1D niz od 6,
                          # da mozemo da mu pristupamo kao axes[0], axes[1]... axes[5]
 
    for i, img_id in enumerate(common_ids[:6]):
        ds = pydicom.dcmread(dicom_index[img_id])
        image = ds.pixel_array.astype(np.float32)
        mask = load_mask_bool(mask_index[img_id])
 
        # np.where(mask) vraca DVA niza: sve y-koordinate i sve x-koordinate
        # piksela gde je mask==True. To su, drugim recima, koordinate SVIH
        # "belih" piksela maske.
        ys, xs = np.where(mask)
 
        # Bounding box = najmanji pravougaonik koji obuhvata SVE te piksele.
        # ys.min()/ys.max() = najgornji/najdonji beli piksel, isto za x.
        # Dodajemo "pad" (80px) sa svake strane da vidimo i malo OKOLNOG
        # tkiva, ne samo golu masku - inace ne bismo imali kontekst.
        pad = 80
        y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad)
 
        # Isecanje niza (slicing) - image[y0:y1, x0:x1] uzima samo taj region,
        # DIREKTNO iz numpy niza (precizno, za razliku od mog ranijeg pokusaja
        # da pogadjam koordinate na vec generisanoj matplotlib slici).
        crop_img = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
 
        axes[i].imshow(crop_img, cmap="gray")
        # .contour() umesto .imshow() za masku - crta SAMO liniju granice
        # (gde maska prelazi iz False u True), ne popunjava celu oblast.
        # To nam daje jasniji prikaz nego providna crvena mrlja od ranije.
        axes[i].contour(crop_mask, levels=[0.5], colors="red", linewidths=2)
        axes[i].set_title(f"ID {img_id}")
        axes[i].axis("off")
 
    plt.tight_layout()
    plt.savefig("provera_bbox_zoom.png", dpi=150)
    print("\nSacuvano: provera_bbox_zoom.png")
 
 
if __name__ == "__main__":
    main()