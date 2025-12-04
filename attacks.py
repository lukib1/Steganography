import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
from PIL import ImageFilter
import shutil

from code2_new import (
    load_grayscale_image, save_grayscale_image,
    partition_host_image,
    permute_watermark, depermute_watermark,
    embed_watermark_in_ftl, embed_watermark_in_fbr_U,
    combine_subimages, extract_watermark_from_ftlw, extract_bits_from_fbrw_U,
    psnr, ber, normalized_correlation
)

# =========================================================
#  NAPADI
# =========================================================


def attack_jpeg(img_arr, quality=30):
    img = Image.fromarray(img_arr.astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    attacked = Image.open(buffer).convert("L")
    return np.array(attacked)


def attack_gaussian_noise(img_arr, sigma=10):
    noise = np.random.normal(0, sigma, img_arr.shape)
    noisy = img_arr.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def attack_blur(img_arr, radius=1.0):
    img = Image.fromarray(img_arr.astype(np.uint8))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)

"""
#bikubicna interpolacija, njeznija od bilinear
def attack_resize(img_arr, scale=0.5):
    h, w = img_arr.shape
    img = Image.fromarray(img_arr.astype(np.uint8))
    small = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    back = small.resize((w, h), Image.BICUBIC)
    return np.array(back)
"""


# na 0.5 ga ubije, veci i manje scale bolji. Ako se poveca T je bolje
def attack_resize(img_arr, scale=0.5):
    H, W = img_arr.shape
    img = Image.fromarray(img_arr.astype(np.uint8))
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    small = img.resize((new_w, new_h), resample=Image.BILINEAR)
    back = small.resize((W, H), resample=Image.BILINEAR)
    return np.array(back)



def attack_rotation(img_arr, angle=20):
    img = Image.fromarray(img_arr.astype(np.uint8))
    rot = img.rotate(angle, resample=Image.BILINEAR, expand=True)

    w0, h0 = img.size
    w1, h1 = rot.size
    left = (w1 - w0) // 2
    top = (h1 - h0) // 2
    cropped = rot.crop((left, top, left + w0, top + h0))
    return np.array(cropped)


def attack_rotation2(img_arr, angle=20):
    img = Image.fromarray(img_arr.astype(np.uint8))
    # prvi put: +angle, bez expand, bilinear, pozadina crna
    rot = img.rotate(-angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    # drugi put: -angle, opet bez expand
    rot_back = rot.rotate(angle, resample=Image.BILINEAR, expand=False,fillcolor=0)
    return np.array(rot_back)



def attack_median_3x3(img_arr):
    """Median filter 3x3."""
    img = Image.fromarray(img_arr.astype(np.uint8))
    filt = img.filter(ImageFilter.MedianFilter(size=3))
    return np.array(filt)


def attack_salt_pepper(img_arr, density=0.01):
    """Salt & pepper šum zadane gustoće."""
    out = img_arr.copy()
    H, W = out.shape
    N = H * W
    num_sp = int(density * N)
    if num_sp == 0:
        return out

    flat = out.flatten()
    idx = np.random.choice(N, num_sp, replace=False)
    half = num_sp // 2
    flat[idx[:half]] = 0
    flat[idx[half:]] = 255
    return flat.reshape(out.shape)


def attack_cropping(img_arr, crop_ratio=0.25):
    H, W = img_arr.shape
    out = img_arr.copy()
    scale = np.sqrt(crop_ratio)
    crop_h = int(H * scale)
    crop_w = int(W * scale)
    out[0:crop_h, 0:crop_w] = 0
    return out


def attack_brightness(img_arr, factor=1.2):
    """Promjena osvjetljenja (npr. +20% => factor=1.2)."""
    img = Image.fromarray(img_arr.astype(np.uint8))
    enh = ImageEnhance.Brightness(img)
    bright = enh.enhance(factor)
    return np.array(bright)


def attack_gamma(img_arr, gamma=0.9):
    """Gamma korekcija."""
    x = img_arr.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return y


def attack_rowcol_blanking(img_arr, step=20):
    """Row/column blanking – neke redove i stupce postavimo na 0."""
    out = img_arr.copy()
    H, W = out.shape
    rows = np.arange(0, H, step)
    cols = np.arange(0, W, step)
    out[rows, :] = 0
    out[:, cols] = 0
    return out



def attack_rowcol_copying(img_arr, step=20):
    """Row/column copying - neke redove/stupce kopiramo iz susjednih."""
    out = img_arr.copy()
    H, W = out.shape

    rows = np.arange(step, H, step)
    for r in rows:
        out[r, :] = out[r - 1, :]

    cols = np.arange(step, W, step)
    for c in cols:
        out[:, c] = out[:, c - 1]

    return out



def attack_bitplane_lsb(img_arr):
    """Bit-plane removal (LSB) - poništavamo zadnji bit."""
    return (img_arr & 0xFe).astype(np.uint8)


# =========================================================
#  EMBEDDING JEDNOM, PA VIŠE NAPADA
# =========================================================

def embed_once(host_path, wm_path, secret_key, block_size, T, alpha):
    """
    Učita host + watermark, ugradi watermark (D + U),
    vrati: host, watermarked, ground-truth watermark bitove (0/1) i resize-an watermark.
    """
    host = load_grayscale_image(host_path)
    wm_orig = load_grayscale_image(wm_path)

    # Partition host
    ftl, ftr, fbl, fbr = partition_host_image(host)

    # Resize watermark na broj blokova
    H_ftl = ftl.shape[0]
    blocks_per_dim = H_ftl // block_size

    wm_resized_img = Image.fromarray(wm_orig.astype(np.uint8)).resize(
        (blocks_per_dim, blocks_per_dim),
        Image.NEAREST
    )
    wm_resized = np.array(wm_resized_img)
    wm_true_bits = (wm_resized // 255).astype(np.uint8)

    # permutirani watermark
    wm_perm = permute_watermark(wm_resized, secret_key)

    # embedding u D i U
    ftl_w = embed_watermark_in_ftl(ftl, wm_perm, block_size, T=T)
    fbr_w = embed_watermark_in_fbr_U(fbr, wm_perm, block_size, alpha=alpha)

    watermarked = combine_subimages(ftl_w, ftr, fbl, fbr_w)

    return host, wm_true_bits, wm_resized, watermarked


def evaluate_single_attack(idx, name, attack_fn,
                           host, wm_true_bits, watermarked,
                           block_size, T, secret_key,
                           out_dir="results_paper"):
    """
    Primijeni jedan napad, extraktaj watermark, spremi slike i vrati metrike.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Napad
    attacked = attack_fn(watermarked)

    # Spremi napadnutu sliku
    attacked_name = os.path.join(out_dir, f"{idx:02d}_{name}_attacked.png")
    save_grayscale_image(attacked, attacked_name)

    # Partition napadnute slike
    ftlw_ex, _, _, fbrw_ex = partition_host_image(attacked)

    # Extract iz D
    wm_bits_D, _ = extract_watermark_from_ftlw(ftlw_ex, block_size, T=T)
    wm_bits_D_dep = depermute_watermark(wm_bits_D, secret_key)
    wm_img_D_dep = (wm_bits_D_dep * 255).astype(np.uint8)
    wm_D_name = os.path.join(out_dir, f"{idx:02d}_{name}_wm_D.png")
    save_grayscale_image(wm_img_D_dep, wm_D_name)

    # Extract iz U
    wm_bits_U, _ = extract_bits_from_fbrw_U(fbrw_ex, block_size)
    wm_bits_U_dep = depermute_watermark(wm_bits_U, secret_key)
    wm_img_U_dep = (wm_bits_U_dep * 255).astype(np.uint8)
    wm_U_name = os.path.join(out_dir, f"{idx:02d}_{name}_wm_U.png")
    save_grayscale_image(wm_img_U_dep, wm_U_name)

    # METRIKE
    psnr_val = psnr(host, attacked)
    ber_D = ber(wm_true_bits, wm_bits_D_dep)
    ber_U = ber(wm_true_bits, wm_bits_U_dep)
    nc_D = normalized_correlation(wm_true_bits, wm_bits_D_dep)
    nc_U = normalized_correlation(wm_true_bits, wm_bits_U_dep) 

    return {
        "Attack": name,
        "PSNR": psnr_val,
        "BER_D": ber_D,
        "BER_U": ber_U,
        "NC_D": nc_D,
        "NC_U": nc_U,
        "attacked_path": attacked_name,
        "wm_D_path": wm_D_name,
        "wm_U_path": wm_U_name,
    }


# =========================================================
#  GLAVNI EKSPERIMENT – PRAĆENJE PAPERA
# =========================================================

def run_paper_experiment(host_path, wm_path, secret_key,
                         block_size=8, T=60, alpha=0.05,
                         out_dir="results_paper"):
    
    # 0) Očisti folder s rezultatima
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)  # izbriše cijeli folder i sve u njemu

    os.makedirs(out_dir, exist_ok=True)
    
    # 1) Embed jednom
    host, wm_true_bits, wm_resized, watermarked = embed_once(
        host_path, wm_path, secret_key, block_size, T, alpha
    )

    # Spremi osnovne slike: original, resized watermark, permutirani host
    save_grayscale_image(host, os.path.join(out_dir, "host_original.png"))
    save_grayscale_image(wm_resized, os.path.join(out_dir, "watermark_resized.png"))
    save_grayscale_image(watermarked, os.path.join(out_dir, "watermarked_no_attack.png"))

    psnr_no_attack = psnr(host, watermarked)

    print(f"\nPSNR bez napada (samo embedding): {psnr_no_attack:.3f} dB")
    print(f"Parametri: T = {T}, alpha = {alpha}\n")

    # 2) Lista napada (po uzoru na figure iz rada)
    attacks = [
        ("No_attack",          lambda img: img),
        ("JPEG_Q70",           lambda img: attack_jpeg(img, quality=70)),
        ("Gaussian_noise_10",  lambda img: attack_gaussian_noise(img, sigma=10)),
        ("Rotation_20deg",     lambda img: attack_rotation(img, angle=20)),
        ("Rotation2_20deg",    lambda img: attack_rotation2(img, angle=20)),
        ("Resize_0.5",         lambda img: attack_resize(img, scale=0.5)),
        ("Median_3x3",         lambda img: attack_median_3x3(img)),
        ("Blur_1.0",           lambda img: attack_blur(img, radius=1.0)),
        ("SaltPepper_0.01",    lambda img: attack_salt_pepper(img, density=0.01)),
        ("Cropping_25pct",     lambda img: attack_cropping(img, crop_ratio=0.25)),
        ("Brightness_20pct",   lambda img: attack_brightness(img, factor=1.2)),
        ("Gamma_0.9",          lambda img: attack_gamma(img, gamma=0.9)),
        ("RowCol_Blanking",    lambda img: attack_rowcol_blanking(img, step=20)),
        ("RowCol_Copying",     lambda img: attack_rowcol_copying(img, step=20)),
        ("Bitplane_LSB",       lambda img: attack_bitplane_lsb(img)),
    ]

    results = []
    for idx, (name, fn) in enumerate(attacks, start=1):
        res = evaluate_single_attack(
            idx, name, fn,
            host, wm_true_bits, watermarked,
            block_size, T, secret_key,
            out_dir=out_dir
        )
        results.append(res)

    # 3) Rezultati -> DataFrame + CSV
    df = pd.DataFrame(results)

    # zaokruživanje
    for col in ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]:
        df[col] = df[col].astype(float).round(3)

    csv_path = os.path.join(out_dir, "paper_attacks_results.csv")
    df.to_csv(csv_path, sep=";", index=False, float_format="%.3f")

    # 4) Lijepi ispis (PSNR + (NC, BER) po grani)
    print("====================================================")
    print("REZULTATI NAPADA (PSNR + NC/BER iz D i U grane)")
    print("====================================================\n")

    for r in results:
        print(
            f"{r['Attack']:15s}  "
            f"PSNR={r['PSNR']:.3f} dB   "
            f"(D: NC={r['NC_D']:.3f}, BER={r['BER_D']:.3f})   "
            f"(U: NC={r['NC_U']:.3f}, BER={r['BER_U']:.3f})"
        )

    print(f"\nCSV spremljen u: {csv_path}")
    print(f"Sve slike spremljene u folder: {out_dir}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    host_path = "picture.png"   
    wm_path   = "ivan.png"    
    secret_key = "my_secret_key_123"
    block_size = 8

    # Parametri koje smo odabrali na temelju sweep-a
    T = 60.0
    alpha = 0.05

    run_paper_experiment(
        host_path, wm_path, secret_key,
        block_size=block_size,
        T=T, alpha=alpha,
        out_dir="results_paper"
    )

