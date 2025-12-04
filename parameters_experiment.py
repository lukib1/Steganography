import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from code2_new import (
    load_grayscale_image, partition_host_image,
    permute_watermark, depermute_watermark,
    embed_watermark_in_ftl, embed_watermark_in_fbr_U,
    combine_subimages, extract_watermark_from_ftlw, extract_bits_from_fbrw_U,
    psnr, ber, normalized_correlation,
)

from attacks import (
    attack_jpeg, attack_gaussian_noise, attack_blur, attack_resize
)

# ---------------------------------------------------------
# Lista napada koje koristimo za prosjek robusnosti
# ---------------------------------------------------------

ATTACKS = [
    ("JPEG_Q70",           lambda img: attack_jpeg(img, quality=70)),
    ("Gaussian_noise_10",  lambda img: attack_gaussian_noise(img, sigma=10)),
    ("Blur_1.0",           lambda img: attack_blur(img, radius=1.0)),
    ("Resize_0.5",         lambda img: attack_resize(img, scale=0.5)),
]

# ---------------------------------------------------------
# Embedding + (višestruki) napadi + extraction + metrike
# ---------------------------------------------------------

def embed_and_evaluate_multi(host_path, wm_path, secret_key,
                             block_size, T, alpha,
                             attacks):
    """
    1) Učita host i watermark
    2) Embedda watermark u D (ftl) i U (fbr)
    3) Izračuna PSNR(host, watermarked) bez napada
    4) Za SVAKI attack_fn u 'attacks' primijeni napad na watermarked,
       extrakta watermark, računa BER i NC
    5) Vraća PSNR + PROSJEČNE BER/NC vrijednosti preko svih napada
    """
    # ---- LOAD IMAGES ----
    host = load_grayscale_image(host_path)
    wm_original = load_grayscale_image(wm_path)

    # ---- PARTITION HOST ----
    ftl, ftr, fbl, fbr = partition_host_image(host)

    # ---- RESIZE WATERMARK TO MATCH BLOCK CAPACITY ----
    H_ftl = ftl.shape[0]
    blocks_per_dim = H_ftl // block_size

    wm_resized_img = Image.fromarray(wm_original.astype(np.uint8)).resize(
        (blocks_per_dim, blocks_per_dim),
        Image.NEAREST
    )
    wm_resized = np.array(wm_resized_img)

    # ground-truth bitovi watermarka (0/1)
    wm_true_bits = (wm_resized // 255).astype(np.uint8)

    # ---- PERMUTE WATERMARK ----
    wm_perm = permute_watermark(wm_resized, secret_key)

    # ---- EMBEDDING ----
    ftl_w = embed_watermark_in_ftl(ftl, wm_perm, block_size, T=T)
    fbr_w = embed_watermark_in_fbr_U(fbr, wm_perm, block_size, alpha=alpha)

    watermarked = combine_subimages(ftl_w, ftr, fbl, fbr_w)

    # PSNR bez napada (kao u paperu)
    PSNR_no_attack = psnr(host, watermarked)

    # ---- LOOP PREKO NAPADA ----
    ber_D_list, ber_U_list = [], []
    nc_D_list, nc_U_list = [], []

    for name, attack_fn in attacks:
        attacked = attack_fn(watermarked)

        # EXTRACTION
        ftlw_ex, _, _, fbrw_ex = partition_host_image(attacked)

        wm_bits_D, _ = extract_watermark_from_ftlw(ftlw_ex, block_size, T=T)
        wm_bits_U, _ = extract_bits_from_fbrw_U(fbrw_ex, block_size)

        # de-permute extracted bits
        wm_bits_D_dep = depermute_watermark(wm_bits_D, secret_key)
        wm_bits_U_dep = depermute_watermark(wm_bits_U, secret_key)

        # METRIKE za ovaj napad
        ber_D_list.append(ber(wm_true_bits, wm_bits_D_dep))
        ber_U_list.append(ber(wm_true_bits, wm_bits_U_dep))
        nc_D_list.append(normalized_correlation(wm_true_bits, wm_bits_D_dep))
        nc_U_list.append(normalized_correlation(wm_true_bits, wm_bits_U_dep))

    # prosjek preko svih napada
    BER_D = float(np.mean(ber_D_list))
    BER_U = float(np.mean(ber_U_list))
    NC_D  = float(np.mean(nc_D_list))
    NC_U  = float(np.mean(nc_U_list))

    return PSNR_no_attack, BER_D, BER_U, NC_D, NC_U

# ---------------------------------------------------------
# Sweep po T (D grana)
# ---------------------------------------------------------

def sweep_T(host_path, wm_path, secret_key, block_size, alpha_fixed, T_values):
    rows = []

    for T in T_values:
        PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate_multi(
            host_path, wm_path, secret_key,
            block_size, T, alpha_fixed,
            attacks=ATTACKS
        )

        rows.append({
            "T": T,
            "alpha": alpha_fixed,
            "PSNR": PSNR_no_attack,
            "BER_D": BER_D,
            "BER_U": BER_U,
            "NC_D": NC_D,
            "NC_U": NC_U,
        })

    df_T = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]
    df_T[numeric_cols] = df_T[numeric_cols].astype(float).round(3)

    df_T.to_csv(
        "T_sweep_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    # --- GRAF 1: T vs PSNR (no attack) ---
    plt.figure()
    plt.plot(df_T["T"], df_T["PSNR"], marker="o")
    plt.xlabel("Step size T")
    plt.ylabel("PSNR (dB)")
    plt.title("Step size T vs PSNR (no attack)")
    plt.grid(True)
    plt.savefig("T_vs_PSNR.png", dpi=300, bbox_inches="tight")

    # --- GRAF 2: T vs NC_D (avg preko 4 napada) ---
    plt.figure()
    plt.plot(df_T["T"], df_T["NC_D"], marker="o")
    plt.xlabel("Step size T")
    plt.ylabel("NC_D (avg over attacks)")
    plt.title("Step size T vs NC_D (avg over JPEG, noise, blur, resize)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("T_vs_NC_D_avg_attacks.png", dpi=300, bbox_inches="tight")

    print("\n===== SWEEP PO T ZAVRSEN =====")
    print(df_T.to_string(index=False))

    return df_T

# ---------------------------------------------------------
# Sweep po alpha (U grana)
# ---------------------------------------------------------

def sweep_alpha(host_path, wm_path, secret_key, block_size, T_fixed, alpha_values):
    rows = []

    for alpha in alpha_values:
        PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate_multi(
            host_path, wm_path, secret_key,
            block_size, T_fixed, alpha,
            attacks=ATTACKS
        )

        rows.append({
            "alpha": alpha,
            "T": T_fixed,
            "PSNR": PSNR_no_attack,
            "BER_D": BER_D,
            "BER_U": BER_U,
            "NC_D": NC_D,
            "NC_U": NC_U,
        })

    df_A = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]
    df_A[numeric_cols] = df_A[numeric_cols].astype(float).round(3)

    df_A.to_csv(
        "alpha_sweep_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    # --- GRAF 3: alpha vs PSNR ---
    plt.figure()
    plt.plot(df_A["alpha"], df_A["PSNR"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("PSNR (dB)")
    plt.title("alpha vs PSNR (no attack)")
    plt.grid(True)
    plt.savefig("alpha_vs_PSNR.png", dpi=300, bbox_inches="tight")

    # --- GRAF 4: alpha vs NC_U (avg preko napada) ---
    plt.figure()
    plt.plot(df_A["alpha"], df_A["NC_U"], marker="o")
    plt.xlabel("alpha")
    plt.ylabel("NC_U (avg over attacks)")
    plt.title("alpha vs NC_U (avg over JPEG, noise, blur, resize)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.savefig("alpha_vs_NC_U_avg_attacks.png", dpi=300, bbox_inches="tight")

    print("\n===== SWEEP PO ALPHA ZAVRSEN =====")
    print(df_A.to_string(index=False))

    return df_A

# ---------------------------------------------------------
# Zajednički sweep po T i alpha (2D grid)
# ---------------------------------------------------------

def sweep_T_alpha_joint(host_path, wm_path, secret_key,
                        T_values, alpha_values,
                        block_size, PSNR_min):
    rows = []

    for T in T_values:
        for alpha in alpha_values:
            PSNR_no_attack, BER_D, BER_U, NC_D, NC_U = embed_and_evaluate_multi(
                host_path, wm_path, secret_key,
                block_size, T, alpha,
                attacks=ATTACKS
            )

            rows.append({
                "T": T,
                "alpha": alpha,
                "PSNR": PSNR_no_attack,
                "BER_D": BER_D,
                "BER_U": BER_U,
                "NC_D": NC_D,
                "NC_U": NC_U,
            })

    df = pd.DataFrame(rows)
    numeric_cols = ["PSNR", "BER_D", "BER_U", "NC_D", "NC_U"]
    df[numeric_cols] = df[numeric_cols].astype(float).round(3)

    df["NC_sum"] = (df["NC_D"] + df["NC_U"]).round(3)

    df.to_csv(
        "T_alpha_grid_results.csv",
        sep=";",
        index=False,
        float_format="%.3f"
    )

    df_ok = df[df["PSNR"] >= PSNR_min].copy()
    if df_ok.empty:
        print(f"\nUPOZORENJE: niti jedan (T, alpha) nema PSNR >= {PSNR_min} dB.")
        print("Biramo najbolji par prema NC_sum bez PSNR praga.\n")
        df_ok = df.copy()

    best_row = df_ok.sort_values("NC_sum", ascending=False).iloc[0]
    best_T = float(best_row["T"])
    best_alpha = float(best_row["alpha"])

    print("\n===== ZAJEDNICKI SWEEP PO T I ALPHA =====")
    print(df.to_string(index=False))
    print("\nFiltrirano (PSNR >= {:.1f} dB):".format(PSNR_min))
    print(df_ok.to_string(index=False))
    print("\n>> Preporuceni parametri (na temelju PSNR praga i NC_sum, avg over 4 attacks):")
    print(f"   T*     = {best_T}")
    print(f"   alpha* = {best_alpha}")

    return df, best_T, best_alpha


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    host_path = "picture.png"
    wm_path   = "ivan.png"
    secret_key = "my_secret_key_123"
    block_size = 8

    # 1) Sweep po T (uz fiksni alpha)
    df_T = sweep_T(
        host_path, wm_path, secret_key,
        block_size=block_size,
        alpha_fixed=0.05,            
        T_values=list(range(10, 101, 10))
    )

    # 2) Sweep po alpha (uz fiksni T)
    df_A = sweep_alpha(
        host_path, wm_path, secret_key,
        block_size=block_size,
        T_fixed=60,                    
        alpha_values=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )

    # 3) Zajednička optimizacija T, alpha (2D grid, prosjek preko 4 napada)
    df_grid, T_star, alpha_star = sweep_T_alpha_joint(
        host_path, wm_path, secret_key,
        T_values=list(range(10, 101, 10)),
        alpha_values=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        block_size=block_size,
        PSNR_min=40.0
    )


