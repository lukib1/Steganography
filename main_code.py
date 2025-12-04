from matplotlib.cbook import flatten
import numpy as np  # numerical arrays and matrix operations
from PIL import Image  # image loading and saving
# globalni spremnik za parametre D-grane (koriste se pri ekstrakciji)
_LAST_DMIN = None
_LAST_DMAX = None


#for preping images
def load_grayscale_image(path):
    img = Image.open(path).convert("L")
    return np.array(img)

def save_grayscale_image(arr, path):
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


# =========================
#  A. Host Image Partition
# =========================

# (1) and (2)
def partition_host_image(host_img):
    N = host_img.shape[0]
    half = N // 2

    ftl = host_img[0:half,     0:half]   # top-left
    ftr = host_img[0:half,     half:N]   # top-right
    fbl = host_img[half:N,     0:half]   # bottom-left
    fbr = host_img[half:N,     half:N]   # bottom-right

    return ftl, ftr, fbl, fbr


#=========================
#B. Watermark Embedding in D Matrix
#=========================

# (3) + (4)
# we are doing SVD on blocks of ftl and collecting Dlarge
def svd_blocks_and_collect_Dlarge(ftl, block_size, wm_shape):
    H = ftl.shape[0] #number of rows
    blocks_per_dim = H // block_size 
    num_blocks = blocks_per_dim * blocks_per_dim #total number of blocks

    # Lists to store U, S, Vt from SVD for each block
    U_list = [] 
    S_list = []
    Vt_list = []
    Dlarge_flat = np.zeros(num_blocks, dtype=float) #array to store largest singular values

    idx = 0
    for by in range(blocks_per_dim): #loop over block rows
        for bx in range(blocks_per_dim): #loop over block columns
            y0 = by * block_size
            x0 = bx * block_size
            block = ftl[y0:y0+block_size, x0:x0+block_size].astype(float)

            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            U_list.append(U)
            S_list.append(S)
            Vt_list.append(Vt)

            Dlarge_flat[idx] = S[0] #S is sorted in descending order, so it's the first element  
            idx += 1

    Dlarge = Dlarge_flat.reshape(wm_shape)  #reshape to match watermark shape
    return U_list, S_list, Vt_list, Dlarge

# (5) + (6)
def modify_Dlarge_with_watermark(Dlarge, wm_bits, T, dmin=None, dmax=None):
    """
    Kvantizacija najvećih singularnih vrijednosti prema tablici u radu.

    - cijeli raspon [dmin, dmax] dijelimo na binove širine T
      (d_low = dmin + k*T, d_high = d_low + T)
    - ako je bit = 1 -> Dlarge ide u donju polovicu (Range1)
      Range1 = [d_low, (d_low + d_high)/2]
      i postavimo ga u sredinu Range1
    - ako je bit = 0 -> Dlarge ide u gornju polovicu (Range2)
      Range2 = [(d_low + d_high)/2, d_high]
      i postavimo ga u sredinu Range2
    """

    if dmin is None:
        dmin = Dlarge.min()
    if dmax is None:
        dmax = Dlarge.max()

    Dlarge_flat = Dlarge.flatten().astype(float)
    wm_flat = wm_bits.flatten().astype(int)

    num_bins = int(np.ceil((dmax - dmin) / T))  # broj binova koji pokrivaju [dmin, dmax]

    modified_flat = np.zeros_like(Dlarge_flat, dtype=float)

    for i, d in enumerate(Dlarge_flat):
        bit = wm_flat[i]

        # indeks bina u [0, num_bins-1]
        bin_idx = int((d - dmin) // T)
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1

        d_low  = dmin + bin_idx * T
        d_high = d_low + T
        mid    = 0.5 * (d_low + d_high)

        if bit == 1:
            # u Range1 -> u sredinu donje polovice
            new_d = 0.5 * (d_low + mid)
        else:
            # u Range2 -> u sredinu gornje polovice
            new_d = 0.5 * (mid + d_high)

        modified_flat[i] = new_d

    return modified_flat.reshape(Dlarge.shape)


def embed_watermark_in_ftl(ftl, wm, block_size=8, T=5.0):
    """
    Embeddanje watermarka u D-matricu (top-left kvadrant).

    Uz to globalno spremamo dmin i dmax, kako bi ih pri ekstrakciji
    koristili istu tablicu kvantizacije kao i ovdje (kao u radu).
    """
    global _LAST_DMIN, _LAST_DMAX

    wm_bits = (wm // 255).astype(np.uint8)

    U_list, S_list, Vt_list, Dlarge = svd_blocks_and_collect_Dlarge(
        ftl, block_size, wm_bits.shape
    )

    dmin = Dlarge.min()
    dmax = Dlarge.max()
    _LAST_DMIN = dmin
    _LAST_DMAX = dmax

    Dlarge_modified = modify_Dlarge_with_watermark(Dlarge, wm_bits, T, dmin, dmax)

    ftl_w = reconstruct_ftl_from_svd(
        U_list, S_list, Vt_list, Dlarge_modified, block_size, ftl.shape
    )
    return ftl_w


def combine_subimages(ftl_w, ftr, fbl, fbr_w):
    half = ftl_w.shape[0]
    N = half * 2
    result = np.zeros((N, N), dtype=np.uint8)

    result[0:half,     0:half]   = ftl_w
    result[0:half,     half:N]   = ftr
    result[half:N,     0:half]   = fbl
    result[half:N,     half:N]   = fbr_w

    return result

# (7)

def reconstruct_ftl_from_svd(U_list, S_list, Vt_list, Dlarge_modified, block_size, ftl_shape):

    H = ftl_shape[0]
    blocks_per_dim = H // block_size

    ftl_w = np.zeros((H, H), dtype=float)
    D_flat = Dlarge_modified.flatten()

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            U = U_list[idx]
            S = S_list[idx].copy()
            Vt = Vt_list[idx]

            S[0] = D_flat[idx]

            block_rec = U @ np.diag(S) @ Vt

            y0 = by * block_size
            x0 = bx * block_size
            ftl_w[y0:y0+block_size, x0:x0+block_size] = block_rec

            idx += 1

    ftl_w = np.clip(ftl_w, 0, 255).astype(np.uint8)
    return ftl_w


# =========================
#  Permutation functions
# =========================

def _rng_from_key(key):
    seed = abs(hash(key)) % (2**32)
    return np.random.default_rng(seed)

def permute_watermark(wm_img, key):
    flat = wm_img.flatten()
    num_pixels = flat.size

    rng = _rng_from_key(key)
    perm_indices = np.arange(num_pixels)
    rng.shuffle(perm_indices)

    perm_flat = flat[perm_indices]
    return perm_flat.reshape(wm_img.shape)


def depermute_watermark(wm_perm, key):
    flat = wm_perm.flatten()
    num_pixels = flat.size

    rng = _rng_from_key(key)
    perm_indices = np.arange(num_pixels)
    rng.shuffle(perm_indices)

    inv_perm = np.zeros_like(perm_indices)
    inv_perm[perm_indices] = np.arange(num_pixels)

    original_flat = flat[inv_perm]
    return original_flat.reshape(wm_perm.shape)


# (8) + (9)
def embed_watermark_in_fbr_U(fbr, wm, block_size=8, alpha=0.1):
    """
    U matricu U (donji desni kvadrant) ugrađujemo bit tako da
    kontroliramo razliku |u11| - |u21|:

        bit = 1  ->  |u11| >= |u21| + alpha
        bit = 0  ->  |u21| >= |u11| + alpha

    To je u duhu rada: bit je određen usporedbom magnituda stupca U,
    a alpha je sigurnosna margina. Ekstrakcija ostaje:
        bit = 1 ako |u11| > |u21|, inače 0.
    """
    
    H = fbr.shape[0]

    wm_bits = (wm // 255).astype(np.uint8)
    wm_flat = wm_bits.flatten()
    num_bits = wm_flat.size

    blocks_per_dim = H // block_size
    fbr_w = np.zeros_like(fbr, dtype=float)

    bit_idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size

            block = fbr[y0:y0+block_size, x0:x0+block_size].astype(float)
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            if bit_idx < num_bits:
                bit = wm_flat[bit_idx]
                bit_idx += 1

                u11 = U[0, 0]
                u21 = U[1, 0]
                
                
                a = abs(u11)
                b = abs(u21)
                diff = a - b

                if bit == 1:
                    # želimo a >= b + alpha  (diff >= alpha)
                    if diff < alpha:
                        mean = 0.5 * (a + b)
                        a_new = mean + alpha / 2.0
                        b_new = mean - alpha / 2.0
                    else:
                        a_new, b_new = a, b
                else:
                    # bit = 0 -> želimo b >= a + alpha  (diff <= -alpha)
                    if diff > -alpha:
                        mean = 0.5 * (a + b)
                        a_new = mean - alpha / 2.0
                        b_new = mean + alpha / 2.0
                    else:
                        a_new, b_new = a, b

                # vrati originale znakove
                sign11 = 1.0 if u11 >= 0 else -1.0
                sign21 = 1.0 if u21 >= 0 else -1.0
                U[0, 0] = sign11 * a_new
                U[1, 0] = sign21 * b_new
                
                
                """
                a = abs(u11)
                b = abs(u21)
                u_diff = a - b

                # implementacija jednadžbi (14) i (15)
                if (bit == 1 and u_diff > alpha) or (bit == 0 and u_diff < alpha):
                    # slučaj (14)
                    U[1, 0] = -abs(b - (alpha - u_diff) / 2.0)
                    U[0, 0] = -abs(a + (alpha - u_diff) / 2.0)
                else:
                    # slučaj (15)
                    U[1, 0] = -abs(b - (alpha + u_diff) / 2.0)
                    U[0, 0] = -abs(a + (alpha + u_diff) / 2.0)    
                """    
                    
            block_w = U @ np.diag(S) @ Vt
            fbr_w[y0:y0+block_size, x0:x0+block_size] = block_w

    fbr_w = np.clip(fbr_w, 0, 255).astype(np.uint8)
    return fbr_w


# =========================
#  Extraction from D (ftlw)
# =========================


def svd_blocks_and_collect_D_from_ftlw(ftlw, block_size):
    H = ftlw.shape[0]
    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    Dlarge_flat = np.zeros(num_blocks, dtype=float)

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size
            block = ftlw[y0:y0+block_size, x0:x0+block_size].astype(float)

            _, S, _ = np.linalg.svd(block, full_matrices=False)
            Dlarge_flat[idx] = S[0]
            idx += 1

    Dlarge = Dlarge_flat.reshape((blocks_per_dim, blocks_per_dim))
    return Dlarge


def extract_bits_from_Dlarge(Dlarge, T, dmin, dmax):
    D_flat = Dlarge.flatten().astype(float)
    bits_flat = np.zeros_like(D_flat, dtype=np.uint8)

    num_bins = int(np.ceil((dmax - dmin) / T))

    for i, d in enumerate(D_flat):
        bin_idx = int((d - dmin) // T)
        if bin_idx < 0:
            bin_idx = 0
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1

        d_low  = dmin + bin_idx * T
        d_high = d_low + T
        mid    = 0.5 * (d_low + d_high)

        # ako je u Range1 -> bit=1, inače bit=0 (Range2)
        bits_flat[i] = 1 if d < mid else 0

    wm_bits = bits_flat.reshape(Dlarge.shape)
    return wm_bits



def extract_watermark_from_ftlw(ftlw, block_size=8, T=5.0, dmin=None, dmax=None):
    """
    Ekstrakcija watermarka iz D-grane.

    Ako dmin/dmax nisu eksplicitno zadani, pokušavamo koristiti
    vrijednosti iz zadnjeg embedanja (_LAST_DMIN/_LAST_DMAX).
    Ako ni njih nema (npr. samo ekstrakcija tuđe slike), kao fallback
    uzimamo dmin/dmax iz trenutnog Dlarge (nije idealno, ali radi).
    """
    global _LAST_DMIN, _LAST_DMAX

    Dlarge = svd_blocks_and_collect_D_from_ftlw(ftlw, block_size)

    if dmin is None or dmax is None:
        if _LAST_DMIN is not None and _LAST_DMAX is not None:
            dmin = _LAST_DMIN
            dmax = _LAST_DMAX
        else:
            dmin = Dlarge.min()
            dmax = Dlarge.max()

    wm_bits = extract_bits_from_Dlarge(Dlarge, T, dmin, dmax)
    wm_img = (wm_bits * 255).astype(np.uint8)
    return wm_bits, wm_img



# =========================
#  Extraction from U (fbrw)
# =========================

def extract_bits_from_fbrw_U(fbrw, block_size=8):
    """
    Inverzna operacija: za svaki blok gledamo |u11| i |u21|
    i donosimo odluku:
       bit = 1  ako je |u11| > |u21|
       bit = 0  inače
    """
    H = fbrw.shape[0]
    blocks_per_dim = H // block_size
    num_blocks = blocks_per_dim * blocks_per_dim

    bits_flat = np.zeros(num_blocks, dtype=np.uint8)

    idx = 0
    for by in range(blocks_per_dim):
        for bx in range(blocks_per_dim):
            y0 = by * block_size
            x0 = bx * block_size

            block = fbrw[y0:y0+block_size, x0:x0+block_size].astype(float)
            U, _, _ = np.linalg.svd(block, full_matrices=False)

            u11 = U[0, 0]
            u21 = U[1, 0]

            bits_flat[idx] = 1 if abs(u11) > abs(u21) else 0
            idx += 1

    wm_bits = bits_flat.reshape((blocks_per_dim, blocks_per_dim))
    wm_img = (wm_bits * 255).astype(np.uint8)
    return wm_bits, wm_img

# =========================
#  METRIKE
# =========================

def mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2, max_val=255.0):
    m = mse(img1, img2)
    if m == 0:
        return np.inf
    return 10 * np.log10((max_val ** 2) / m)

def ber(wm_true_bits, wm_extracted_bits):
    wm_true_bits = wm_true_bits.flatten()
    wm_extracted_bits = wm_extracted_bits.flatten()
    errors = np.sum(wm_true_bits != wm_extracted_bits)
    return errors / wm_true_bits.size

def normalized_correlation(wm_true, wm_extracted):
    wm_true = wm_true.astype(np.float64)
    wm_extracted = wm_extracted.astype(np.float64)
    num = np.sum(wm_true * wm_extracted)
    den = np.sqrt(np.sum(wm_true**2) * np.sum(wm_extracted**2))
    if den == 0:
        return 0
    return num / den

# =========================
#  MAIN: embedding + extraction demo
# =========================

if __name__ == "__main__":
    #ucitavanje slika i parametri
    host_path   = "picture.png"   
    wm_path     = "ivan.png"       
    secret_key  = "my_secret_key_123"
    block_size  = 8

    # parametri embedanja (možeš ih mijenjati)
    T     = 60.0
    alpha = 0.03

    # -------------------------------------------------
    #  1) UČITAJ I PRIPREMI
    # -------------------------------------------------
    host = load_grayscale_image(host_path)
    wm_original = load_grayscale_image(wm_path)

    # particioniraj host
    ftl, ftr, fbl, fbr = partition_host_image(host)

    # resize watermarka na broj blokova u ftl
    H_ftl = ftl.shape[0]
    blocks_per_dim = H_ftl // block_size

    wm_resized_img = Image.fromarray(wm_original.astype(np.uint8)).resize(
        (blocks_per_dim, blocks_per_dim),
        Image.NEAREST
    )
    wm_resized = np.array(wm_resized_img)

    # ground-truth bitovi (0/1)
    wm_true_bits = (wm_resized // 255).astype(np.uint8)

    # spremi resizeani watermark radi kontrole
    save_grayscale_image(wm_resized, "wm_resized.png")

    # -------------------------------------------------
    #  2) PERMUTACIJA + EMBEDDING (D + U)
    # -------------------------------------------------
    wm_perm = permute_watermark(wm_resized, secret_key)

    ftl_w = embed_watermark_in_ftl(ftl, wm_perm, block_size=block_size, T=T)
    fbr_w = embed_watermark_in_fbr_U(fbr, wm_perm, block_size=block_size, alpha=alpha)

    watermarked_host = combine_subimages(ftl_w, ftr, fbl, fbr_w)

    save_grayscale_image(host,             "host_original.png")
    save_grayscale_image(watermarked_host, "host_watermarked.png")

    psnr_no_attack = psnr(host, watermarked_host)
    print(f"PSNR (host vs watermarked, bez napada): {psnr_no_attack:.3f} dB")

    # -------------------------------------------------
    #  3) EKSTRAKCIJA IZ WATERMARKED SLIKE (BEZ NAPADA)
    # -------------------------------------------------
    # (možemo direktno koristiti watermarked_host, bez ponovnog učitavanja)
    ftlw_ex, _, _, fbrw_ex = partition_host_image(watermarked_host)

    # --- D grana ---
    wm_bits_D_perm, _ = extract_watermark_from_ftlw(ftlw_ex, block_size=block_size, T=T)
    wm_bits_D = depermute_watermark(wm_bits_D_perm, secret_key)
    wm_img_D  = (wm_bits_D * 255).astype(np.uint8)
    save_grayscale_image(wm_img_D, "extracted_D_no_attack.png")

    # --- U grana ---
    wm_bits_U_perm, _ = extract_bits_from_fbrw_U(fbrw_ex, block_size=block_size)
    wm_bits_U = depermute_watermark(wm_bits_U_perm, secret_key)
    wm_img_U  = (wm_bits_U * 255).astype(np.uint8)
    save_grayscale_image(wm_img_U, "extracted_U_no_attack.png")

    # -------------------------------------------------
    #  4) METRIKE (BER + NC ZA OBE GRANE)
    # -------------------------------------------------
    ber_D = ber(wm_true_bits, wm_bits_D)
    ber_U = ber(wm_true_bits, wm_bits_U)

    nc_D = normalized_correlation(wm_true_bits, wm_bits_D)
    nc_U = normalized_correlation(wm_true_bits, wm_bits_U)

    print("\nMETRIKE BEZ NAPADA (trebale bi biti idealne ili jako blizu):")
    print(f"  D grana: BER_D = {ber_D:.6f},  NC_D = {nc_D:.6f}")
    print(f"  U grana: BER_U = {ber_U:.6f},  NC_U = {nc_U:.6f}")
