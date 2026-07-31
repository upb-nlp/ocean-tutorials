import argparse
import os
import sys
import textwrap

# Windows consoles default to cp1252, which cannot print the box-drawing
# characters used below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = "./outputs/image_preprocessing"


# Check that packages are importable
def require(packages: list[str]):
    # Friendly name → import name for the pip packages whose module differs.
    import_names = {"scikit-image": "skimage", "scikit-learn": "sklearn"}
    missing = []
    for pkg in packages:
        try:
            __import__(import_names.get(pkg, pkg))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"     Install with:  pip install {' '.join(missing)}\n")
        sys.exit(1)


def get_plt():
    """Return pyplot with a non-interactive backend (save-to-file only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_fig(fig, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    get_plt().close(fig)
    print(f"    ✓ figure saved to {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def section(title: str):
    width = 70
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def subsection(title: str):
    print(f"\n  ── {title} {'─' * max(1, 60 - len(title))}")


# Simple ASCII table printer.
def show_table(headers: list, rows: list, col_width: int = 20):
    fmt = "  " + "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("  " + "-" * (col_width * len(headers)))
    for row in rows:
        print(fmt.format(*[str(c)[:col_width - 1] for c in row]))


def describe(name: str, arr):
    """One-line summary of an image array: shape, dtype, value range."""
    import numpy as np
    print(f"  {name:<22} shape={str(arr.shape):<16} dtype={str(arr.dtype):<9}"
          f" range=[{np.min(arr):.3g}, {np.max(arr):.3g}]")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 1 — COLOR SPACES, RESIZING, NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════

def lab1_color_resize_normalize():
    require(["scikit-image", "numpy", "matplotlib"])

    import numpy as np
    from skimage import data, color
    from skimage.transform import resize
    plt = get_plt()

    section("1 — COLOR SPACES, RESIZING, NORMALIZATION")
    print(textwrap.dedent("""
      An image is just a NumPy array of numbers. Before any model sees it,
      three things must be pinned down: how colour is encoded, how big the
      array is, and what range its values live in. Get these wrong and every
      downstream step inherits the mistake.
    """))

    # ── 1.1  An image is an array ────────────────────────────────────────────
    subsection("1.1  An image is a NumPy array")

    rgb = data.astronaut()  # 512×512×3 uint8 photograph
    describe("Original (uint8)", rgb)
    print("\n  A colour image has shape (height, width, channels). Each pixel is")
    print("  three bytes — R, G, B — in the range 0–255. Row 0 is the TOP of the")
    print("  image; the last axis indexes colour.")
    print(f"\n  Top-left pixel RGB value: {tuple(int(v) for v in rgb[0, 0])}")

    # ── 1.2  Colour spaces ───────────────────────────────────────────────────
    subsection("1.2  Colour spaces — RGB, grayscale, HSV, LAB")

    gray = color.rgb2gray(rgb)      # → float64 in [0, 1]
    hsv  = color.rgb2hsv(rgb)
    lab  = color.rgb2lab(rgb)

    print("  The same picture, re-encoded. Different tasks want different spaces:\n")
    show_table(
        ["Space", "Shape", "Channel 0", "Channel 1", "Channel 2"],
        [
            ["RGB",  str(rgb.shape),  "Red",        "Green",      "Blue"],
            ["Gray", str(gray.shape), "Intensity",  "-",          "-"],
            ["HSV",  str(hsv.shape),  "Hue",        "Saturation", "Value"],
            ["LAB",  str(lab.shape),  "Lightness",  "a (g–r)",    "b (b–y)"],
        ],
        col_width=14,
    )
    print(textwrap.dedent("""
      • Grayscale   — drop colour when only structure matters (edges, OCR).
                      Note: 0.299R+0.587G+0.114B, NOT a flat average — the eye
                      is most sensitive to green.
      • HSV         — separates colour (hue) from brightness (value); ideal for
                      colour thresholding under changing light.
      • LAB         — perceptually uniform: equal numeric steps ≈ equal visual
                      steps. Used for colour-accurate comparisons.
    """))

    # ── 1.3  Resizing and interpolation ──────────────────────────────────────
    subsection("1.3  Resizing — interpolation and aspect ratio")

    small = resize(rgb, (64, 64), anti_aliasing=True)
    big   = resize(small, (256, 256), order=0, anti_aliasing=False)   # nearest
    big_c = resize(small, (256, 256), order=3, anti_aliasing=True)    # cubic

    describe("Downscaled 512→64", small)
    describe("Upscaled  64→256 (nearest)", big)
    describe("Upscaled  64→256 (cubic)", big_c)
    print(textwrap.dedent("""
      Downscaling THROWS AWAY information — anti-aliasing (a slight blur first)
      prevents jagged aliasing artefacts. Upscaling INVENTS pixels: nearest
      neighbour looks blocky, cubic looks smooth but neither adds real detail.
      CNNs need a fixed input size, so resizing (or cropping/padding) is
      mandatory — and squashing a non-square image distorts aspect ratio.
    """))

    # ── 1.4  Normalization ───────────────────────────────────────────────────
    subsection("1.4  Normalization — putting values on a common scale")

    img = gray.astype(np.float64)
    minmax = (img - img.min()) / (img.max() - img.min())
    standard = (img - img.mean()) / img.std()

    print("  Neural nets train best when inputs are small and centred.\n")
    show_table(
        ["Method", "Formula", "Min", "Max", "Mean", "Std"],
        [
            ["Raw uint8/255", "x/255",           f"{img.min():.2f}",     f"{img.max():.2f}",
             f"{img.mean():.2f}",     f"{img.std():.2f}"],
            ["Min-max [0,1]", "(x-min)/(max-min)", f"{minmax.min():.2f}",  f"{minmax.max():.2f}",
             f"{minmax.mean():.2f}",  f"{minmax.std():.2f}"],
            ["Standardize",   "(x-μ)/σ",          f"{standard.min():.2f}", f"{standard.max():.2f}",
             f"{standard.mean():.2f}", f"{standard.std():.2f}"],
        ],
        col_width=13,
    )
    print(textwrap.dedent("""
      Standardization (zero mean, unit variance) is what most CNNs use — often
      with fixed per-channel means/stds computed over the whole training set
      (the famous ImageNet [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]).
      Crucial rule: compute statistics on TRAIN only, then apply to val/test.
    """))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes[0, 0].imshow(rgb);                     axes[0, 0].set_title("RGB (original)")
    axes[0, 1].imshow(gray, cmap="gray");       axes[0, 1].set_title("Grayscale")
    axes[0, 2].imshow(hsv[:, :, 0], cmap="hsv"); axes[0, 2].set_title("HSV — Hue")
    axes[0, 3].imshow(hsv[:, :, 1], cmap="viridis"); axes[0, 3].set_title("HSV — Saturation")
    axes[1, 0].imshow(small);                   axes[1, 0].set_title("Resized 64×64")
    axes[1, 1].imshow(big);                     axes[1, 1].set_title("Upscaled (nearest)")
    axes[1, 2].imshow(big_c);                   axes[1, 2].set_title("Upscaled (cubic)")
    axes[1, 3].hist(img.ravel(), bins=60, color="#4c72b0")
    axes[1, 3].set_title("Intensity histogram")
    for ax in axes.ravel():
        if ax is not axes[1, 3]:
            ax.axis("off")
    fig.suptitle("Colour spaces, resizing, normalization", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "01_color_resize_normalize.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 2 — HISTOGRAM EQUALIZATION AND CONTRAST ADJUSTMENT
# ════════════════════════════════════════════════════════════════════════════

def lab2_histogram_contrast():
    require(["scikit-image", "numpy", "matplotlib"])

    import numpy as np
    from skimage import data, exposure, img_as_float
    plt = get_plt()

    section("2 — HISTOGRAM EQUALIZATION AND CONTRAST ADJUSTMENT")
    print(textwrap.dedent("""
      Contrast is how spread-out an image's intensities are. A photo of the
      moon taken through haze crams every pixel into a narrow band — the
      histogram is a thin spike, detail is invisible. These techniques
      redistribute intensities to reveal what is already there.
    """))

    img = img_as_float(data.moon())   # classic low-contrast grayscale image

    # ── 2.1  The histogram tells the story ───────────────────────────────────
    subsection("2.1  Reading the histogram")

    p2, p98 = np.percentile(img, (2, 98))
    print(f"  Image intensity range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  Middle 96% of pixels lie in: [{p2:.3f}, {p98:.3f}]")
    print(f"  Std-dev (contrast proxy): {img.std():.3f}  — low means washed-out")
    print("\n  Nearly all pixels sit in a narrow band → the image looks flat/hazy.")

    # ── 2.2  Global histogram equalization ───────────────────────────────────
    subsection("2.2  Global histogram equalization")

    eq = exposure.equalize_hist(img)
    print("  equalize_hist remaps intensities so the CUMULATIVE histogram becomes")
    print("  a straight line — every brightness level ends up equally common.\n")
    show_table(
        ["Image", "Min", "Max", "Mean", "Std (contrast)"],
        [
            ["Original",       f"{img.min():.3f}", f"{img.max():.3f}", f"{img.mean():.3f}", f"{img.std():.3f}"],
            ["Equalized",      f"{eq.min():.3f}",  f"{eq.max():.3f}",  f"{eq.mean():.3f}",  f"{eq.std():.3f}"],
        ],
        col_width=16,
    )
    print("\n  Std-dev jumps → contrast expanded to use the full range.")

    # ── 2.3  Adaptive equalization (CLAHE) ───────────────────────────────────
    subsection("2.3  CLAHE — adaptive, local equalization")

    clahe = exposure.equalize_adapthist(img, clip_limit=0.03)
    print("  Global equalization uses ONE histogram for the whole image, so a")
    print("  bright region can wash out a dark one. CLAHE equalizes small tiles")
    print("  independently and clips the histogram to limit noise amplification.")
    print(f"\n  CLAHE contrast (std): {clahe.std():.3f}  — local detail preserved.")

    # ── 2.4  Contrast stretching and gamma ───────────────────────────────────
    subsection("2.4  Contrast stretching and gamma correction")

    stretched = exposure.rescale_intensity(img, in_range=(p2, p98))
    gamma_dark   = exposure.adjust_gamma(img, gamma=2.0)   # γ>1 darkens
    gamma_bright = exposure.adjust_gamma(img, gamma=0.5)   # γ<1 brightens

    print("  Contrast stretch: linearly map the 2nd–98th percentile to [0,1]")
    print("  (robust to outliers). Gamma: a NON-linear tone curve, out = in^γ.\n")
    show_table(
        ["Transform", "Mean", "Std", "Effect"],
        [
            ["Percentile stretch", f"{stretched.mean():.3f}",    f"{stretched.std():.3f}",    "linear, robust"],
            ["Gamma 2.0",          f"{gamma_dark.mean():.3f}",   f"{gamma_dark.std():.3f}",   "darker midtones"],
            ["Gamma 0.5",          f"{gamma_bright.mean():.3f}", f"{gamma_bright.std():.3f}", "brighter midtones"],
        ],
        col_width=19,
    )
    print(textwrap.dedent("""
      When to use which:
      • Contrast stretch — quick, safe, linear; good default.
      • Histogram eq.    — maximises global contrast; can over-amplify noise.
      • CLAHE            — best for uneven lighting (medical, satellite imagery).
      • Gamma            — matches display/perceptual response; brightening or
                           darkening without clipping the extremes.
    """))

    # ── figure ───────────────────────────────────────────────────────────────
    variants = [
        ("Original",       img),
        ("Global equalize", eq),
        ("CLAHE",          clahe),
        ("Gamma 0.5",      gamma_bright),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for col, (title, im) in enumerate(variants):
        axes[0, col].imshow(im, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(title)
        axes[0, col].axis("off")
        axes[1, col].hist(im.ravel(), bins=60, color="#4c72b0")
        # overlay the cumulative distribution
        cdf, bins = exposure.cumulative_distribution(im, nbins=60)
        ax2 = axes[1, col].twinx()
        ax2.plot(bins, cdf, color="#c44e52", lw=1.5)
        ax2.set_ylim(0, 1); ax2.set_yticks([])
        axes[1, col].set_yticks([])
    axes[1, 0].set_ylabel("count")
    fig.suptitle("Histogram equalization & contrast (blue=histogram, red=CDF)",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, "02_histogram_contrast.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 3 — DATA AUGMENTATION STRATEGIES
# ════════════════════════════════════════════════════════════════════════════

def _augment(img, rng, np, transform, util):
    """Apply one randomly chosen label-preserving augmentation."""
    choice = rng.integers(0, 6)
    if choice == 0:
        return np.fliplr(img), "horizontal flip"
    if choice == 1:
        angle = rng.uniform(-25, 25)
        return transform.rotate(img, angle, mode="reflect"), f"rotate {angle:+.0f}°"
    if choice == 2:
        tx, ty = rng.uniform(-15, 15, size=2)
        tf = transform.AffineTransform(translation=(tx, ty))
        return transform.warp(img, tf.inverse, mode="reflect"), f"shift ({tx:+.0f},{ty:+.0f})"
    if choice == 3:
        shear = rng.uniform(-0.3, 0.3)
        tf = transform.AffineTransform(shear=shear)
        return transform.warp(img, tf.inverse, mode="reflect"), f"shear {shear:+.2f}"
    if choice == 4:
        return util.random_noise(img, mode="gaussian", var=0.01, rng=rng), "gaussian noise"
    scale = rng.uniform(1.05, 1.35)
    return transform.rescale(img, scale, channel_axis=-1 if img.ndim == 3 else None,
                             anti_aliasing=True)[:img.shape[0], :img.shape[1]], f"zoom ×{scale:.2f}"


def lab3_augmentation():
    require(["scikit-image", "numpy", "matplotlib"])

    import numpy as np
    from skimage import data, transform, util, img_as_float
    plt = get_plt()

    section("3 — DATA AUGMENTATION STRATEGIES")
    print(textwrap.dedent("""
      A model only knows the images it was trained on. Augmentation manufactures
      new, plausible training examples by transforming existing ones — teaching
      invariance (a cat is a cat, flipped or dimmed) and fighting overfitting,
      all for free from data you already have.
    """))

    img = img_as_float(data.astronaut())

    # ── 3.1  Geometric transforms ────────────────────────────────────────────
    subsection("3.1  Geometric transforms")

    print("  Change WHERE pixels are — the object's identity is unchanged:\n")
    show_table(
        ["Transform", "What it teaches"],
        [
            ["Horizontal flip", "left/right symmetry (faces, scenes)"],
            ["Rotation ±deg",   "camera tilt invariance"],
            ["Translation",     "object can appear anywhere in frame"],
            ["Shear / affine",  "mild perspective / viewpoint change"],
            ["Scale / crop",    "object distance / zoom invariance"],
        ],
        col_width=18,
    )

    # ── 3.2  Photometric transforms ──────────────────────────────────────────
    subsection("3.2  Photometric transforms")

    print("  Change pixel VALUES — simulate lighting, sensor and weather noise:\n")
    show_table(
        ["Transform", "What it teaches"],
        [
            ["Brightness/gamma", "different exposure / time of day"],
            ["Contrast jitter",  "haze, backlighting"],
            ["Gaussian noise",   "sensor / ISO noise robustness"],
            ["Blur",             "motion / out-of-focus tolerance"],
            ["Colour jitter",    "white-balance / camera differences"],
        ],
        col_width=18,
    )

    # ── 3.3  The label-preserving rule ───────────────────────────────────────
    subsection("3.3  The golden rule — stay label-preserving")

    print(textwrap.dedent("""
      Augmentation must not change the correct answer:
      • Vertically flipping a '6' turns it into a '9' — WRONG for digit OCR.
      • Flipping a photo of a car horizontally — fine.
      • Rotating a chest X-ray 90° — anatomically impossible, hurts the model.
      Choose transforms that match the REAL variation your data will see.
    """))

    # ── 3.4  A random augmentation pipeline ──────────────────────────────────
    subsection("3.4  Generating augmented variants (seeded, reproducible)")

    rng = np.random.default_rng(42)
    n = 11
    variants = [_augment(img, rng, np, transform, util) for _ in range(n)]
    print(f"  Generated {n} random variants from ONE source image:")
    for _, name in variants:
        print(f"      • {name}")
    print("\n  In training, a fresh random augmentation is applied every epoch, so")
    print("  the model effectively never sees the exact same image twice.")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    axes = axes.ravel()
    axes[0].imshow(img); axes[0].set_title("ORIGINAL", fontweight="bold")
    for ax, (im, name) in zip(axes[1:], variants):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("One image → many training examples", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "03_augmentation.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 4 — FEATURE EXTRACTION BASICS (EDGES, TEXTURES, HOG)
# ════════════════════════════════════════════════════════════════════════════

def lab4_features():
    require(["scikit-image", "numpy", "matplotlib"])

    import numpy as np
    from skimage import data, filters, feature, img_as_float, img_as_ubyte
    plt = get_plt()

    section("4 — FEATURE EXTRACTION BASICS (EDGES, TEXTURES, HOG)")
    print(textwrap.dedent("""
      Before deep learning learned features automatically, vision relied on
      HAND-CRAFTED descriptors: numbers summarising edges, texture, and shape.
      They are fast, interpretable, need no training, and still power classical
      pipelines — plus they make what a CNN's first layers learn intuitive.
    """))

    gray = img_as_float(data.camera())   # 512×512 photographer, rich in edges

    # ── 4.1  Edges — gradients ───────────────────────────────────────────────
    subsection("4.1  Edge detection — image gradients")

    sobel = filters.sobel(gray)
    canny = feature.canny(gray, sigma=2.0)
    print("  An EDGE is where intensity changes sharply — a large gradient.\n")
    show_table(
        ["Detector", "Output", "Idea"],
        [
            ["Sobel", "gradient magnitude (float)", "√(Gx²+Gy²) per pixel"],
            ["Canny", "thin binary edge map",       "gradient + non-max suppression + hysteresis"],
        ],
        col_width=24,
    )
    print(f"\n  Sobel response: mean={sobel.mean():.4f}, max={sobel.max():.4f}")
    print(f"  Canny found {canny.sum():,} edge pixels ({100*canny.mean():.1f}% of image)")

    # ── 4.2  Texture — LBP and GLCM ──────────────────────────────────────────
    subsection("4.2  Texture descriptors — LBP and GLCM")

    P, R = 8, 1
    lbp = feature.local_binary_pattern(gray, P, R, method="uniform")
    n_bins = P + 2
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    g = img_as_ubyte(gray)
    glcm = feature.graycomatrix(g, distances=[1], angles=[0], levels=256,
                                symmetric=True, normed=True)
    props = {p: float(feature.graycoprops(glcm, p)[0, 0])
             for p in ["contrast", "homogeneity", "energy", "correlation"]}

    print("  Texture = the spatial PATTERN of intensities (smooth? rough? striped?).\n")
    print(f"  Local Binary Pattern ({n_bins}-bin histogram, rotation-invariant):")
    print("      " + "  ".join(f"{v:.3f}" for v in lbp_hist))
    print("\n  Gray-Level Co-occurrence Matrix (GLCM) summary statistics:")
    for name, val in props.items():
        print(f"      {name:<13} {val:.4f}")
    print(textwrap.dedent("""
      • LBP thresholds each pixel against its neighbours → a code; the histogram
        of codes describes local texture. Great for faces and materials.
      • GLCM counts how often intensity pairs co-occur; contrast/homogeneity/
        energy/correlation summarise coarseness and regularity.
    """))

    # ── 4.3  HOG — shape via gradient orientations ───────────────────────────
    subsection("4.3  Histogram of Oriented Gradients (HOG)")

    hog_vec, hog_img = feature.hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys",
        visualize=True, feature_vector=True)
    print("  HOG bins the gradient DIRECTIONS inside small cells, then block-")
    print("  normalizes for lighting invariance. The result encodes SHAPE.\n")
    print(f"  Feature vector length: {hog_vec.size:,} numbers")
    print("  This fixed-length vector is what you feed a classifier (Lab 5).")
    print(textwrap.dedent("""
      HOG powered the classic pedestrian detector (Dalal & Triggs, 2005). Note
      the progression: edges → texture → HOG shape. A CNN's early layers learn
      strikingly similar edge/texture detectors — the difference is it learns
      them from data instead of us hand-designing them.
    """))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes[0, 0].imshow(gray, cmap="gray");   axes[0, 0].set_title("Original")
    axes[0, 1].imshow(sobel, cmap="magma"); axes[0, 1].set_title("Sobel gradient")
    axes[0, 2].imshow(canny, cmap="gray");  axes[0, 2].set_title("Canny edges")
    axes[1, 0].imshow(lbp, cmap="gray");    axes[1, 0].set_title("LBP codes (texture)")
    axes[1, 1].imshow(hog_img, cmap="gray"); axes[1, 1].set_title("HOG visualization")
    axes[1, 2].bar(range(n_bins), lbp_hist, color="#4c72b0")
    axes[1, 2].set_title("LBP histogram"); axes[1, 2].set_xlabel("code")
    for ax in axes.ravel():
        if ax is not axes[1, 2]:
            ax.axis("off")
    fig.suptitle("Hand-crafted features: edges, texture, HOG", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "04_features.png")


# ════════════════════════════════════════════════════════════════════════════
#  LAB 5 — FULL PIPELINE: PREPROCESS + AUGMENT FOR CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

def lab5_pipeline():
    require(["scikit-image", "scikit-learn", "numpy", "matplotlib"])

    import numpy as np
    from skimage import exposure, transform, util, feature
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    plt = get_plt()

    section("5 — FULL PIPELINE: PREPROCESS + AUGMENT → CLASSIFICATION")
    print(textwrap.dedent("""
      Everything from Labs 1–4 combined into one end-to-end pipeline:

        raw images → normalize + equalize → augment TRAIN set
                   → HOG features → SVM classifier → evaluate
    """))

    # ── 5.1  Load the dataset ────────────────────────────────────────────────
    subsection("5.1  Load the corpus — Olivetti faces")

    print("  Fetching Olivetti faces (downloads a few MB on first run)…")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    images, y = faces.images, faces.target   # (400, 64, 64) float, 40 people
    n_classes = len(np.unique(y))
    print(f"  {len(images)} images · {n_classes} people · "
          f"{images.shape[1]}×{images.shape[2]} grayscale, values in "
          f"[{images.min():.2f}, {images.max():.2f}]")

    # ── 5.2  Stage 1: preprocess (Labs 1–2) ──────────────────────────────────
    subsection("5.2  Stage 1 — normalize + contrast equalize (Labs 1–2)")

    def preprocess(im):
        im = exposure.equalize_hist(im)                       # even out lighting
        return (im - im.mean()) / (im.std() + 1e-8)           # standardize

    print("  Per-image histogram equalization (kill lighting differences) then")
    print("  standardization to zero-mean/unit-variance.")

    # ── 5.3  Split, then augment the TRAIN set only (Lab 3) ───────────────────
    subsection("5.3  Stage 2 — split, then augment TRAIN only (Lab 3)")

    X_tr_img, X_te_img, y_tr, y_te = train_test_split(
        images, y, test_size=0.25, random_state=42, stratify=y)
    print(f"  Train: {len(X_tr_img)} images   Test: {len(X_te_img)} images")
    print("  ⚠ Augment AFTER the split, TRAIN only — augmenting before leaks")
    print("    near-duplicate images across the split and inflates accuracy.\n")

    rng = np.random.default_rng(0)

    def augment_face(im):
        variants = [im, np.fliplr(im)]                        # flip is safe for faces
        variants.append(transform.rotate(im, rng.uniform(-12, 12), mode="edge"))
        variants.append(util.random_noise(im, mode="gaussian", var=0.005, rng=rng))
        return variants

    aug_imgs, aug_y = [], []
    for im, label in zip(X_tr_img, y_tr):
        for v in augment_face(im):
            aug_imgs.append(v)
            aug_y.append(label)
    print(f"  Augmentation expanded train set: {len(X_tr_img)} → {len(aug_imgs)} images (×4)")

    # ── 5.4  Stage 3: HOG features (Lab 4) ────────────────────────────────────
    subsection("5.4  Stage 3 — HOG feature extraction (Lab 4)")

    def to_features(img_list):
        feats = [feature.hog(preprocess(im), orientations=9,
                             pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             block_norm="L2-Hys", feature_vector=True)
                 for im in img_list]
        return np.asarray(feats)

    X_tr_base = to_features(X_tr_img)     # baseline: no augmentation
    X_tr_aug  = to_features(aug_imgs)     # augmented train set
    X_te      = to_features(X_te_img)
    print(f"  HOG feature matrix: {X_tr_aug.shape[1]:,} features per image")
    print(f"  Baseline train: {X_tr_base.shape}   Augmented train: {X_tr_aug.shape}"
          f"   Test: {X_te.shape}")

    # ── 5.5  Stage 4: train classifier — does augmentation help? ──────────────
    subsection("5.5  Stage 4 — SVM classifier (baseline vs augmented)")

    def train_eval(Xtr, ytr):
        clf = SVC(kernel="linear", C=1.0)
        clf.fit(Xtr, ytr)
        return accuracy_score(y_te, clf.predict(X_te)), clf

    acc_base, _        = train_eval(X_tr_base, y_tr)
    acc_aug, clf_aug   = train_eval(X_tr_aug, np.asarray(aug_y))

    print(f"\n  Chance level (1/{n_classes} classes):     {1/n_classes:.1%}")
    print(f"  Accuracy — no augmentation:      {acc_base:.1%}")
    print(f"  Accuracy — with augmentation:    {acc_aug:.1%}")
    delta = acc_aug - acc_base
    verdict = "helped" if delta > 0 else ("no change" if delta == 0 else "hurt")
    print(f"  Augmentation {verdict}: {delta:+.1%}")

    # ── 5.6  Detailed report ─────────────────────────────────────────────────
    subsection("5.6  Classification report (augmented model, first 6 people)")

    preds = clf_aug.predict(X_te)
    labels_subset = sorted(np.unique(y_te))[:6]
    print(textwrap.indent(
        classification_report(y_te, preds, labels=labels_subset,
                              target_names=[f"person {i}" for i in labels_subset],
                              zero_division=0), "  "))

    print(textwrap.dedent(f"""
      A classical pipeline — equalize → augment → HOG → linear SVM — reaches
      {acc_aug:.0%} on 40-way face recognition with only a handful of images
      per person. Augmentation adds robustness for free; HOG turns pixels into
      a shape descriptor a linear model can separate.

      Next step: let a CNN LEARN the features instead of hand-crafting HOG —
      but the preprocessing and augmentation in this tutorial stay exactly the
      same in front of any deep model.
    """))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 6, figsize=(14, 8))
    # row 0: sample raw faces
    for j in range(6):
        axes[0, j].imshow(X_tr_img[j], cmap="gray"); axes[0, j].axis("off")
    axes[0, 0].set_ylabel("raw", rotation=0, labelpad=25);
    # row 1: augmented variants of the first face
    for j, v in enumerate(augment_face(X_tr_img[0])[:4]):
        axes[1, j].imshow(np.clip(v, 0, 1), cmap="gray"); axes[1, j].axis("off")
    for j in range(4, 6):
        axes[1, j].axis("off")
    # row 2: HOG visualization of a few faces
    for j in range(6):
        _, hog_img = feature.hog(preprocess(X_te_img[j]), orientations=9,
                                 pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                 visualize=True, feature_vector=True)
        axes[2, j].imshow(hog_img, cmap="gray"); axes[2, j].axis("off")
    axes[0, 0].set_title("raw faces →", loc="left", fontsize=10)
    axes[1, 0].set_title("augmented (1 face) →", loc="left", fontsize=10)
    axes[2, 0].set_title("HOG features →", loc="left", fontsize=10)
    fig.suptitle(f"Face pipeline  (baseline {acc_base:.0%} → augmented {acc_aug:.0%})",
                 fontsize=13)
    fig.tight_layout()
    save_fig(fig, "05_pipeline.png")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Image Preprocessing Tutorial")
    parser.add_argument(
        "--lab", type=int, choices=[1, 2, 3, 4, 5],
        help="Run a specific lab (1=Colour/Resize/Normalize, 2=Histogram/Contrast, "
             "3=Augmentation, 4=Edges/Texture/HOG, 5=Full pipeline)"
    )
    args = parser.parse_args()

    print("\n" + "█" * 70)
    print("  IMAGE PREPROCESSING — FROM PIXELS TO FEATURES  ")
    print("█" * 70)
    print("""
  Labs:
    1 → Colour Spaces, Resizing, Normalization   (scikit-image)
    2 → Histogram Equalization & Contrast         (scikit-image)
    3 → Data Augmentation Strategies              (scikit-image)
    4 → Feature Extraction: Edges, Texture, HOG   (scikit-image)
    5 → Full Pipeline: preprocess + augment → classify  (Olivetti faces, SVM)

  Every lab saves a figure to ./outputs/image_preprocessing/
    """)

    lab_map = {
        1: lab1_color_resize_normalize,
        2: lab2_histogram_contrast,
        3: lab3_augmentation,
        4: lab4_features,
        5: lab5_pipeline,
    }

    if args.lab is not None:
        lab_map[args.lab]()
    else:
        for lab in lab_map.values():
            lab()

    print(f"\n  All figures saved under {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
