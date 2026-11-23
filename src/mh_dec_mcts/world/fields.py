# ground truth: Gaussian or orthomap annotations
import numpy as np

def gaussian_random_field(r: float, shape: tuple[int, int], rng: np.random.Generator,
                          ref_cells: int | None = None) -> np.ndarray:
    """
    Generate a 2D Gaussian random field and cache the results for reuse.
     https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/
    Parameters:
    - cluster_radius: Correlation radius for the Gaussian field.
    - n_cell: Size of the field (n_cell_x x n_cell_y).
    - seed: Random seed for reproducibility (default: None).

    - cache_dir: Directory to store cached fields (default: "cache").
    - ref_cells: Outer scale L0 in cells (config `field.ref` / cell). Wavenumbers are counted in
               cycles per ref_cells cells on both axes, so clusters are round on rectangles, and
               waves longer than ref_cells (|k| < 1) get the |k| = 1 amplitude, so clusters keep
               the same physical size on any field. None: the shortest side (cluster size then
               grows with the field). Square field with ref_cells = its side: exactly the original.

    Returns:
    - 2D binary random field as a numpy array.
    """

    # Ensure cache directory exists
    n_cell_x, n_cell_y = shape

    # Helper functions
    def _fft_indices(n):
        a = list(range(0, int(np.floor(n / 2)) + 1))
        b = reversed(range(1, (n + 1) // 2))  # was floor(n / 2): one index short for odd n
        b = [-i for i in b]
        return a + b

    n = ref_cells or min(n_cell_x, n_cell_y)
    sx, sy = n / n_cell_x, n / n_cell_y

    def _pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        kx, ky = kx * sx, ky * sy
        val = np.sqrt(max(1.0, np.sqrt(kx**2 + ky**2)) ** (-r))
        return val

    # Generate amplitude for the given cluster_radius
    amplitude = np.zeros((n_cell_x, n_cell_y))
    fft_indices_x = _fft_indices(n_cell_x)
    fft_indices_y = _fft_indices(n_cell_y)

    for i, kx in enumerate(fft_indices_x):
        for j, ky in enumerate(fft_indices_y):
            amplitude[i, j] = _pk2(kx, ky)

    # Generate Gaussian random field
    noise = np.fft.fft2(rng.normal(size=shape))
    random_field = np.fft.ifft2(noise * amplitude).real
    normalized_random_field = (random_field - np.min(random_field)) / (
        np.max(random_field) - np.min(random_field)
    )

    # Make field binary
    normalized_random_field[normalized_random_field >= 0.5] = 1
    normalized_random_field[normalized_random_field < 0.5] = 0

    binary_field = normalized_random_field.astype(np.uint8)

    return binary_field

def load_annotations(path: str, crop=None, labels=None) -> np.ndarray:             # labels {2,10}→1, crop, check values ⊆ {0,1}
    """
    Load annotations from a .txt file, convert to binary matrix and optionally crop the array.

     labels  {raw label: 0 or 1}, the mapping to the binary classes (config `field.labels`);
            None = the file must already be binary. Any label not in the mapping is an error.
            Wheat orthomap (AGR-o-RAMA/active_sensing): {0: 0, 1: 1, 2: 1, 10: 1},
            class 0 = lodged, 1 = not lodged.
    crop    (r0, r1, c0, c1) keeps rows r0:r1 and columns c0:c1 (wheat orthomap: 3, 113, 13, 73).

    """


    labels = {0: 0, 1: 1} if labels is None else {int(k): int(v) for k, v in labels.items()}
    if not set(labels.values()) <= {0, 1}:
        raise ValueError(f"field.labels must map to 0 or 1, got {labels}")
    m = np.loadtxt(path, dtype=int)
    unknown = np.setdiff1d(np.unique(m), list(labels))
    if unknown.size:
        raise ValueError(f"{path}: labels {unknown.tolist()} have no meaning in field.labels {labels}")
    if crop is not None:
        r0, r1, c0, c1 = crop
        m = m[r0:r1, c0:c1]
    lut = np.zeros(max(labels) + 1, np.uint8)
    lut[list(labels)] = list(labels.values())
    return lut[m]
 

