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

    # wavenumbers in cycles per `ref_cells` cells on both axes (round clusters on rectangles).
    # ref_cells = None: the longest side (cluster size grows with the field). Waves longer than
    # ref_cells (k < 1) get the k = 1 amplitude, so clusters keep the same physical size on any
    # field. Square field with ref_cells = its side: exactly the original.
    n = ref_cells or max(n_cell_x, n_cell_y)
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

def load_annotations(path: str, shape: tuple[int, int], crop=None) -> np.ndarray:             # labels {2,10}→1, crop, check values ⊆ {0,1}
    """
    Load annotations from a .txt file, convert to binary matrix and optionally crop the array.
    """
    try:
        matrix = np.loadtxt(path, dtype=int)
        matrix[matrix == 10] = 1
        matrix[matrix == 2] = 1
        assert matrix.all() in [
            0,
            1,
        ], "Invalid label values found in the annotation file"
        if crop is not None:

            matrix = matrix[3:113, 13:73]
        assert (
            matrix.shape == shape
        ), "Grid size mismatch with annotation matrix"
        return matrix
    except Exception as e:
        raise ValueError(f"Error reading the file {path}: {e}")


