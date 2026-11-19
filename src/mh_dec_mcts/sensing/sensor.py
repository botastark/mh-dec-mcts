import numpy as np

def sigma(x: float) -> float:
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

def get_sensor_params(altitude: float, conf_dict=None):
    """
    Return (s0, s1) sensor noise parameters for a given altitude.

    Lookup order:
    1. Exact key in conf_dict (rounded to 2 decimals)
    2. Nearest key in conf_dict
    3. Analytic formula: sigma = a * (1 - exp(-b * altitude))

    This is the single authoritative implementation used by all planners.
    """
    if conf_dict is not None and conf_dict != {}:
        key = np.round(altitude, decimals=2)
        if key in conf_dict:
            return conf_dict[key]
        try:
            keys = np.array(list(conf_dict.keys()), dtype=float)
            idx = np.argmin(np.abs(keys - altitude))
            return conf_dict[keys[idx]]
        except Exception:
            pass
    a, b = 1.0, 0.015
    sigma = a * (1.0 - np.exp(-b * altitude))
    return sigma, sigma