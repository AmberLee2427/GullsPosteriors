""" depricated: use VBMicrolensing instead of VBBinaryLensing for binary lens magnifications"""

import numpy as np
try:
    import VBBinaryLensing
except ImportError as e:
    print(f"Warning: VBBinaryLensing module not available: {e}")
    print("Magnification calculations will fail. Ensure VBBinaryLensing is properly installed.")
    VBBinaryLensing = None


def magnification(self, ss, q, u1, u2, rho, eps=1e-4):
    """Return the binary-lens magnification for an array of separations.

    Parameters
    ----------
    ss : array_like
        Lens separation for each observation epoch in units of ``theta_E``.
    q : float
        Mass ratio of the binary lens ``m_2/m_1``.
    u1 : array_like
        Source-lens separation from the primary for each epoch.
    u2 : array_like
        Source-lens separation from the secondary for each epoch.
    rho : float
        Angular source radius in units of ``theta_E``.
    eps : float, optional
        Relative tolerance used by :mod:`VBBinaryLensing`. Default is ``1e-4``.

    Returns
    -------
    ndarray
        Magnification for each element of ``ss``.
        
    Raises
    ------
    ImportError
        If VBBinaryLensing module is not available.
    RuntimeError
        If magnification calculation fails.
        
    Notes
    -----
    The limb-darkening coefficient gamma is read from self.gamma (loaded from the .prm file).
    """

    # FAIL FAST: If VBBinaryLensing is missing, this is a fatal configuration error
    if VBBinaryLensing is None:
        raise ImportError("VBBinaryLensing module is required but not available. Install with: pip install VBBinaryLensing")

    if self.mag_obj is None:
        self.mag_obj = VBBinaryLensing.VBBinaryLensing()

    self.mag_obj.RelTol = eps
    self.mag_obj.a1 = self.gamma

    mag = np.zeros_like(ss)

    try:
        for i in range(len(ss)):
            mag[i] = self.mag_obj.BinaryMag2(ss[i], q, u1[i], u2[i], rho)
        
        return np.array(mag)
    except Exception as e:
        raise RuntimeError(f"VBBinaryLensing calculation failed: {e}. This indicates a serious problem with the magnification calculation.")
