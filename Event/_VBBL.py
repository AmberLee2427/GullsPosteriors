import numpy as np
import VBBinaryLensing


def magnification(self, ss, q, u1, u2, rho, eps=1e-4, gamma=0.36):
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
    gamma : float, optional
        Limb-darkening coefficient ``a1`` passed to :mod:`VBBinaryLensing`.

    Returns
    -------
    ndarray
        Magnification for each element of ``ss``.
    """

    if self.mag_obj is None:
        self.mag_obj = VBBinaryLensing.VBBinaryLensing()

    self.mag_obj.RelTol = eps
    self.mag_obj.a1 = gamma

    mag = np.zeros_like(ss)

    for i in range(len(ss)):
        mag[i] = self.mag_obj.BinaryMag2(ss[i], q, u1[i], u2[i], rho)

    return np.array(mag)
