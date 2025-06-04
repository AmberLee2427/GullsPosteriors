import numpy as np
import VBMicrolensing

def magnification(self, ss, q, u1, u2, rho, eps=1e-4, gamma=0.36):
    """Return the binary-lens magnification using :mod:`VBMicrolensing`.

    Parameters
    ----------
    ss : array_like
        Lens separation for each epoch in units of ``theta_E``.
    q : float
        Mass ratio ``m_2/m_1`` of the lens system.
    u1 : array_like
        Source position relative to the primary lens for each epoch.
    u2 : array_like
        Source position relative to the secondary lens for each epoch.
    rho : float
        Angular radius of the source in units of ``theta_E``.
    eps : float, optional
        Relative tolerance passed to :mod:`VBMicrolensing`. Default ``1e-4``.
    gamma : float, optional
        Limb-darkening coefficient ``a1`` for :mod:`VBMicrolensing`.

    Returns
    -------
    ndarray
        Magnification for each value of ``ss``.
    """

    if self.mag_obj is None:
        self.mag_obj = VBMicrolensing.VBMicrolensing()

    self.mag_obj.a1 = gamma
    self.mag_obj.RelTol = eps

    mag = np.zeros_like(ss)

	#print('ss', 'q', 'u1', 'u2', 'rho')
    for i in range(len(ss)):
		#print(ss[i], q, u1[i], u2[i], rho)
        mag[i] = self.mag_obj.BinaryMag2(ss[i], q, u1[i], u2[i], rho)

    return np.array(mag)

