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
    
    Raises
    ------
    ValueError
        If any parameter values are physically invalid.
    """
    
    # Parameter validation - catch stupid values before they hang VBM
    if q <= 0 or q > 10:
        raise ValueError(f"Mass ratio q={q} is invalid (must be 0 < q <= 10)")
    
    if rho <= 0 or rho > 10:
        raise ValueError(f"Source radius rho={rho} is invalid (must be 0 < rho <= 1)")
    
    if np.any(ss <= 0) or np.any(ss > 1000):
        raise ValueError(f"Separation ss has invalid values (must be 0 < ss <= 1000): min={np.min(ss)}, max={np.max(ss)}")

    if eps <= 0 or eps > 1e-2:
        raise ValueError(f"Tolerance eps={eps} is invalid (must be 0 < eps <= 0.01)")
    
    if gamma < 0 or gamma > 1:
        raise ValueError(f"Limb darkening gamma={gamma} is invalid (must be 0 <= gamma <= 1)")

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

