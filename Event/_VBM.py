import numpy as np
try:
    import VBMicrolensing
except ImportError as e:
    print(f"Warning: VBMicrolensing module not available: {e}")
    print("Magnification calculations will fail. Ensure VBMicrolensing is properly installed.")
    VBMicrolensing = None
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("VBMicrolensing calculation timed out")

def magnification(self, ss, q, u1, u2, rho, eps=1e-4, timeout=300):
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
    timeout : float, optional
        Maximum time in seconds for VBMicrolensing calculation. Default 300 (5 minutes).

    Returns
    -------
    ndarray
        Magnification for each value of ``ss``.
    
    Raises
    ------
    ImportError
        If VBMicrolensing module is not available.
    ValueError
        If any parameter values are physically invalid.
    RuntimeError
        If magnification calculation fails or times out.
        
    Notes
    -----
    The limb-darkening coefficient gamma is read from self.gamma (loaded from the .prm file).
    """
    
    # FAIL FAST: If VBMicrolensing is missing, this is a fatal configuration error
    if VBMicrolensing is None:
        raise ImportError("VBMicrolensing module is required but not available. Install with: pip install VBMicrolensing")
    
    # Parameter validation - catch stupid values before they hang VBM
    if q <= 0 or q > 10:
        raise ValueError(f"Mass ratio q={q} is invalid (must be 0 < q <= 10)")
    
    if rho <= 0 or rho > 10:
        raise ValueError(f"Source radius rho={rho} is invalid (must be 0 < rho <= 1)")
    
    if np.any(ss <= 0) or np.any(ss > 1000):
        raise ValueError(f"Separation ss has invalid values (must be 0 < ss <= 1000): min={np.min(ss)}, max={np.max(ss)}")

    if eps <= 0 or eps > 1e-2:
        raise ValueError(f"Tolerance eps={eps} is invalid (must be 0 < eps <= 0.01)")
    
    if self.gamma < 0 or self.gamma > 1:
        raise ValueError(f"Limb darkening gamma={self.gamma} is invalid (must be 0 <= gamma <= 1)")

    if self.mag_obj is None:
        self.mag_obj = VBMicrolensing.VBMicrolensing()

    if self.gamma is None:
        raise ValueError('Limb darkening gamma must be set before calling magnification.')

    # Force linear limb darkening profile when available
    if hasattr(self.mag_obj, 'SetLDprofile') and hasattr(self.mag_obj, 'LDprofiles'):
        try:
            self.mag_obj.SetLDprofile(self.mag_obj.LDprofiles.LDlinear)
        except AttributeError:
            pass

    self.mag_obj.a1 = self.gamma
    try:
        self.mag_obj.a2 = 0.0
    except AttributeError:
        pass
    self.mag_obj.RelTol = eps

    mag = np.zeros_like(ss)

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        #print('ss', 'q', 'u1', 'u2', 'rho')
        for i in range(len(ss)):
            #print(ss[i], q, u1[i], u2[i], rho)
            mag[i] = self.mag_obj.BinaryMag2(ss[i], q, u1[i], u2[i], rho)
        
        # Clear the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        return np.array(mag)
        
    except TimeoutError:
        raise RuntimeError(f"VBMicrolensing calculation timed out after {timeout} seconds. This suggests the parameters may be problematic or the system is overloaded.")
    except Exception as e:
        # Clear alarm on any other exception and fail hard
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise RuntimeError(f"VBMicrolensing calculation failed: {e}. This indicates a serious problem with the magnification calculation.")  

