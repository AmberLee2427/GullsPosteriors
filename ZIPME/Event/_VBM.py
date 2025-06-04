import numpy as np
import VBMicrolensing


def magnification(self, ss, q, u1, u2, rho, eps=1e-4, gamma=0.36):

    if self.mag_obj is None:
        self.mag_obj = VBMicrolensing.VBMicrolensing()

    self.mag_obj.a1 = gamma
    self.mag_obj.RelTol = eps

    mag = np.zeros_like(ss)

    for i in range(len(ss)):
        mag[i] = self.mag_obj.BinaryMag2(ss[i], q, u1[i], u2[i], rho)

    return np.array(mag)
