import math


class OneEuroFilter:
    # Smooths a value that changes over time: heavy smoothing while it moves slowly (no jitter),
    # light smoothing while it moves fast (no lag). See https://gery.casiez.net/1euro/
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx = a_d * (x - self.x_prev) / dt + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * abs(dx), dt)
        x = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x, dx, t
        return x
