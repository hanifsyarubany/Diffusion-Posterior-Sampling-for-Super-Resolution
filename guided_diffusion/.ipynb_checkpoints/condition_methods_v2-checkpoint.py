from abc import ABC, abstractmethod
import torch

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs): 
        # We always pass "noisy_measurement" as the measurement argument of A.
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        """
        Compute gradient and value of || y - A(x_0_hat) ||_2
        with a safe fallback for super-resolution (different spatial sizes).
        """
    
        # ----- safe forward operator A(x_0_hat) -----
        # If measurement has different H,W than x_0_hat (super-resolution),
        # avoid calling self.operator.forward (Resizer) to sidestep CUDA asserts
        if (
            measurement is not None
            and measurement.ndim == x_0_hat.ndim
            and measurement.shape[-2:] != x_0_hat.shape[-2:]
        ):
            # Super-resolution case: downsample/resize x_0_hat directly
            # to measurement size using a differentiable interpolator.
            pred_meas = torch.nn.functional.interpolate(
                x_0_hat,
                size=measurement.shape[-2:],   # (H_meas, W_meas)
                mode="bicubic",
                align_corners=False,
            )
        else:
            # Same spatial size (e.g., inpainting, deblurring, denoising)
            # Use the operator as originally implemented.
            pred_meas = self.operator.forward(x_0_hat, **kwargs)
    
        # ----- Gaussian noise case -----
        if self.noiser.__name__ == "gaussian":
            # This keeps the original DPS sign: (y - A(x̂0))
            difference = measurement - pred_meas
    
            # L2 norm over all pixels and batch
            norm = torch.linalg.norm(difference)
    
            # ∇_{x_prev} || y - A(x̂0) ||_2
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            return norm_grad, norm
    
        # ----- Poisson noise case (optional, just keep it robust) -----
        elif self.noiser.__name__ == "poisson":
            eps = 1e-6
            pred = pred_meas.clamp_min(eps)
            # Negative log-likelihood (up to additive constant)
            value = (pred - measurement * torch.log(pred)).sum()
            grad = torch.autograd.grad(outputs=value, inputs=x_prev)[0]
            return grad, value
    
        else:
            raise NotImplementedError(
                f"Noise type '{self.noiser.__name__}' not supported in grad_and_value"
            )

    
    @abstractmethod
    def conditioning(self, *args, **kwargs):
        """
        Must return (x_t_new, distance_scalar).
        """
        pass


@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # Just pass the input without conditioning (for ablations).
    def conditioning(self, x_t, **kwargs):
        dummy = torch.zeros(1, device=x_t.device)
        return x_t, dummy
        

@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self,
                     x_t,
                     noisy_measurement=None,
                     measurement=None,
                     x_prev=None,
                     x_0_hat=None,
                     **kwargs):
        """
        Hard projection onto the measurement manifold using A:

          x_t_new = P(x_t; noisy_measurement)

        implemented via operator.project.
        """
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)

        # If we have a measurement and an x_0_hat prediction, report a distance.
        if (measurement is not None) and (x_0_hat is not None):
            with torch.no_grad():
                pred_meas = self.operator.forward(x_0_hat, **kwargs)
                diff = pred_meas - measurement
                norm = diff.view(diff.shape[0], -1).norm(dim=1).mean()
        else:
            norm = torch.zeros(1, device=x_t.device)

        return x_t, norm


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self,
                     x_prev,
                     x_t,
                     x_0_hat,
                     measurement,
                     noisy_measurement,
                     **kwargs):
        """
        MCG: gradient step + hard projection.

          x_t <- x_t - scale * ∂L/∂x_prev
          x_t <- P(x_t; noisy_measurement)
        """
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev,
            x_0_hat=x_0_hat,
            measurement=measurement,
            **kwargs
        )
        x_t = x_t - norm_grad * self.scale
        
        # followed by a hard projection step
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        

@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        """
        Classic DPS-style posterior sampling (no projection).
        """
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev,
            x_0_hat=x_0_hat,
            measurement=measurement,
            **kwargs
        )
        # DPS step: move along negative gradient (reduce mismatch)
        x_t = x_t - self.scale * norm_grad
        return x_t, norm


###################
# Additional conditioning methods for SR
###################

@register_conditioning_method(name='soft_projection')
class SoftProjection(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        # alpha in [0,1]: 0=no conditioning, 1=hard projection
        self.alpha = kwargs.get('alpha', 0.5)

    def conditioning(self,
                     x_prev,          # kept for API compatibility
                     x_t,
                     x_0_hat,
                     measurement,
                     noisy_measurement,
                     **kwargs):
        """
        Soft projection:

          x_t <- (1 - alpha) * x_t + alpha * P(x_t),

        where P is the measurement projection operator.
        """
        # 1) project current sample onto measurement manifold
        projected = self.project(
            data=x_t,
            noisy_measurement=noisy_measurement,
            **kwargs
        )

        # 2) convex combination between original and projected
        x_t_new = (1.0 - self.alpha) * x_t + self.alpha * projected

        # 3) compute a distance for logging (like ps/mcg)
        with torch.no_grad():
            pred_meas = self.operator.forward(x_0_hat, **kwargs)
            diff = pred_meas - measurement
            distance = diff.view(diff.shape[0], -1).norm(dim=1).mean()

        return x_t_new, distance


@register_conditioning_method(name='ps_annealed')
class PosteriorSamplingAnnealed(ConditioningMethod):
    """
    PosteriorSampling (DPS-style) with an *implicit* annealed guidance strength
    suitable for super-resolution.

    We do NOT depend on timestep `t` explicitly (since the sampler doesn't
    pass it into the conditioning method). Instead we estimate "progress"
    using an SNR-like statistic based on (x_t, x_0_hat):

        x_t      = noisy sample at timestep t
        x_0_hat  = network prediction of x_0 at this timestep

    Define:
        signal_norm = ||x_0_hat||_2
        noise_norm  = ||x_t - x_0_hat||_2  (approx. diffusion noise level)

    Then:
        snr = signal_norm / noise_norm
        progress \in [0,1] is snr normalised within the batch.

    Guidance scale is annealed as:
        scale_eff = min_scale + progress * (base_scale - min_scale)

    So early steps (low SNR, very noisy) use weak guidance,
    while late steps (high SNR, denoised) use strong guidance.
    """
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        # base_scale: maximum guidance strength (late timesteps)
        self.base_scale = kwargs.get('scale', 1.0)
        # min_scale: minimal guidance at the earliest, noisiest steps
        self.min_scale = kwargs.get('min_scale', 0.0)

    def _compute_progress(self, x_prev, x_0_hat):
        """
        Estimate diffusion progress in [0,1] from an SNR-like statistic.
        """
        with torch.no_grad():
            # Flatten to [B, -1]
            x0_flat = x_0_hat.view(x_0_hat.shape[0], -1)
            xt_flat = x_prev.view(x_prev.shape[0], -1)

            noise_flat = xt_flat - x0_flat
            signal_norm = x0_flat.norm(dim=1)          # ||x_0_hat||
            noise_norm = noise_flat.norm(dim=1)        # ||x_t - x_0_hat||

            snr = signal_norm / (noise_norm + 1e-8)    # > 0
            # Normalise SNR within batch to [0,1]
            snr_norm = snr / (snr.max() + 1e-8)
            progress = snr_norm.mean()
            progress = progress.clamp(0.0, 1.0)

        return progress

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        # 1) Compute gradient of data term
        norm_grad, norm = self.grad_and_value(
            x_prev=x_prev,
            x_0_hat=x_0_hat,
            measurement=measurement,
            **kwargs
        )

        # 2) Compute annealed guidance strength
        progress = self._compute_progress(x_prev, x_0_hat)
        effective_scale = self.min_scale + progress * (self.base_scale - self.min_scale)

        # 3) Apply annealed DPS update
        x_t = x_t - effective_scale * norm_grad

        return x_t, norm
