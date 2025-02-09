import numpy as np
from scipy.integrate import quad
from scipy.stats import sem, entropy, norm
from sympy import N, symbols, lambdify
from abc import ABC, abstractmethod

class UTIMConfig:
    """Hardware-resilient configuration with uncertainty margins."""
    def __init__(self, precision='float128', num_runs=21, human_review_buffer=0.01):
        """
        Parameters:
        - precision: Floating-point precision to use ('float128' or 'float64' fallback)
        - num_runs: Number of evaluations to improve statistical stability
        - human_review_buffer: Safety margin for human review threshold
        """
        self.precision = self._validate_precision(precision)
        self.uncertainty_margin = 10 * np.finfo(self.precision).eps
        self.num_runs = num_runs
        self.human_review_buffer = human_review_buffer

    def _validate_precision(self, prec):
        if prec == 'float128' and not hasattr(np, 'float128'):
            print("Warning: float128 not available, using float64")
            return np.float64
        return getattr(np, prec)

class UniversalTIM(ABC):
    """Universal framework for computing Theory Incoherence Measure (U-TIM)."""
    def __init__(self, model_func, reference_func, posterior_sampler, weight_func, config=UTIMConfig()):
        """
        Hardware-aware implementation preserving equation structure.
        """
        # Core mathematical components
        self.f_i = model_func
        self.f_r = reference_func
        self.p_θ_D = posterior_sampler
        self.w = weight_func

        # Numerical hardening
        self.config = config
        self.dtype = config.precision
        self._symbolic_validation = SymbolicValidator(model_func, reference_func)

    @abstractmethod
    def dC_dt(self, x, theta):
        """Preserved temporal coherence derivative."""
        pass

    def _compute_utim_single(self):
        """Original equation implementation with precision control."""
        θ_samples = self.p_θ_D(1000).astype(self.dtype)
        log_probs = np.array([self.p_θ_D.log_prob(θ) for θ in θ_samples], dtype=self.dtype)

        # Apply log-sum-exp trick for numerical stability
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= np.sum(probs)

        # Compute entropy safely using float64 to ensure compatibility
        H = max(entropy(probs.astype(np.float64)), np.finfo(self.dtype).eps)

        # Symbolic validation for critical components
        self._symbolic_validation.check_consistency()

        # Precision-controlled integration
        expectations = []
        for θ in θ_samples:
            integral = quad(lambda x: float(self._integrand(x, θ)), -np.inf, np.inf, epsabs=1e-6, epsrel=1e-3)[0]
            expectations.append(integral)

        return np.mean(expectations, dtype=self.dtype) / H

    def compute_utim(self):
        """Hardware-resilient computation through multiple runs."""
        results = [self._compute_utim_single() for _ in range(self.config.num_runs)]

        score = np.mean(results, dtype=self.dtype)
        uncertainty = sem(np.array(results, dtype=np.float64))  # Standard error of the mean

        return HardwareResilientResult(score, uncertainty, self.config)

class HardwareResilientResult:
    """Result container with uncertainty awareness."""
    def __init__(self, score, uncertainty, config):
        self.score = score
        self.uncertainty = uncertainty
        self.config = config

    @property
    def safe_score(self):
        """Apply uncertainty margin."""
        return max(self.score - self.uncertainty - self.config.uncertainty_margin, 0)

    def needs_human_review(self, threshold):
        """Determine if the result falls into the human review margin."""
        return (self.score < threshold + self.config.human_review_buffer) and \
               (self.safe_score >= threshold)

class SymbolicValidator:
    """Symbolic math validation for critical components."""
    def __init__(self, model_func, reference_func):
        self.x_sym, self.θ_sym = symbols('x θ')
        self.f_i_sym = model_func(self.x_sym, self.θ_sym)
        self.f_r_sym = reference_func(self.x_sym)

    def check_consistency(self, test_point=0.5):
        """Validate symbolic equivalence at test point."""
        numeric_f_i = lambdify((self.x_sym, self.θ_sym), self.f_i_sym)
        numeric_f_r = lambdify(self.x_sym, self.f_r_sym)

        if not np.isclose(numeric_f_i(test_point, test_point), numeric_f_r(test_point), atol=1e-12):
            print("Symbolic validation warning: Model/Reference mismatch")

class PosteriorSampler:
    def __init__(self, loc=0.5, scale=0.1):
        self.dist = norm(loc, scale)

    def __call__(self, n):
        return self.dist.rvs(n).astype(np.float128)

    def log_prob(self, theta):
        return self.dist.logpdf(theta)

class PhysicsTIM(UniversalTIM):
    def dC_dt(self, x, theta):
        return x**2 * (theta - 0.5)  # Example for physics models

if __name__ == "__main__":
    # Define precision-controlled functions
    def model(x, θ):
        return θ * np.exp(-x**2, dtype=np.float128)

    def reference(x):
        return 0.5 * np.exp(-x**2, dtype=np.float128)

    # Set up the U-TIM configuration
    config = UTIMConfig(
        precision='float128',
        num_runs=21,
        human_review_buffer=0.01
    )

    # Run U-TIM computation
    utim = PhysicsTIM(model, reference, PosteriorSampler(0.5, 0.1), lambda x, θ: 1.0, config=config)
    result = utim.compute_utim()

    # Output results
    print(f"Final Score: {result.score:.15f} ± {result.uncertainty:.3e}")
    print(f"Safe Score: {result.safe_score:.15f}")

    if result.needs_human_review(0.3):
        print("Borderline case - Flagging for human review")
