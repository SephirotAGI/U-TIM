import numpy as np
from scipy.integrate import quad
from scipy.stats import sem, entropy, gumbel_r
from sympy import symbols, lambdify
from abc import ABC, abstractmethod

# Pre-1929 crisis data (normalized to [0,1] range for neural convergence)
pre_crash_data = {
    -7: {'production_idx': 0.98, 'margin_debt': 0.34, 'velocity': 0.59, 'debt_growth': 0.45, 'confidence': 0.75},
    -6: {'production_idx': 0.97, 'margin_debt': 0.35, 'velocity': 0.58, 'debt_growth': 0.47, 'confidence': 0.72},
    -5: {'production_idx': 0.96, 'margin_debt': 0.36, 'velocity': 0.57, 'debt_growth': 0.50, 'confidence': 0.70},
    -4: {'production_idx': 0.95, 'margin_debt': 0.37, 'velocity': 0.56, 'debt_growth': 0.52, 'confidence': 0.68},
    -3: {'production_idx': 0.94, 'margin_debt': 0.38, 'velocity': 0.55, 'debt_growth': 0.55, 'confidence': 0.65},
    -2: {'production_idx': 0.93, 'margin_debt': 0.39, 'velocity': 0.54, 'debt_growth': 0.57, 'confidence': 0.62},
    -1: {'production_idx': 0.92, 'margin_debt': 0.40, 'velocity': 0.53, 'debt_growth': 0.60, 'confidence': 0.60},
    0: {'production_idx': 0.90, 'margin_debt': 0.42, 'velocity': 0.52, 'debt_growth': 0.65, 'confidence': 0.55}
}

class UTIMConfig:
    def __init__(self, precision='float64', num_runs=21, crash_threshold=0.85, alpha=1.5):
        self.precision = np.float64  # Standardized for financial data compatibility
        self.num_runs = num_runs
        self.crash_threshold = crash_threshold
        self.alpha = alpha  # Crisis acceleration factor

# Financial data accessors with temporal alignment
def industrial_production(x):
    day = int(np.clip(np.round(x), -7, 0))
    return pre_crash_data[day]['production_idx']

def margin_debt(x):
    day = int(np.clip(np.round(x), -7, 0))
    return pre_crash_data[day]['margin_debt']

def money_velocity(x):
    day = int(np.clip(np.round(x), -7, 0))
    return pre_crash_data[day]['velocity']

def debt_growth(x):
    day = int(np.clip(np.round(x), -7, 0))
    return pre_crash_data[day]['debt_growth']

def investor_confidence(x):
    day = int(np.clip(np.round(x), -7, 0))
    return pre_crash_data[day]['confidence']

# Crisis posterior sampler using extreme value theory
class CrisisPosterior:
    def __init__(self, loc=1.8, scale=0.4):
        self.dist = gumbel_r(loc=loc, scale=scale)
        
    def sample(self, n):
        return self.dist.rvs(n)
    
    def log_prob(self, theta):
        return self.dist.logpdf(theta)

# Crash prediction model with Fisher–Minsky dynamics
class MarketCrashModel:
    def __init__(self, config=UTIMConfig()):
        self.config = config
        
    def _instability_integrand(self, x, theta):
        """Fisherian debt-deflation dynamics"""
        instability = (
            theta * (money_velocity(x) - debt_growth(x)) +
            self.config.alpha * investor_confidence(x)**2
        )
        return np.exp(-instability**2)

    def _compute_crash_risk(self, theta):
        integral, _ = quad(self._instability_integrand, -7, 0, args=(theta,))
        return integral / 7  # Temporal normalization

    def predict_crisis(self):
        posterior = CrisisPosterior()
        samples = posterior.sample(1000)
        
        risks = [self._compute_crash_risk(theta) for theta in samples]
        weighted_risk = np.mean(risks) * np.std(risks)
        
        return {
            'crash_probability': min(weighted_risk, 1.0),
            'threshold_breach': weighted_risk > self.config.crash_threshold,
            'risk_components': {
                'production': np.mean([industrial_production(x) for x in range(-7,1)]),
                'margin_debt': np.mean([margin_debt(x) for x in range(-7,1)]),
                'confidence_decay': investor_confidence(0) - investor_confidence(-7)
            }
        }

# Usage with 1929 parameters
config = UTIMConfig(alpha=1.7, crash_threshold=0.82)
model = MarketCrashModel(config)
result = model.predict_crisis()

print(f"Crash Probability: {result['crash_probability']:.2%}")
print(f"Systemic Risk Alert: {result['threshold_breach']}")
print(f"Production Trend: {result['risk_components']['production']:.3f}")
print(f"Margin Debt Accumulation: {result['risk_components']['margin_debt']:.3f}")
print(f"Confidence Decay Rate: {result['risk_components']['confidence_decay']:.3f}")
