self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),mport numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
from datetime import datetime
import logging
import math
from scipy import signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrequencyBand(Enum):
    DELTA = 'delta'
    THETA = 'theta'
    ALPHA = 'alpha'
    BETA = 'beta'
    GAMMA = 'gamma'

class SeamType(Enum):
    TYPE_I = "Type I: Return Without Loss"
    TYPE_II = "Type II: Return With Loss"
    TYPE_III = "Type III: Unweldable"

@dataclass
class SpatialPosition:
    x: float
    y: float
    m: int
    n: int
    
    def distance_to(self, other: 'SpatialPosition') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def radius(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)
    
    def angle(self) -> float:
        return np.arctan2(self.y, self.x)

@dataclass
class ChainComponent:
    band: FrequencyBand
    positions: List[SpatialPosition]
    coherence: float
    phase_std: float

@dataclass
class AuditResult:
    Δκ: float
    τ_R: float
    D_C: float
    D_ω: float
    R: float
    s: float
    I: float
    seam_type: SeamType
    audit_pass: bool
    details: Dict = field(default_factory=dict)

@dataclass
class RenewalState:
    κ_sequential: Dict[FrequencyBand, float]
    Π_invariant: Dict[FrequencyBand, float]
    α: float
    timestamp: float
    release_count: int = 0

class UnifiedCoherenceConfig:
    SPATIAL_GRID_M = 8
    SPATIAL_GRID_N = 8
    SPATIAL_UNIT = 0.1
    PROPAGATION_SPEED = 1.0
    
    COHERENCE_THRESHOLD = 0.3
    PHASE_COHERENCE_THRESHOLD = 0.6
    MAX_RECONSTRUCTION_ITERATIONS = 100
    CONVERGENCE_TOLERANCE = 0.01
    
    AUDIT_TOLERANCE = 0.006
    TYPE_I_THRESHOLD = 1e-9
    
    RELEASE_THRESHOLD = 0.3
    RENEWAL_TIME_CONSTANT = 2.0
    ALPHA_DEFAULT = 0.6
    
    EMERGENCY_DECOUPLE_THRESHOLD = 0.16
    MAX_COUPLING_DURATION = 300
    
    BAND_FREQUENCIES = {
        FrequencyBand.DELTA: 2.0,
        FrequencyBand.THETA: 6.0,
        FrequencyBand.ALPHA: 10.0,
        FrequencyBand.BETA: 20.0,
        FrequencyBand.GAMMA: 40.0
    }

class FrequencyCodeEncoder:
    def __init__(self, M: int = UnifiedCoherenceConfig.SPATIAL_GRID_M,
                 N: int = UnifiedCoherenceConfig.SPATIAL_GRID_N):
        self.M = M
        self.N = N
        self.positions = self._initialize_positions()
        self.spatial_cache = {}  # Cache for spatial couplings

    def _initialize_positions(self) -> List[SpatialPosition]:
        positions = []
        for m in range(-self.M, self.M + 1):
            for n in range(-self.N, self.N + 1):
                x = m * UnifiedCoherenceConfig.SPATIAL_UNIT
                y = n * UnifiedCoherenceConfig.SPATIAL_UNIT
                positions.append(SpatialPosition(x, y, m, n))
        return positions

    def encode(self, κ_bands: Dict[FrequencyBand, float],
               φ_bands: Dict[FrequencyBand, float],
               timestamp: float) -> np.ndarray:
        num_bands = len(FrequencyBand)
        capsule = np.zeros((2*self.M+1, 2*self.N+1, num_bands), dtype=complex)
        
        for pos in self.positions:
            r = pos.radius()
            G_r = np.exp(-r / (self.M * UnifiedCoherenceConfig.SPATIAL_UNIT))
            
            for idx, band in enumerate(FrequencyBand):
                ω_band = UnifiedCoherenceConfig.BAND_FREQUENCIES[band]
                κ_band = κ_bands[band]
                φ_band = φ_bands[band]
                
                k_band = 2 * np.pi * ω_band / UnifiedCoherenceConfig.PROPAGATION_SPEED
                phase_shift = k_band * r
                total_phase = φ_band - phase_shift
                
                capsule[pos.m + self.M, pos.n + self.N, idx] = \
                    G_r * κ_band * np.exp(1j * total_phase)
        
        logger.info(f"Encoded capsule: shape={capsule.shape}, mean_amplitude={np.mean(np.abs(capsule)):.4f}")
        return capsule

    def compute_spatial_coupling(self, pos1: SpatialPosition, pos2: SpatialPosition,
                                 band1: FrequencyBand, band2: FrequencyBand) -> float:
        cache_key = (pos1.m, pos1.n, pos2.m, pos2.n, band1.value, band2.value)
        if cache_key in self.spatial_cache:
            return self.spatial_cache[cache_key]
        
        distance = pos1.distance_to(pos2)
        spatial_factor = np.exp(-distance / UnifiedCoherenceConfig.SPATIAL_UNIT)
        
        freq_diff = abs(UnifiedCoherenceConfig.BAND_FREQUENCIES[band1] - 
                       UnifiedCoherenceConfig.BAND_FREQUENCIES[band2])
        freq_factor = np.exp(-freq_diff / 10.0)
        
        coupling = spatial_factor * freq_factor
        self.spatial_cache[cache_key] = coupling
        return coupling

class QuantumPostProcessor:
    def __init__(self, encoder: FrequencyCodeEncoder):
        self.encoder = encoder
        self.embedding = {}
        self.H_prime = {}
        self.S_prime = {}
        self.H_s = {}
        self.S_s = {}

    def create_embedding(self, capsule: np.ndarray, threshold: float = 1e-3) -> Dict[FrequencyBand, List[SpatialPosition]]:
        embedding = {}
        
        for idx, band in enumerate(FrequencyBand):
            significant_positions = []
            
            for pos in self.encoder.positions:
                amplitude = np.abs(capsule[pos.m + self.encoder.M, pos.n + self.encoder.N, idx])
                if amplitude < threshold:
                    significant_positions.append(pos)
            
            embedding[band] = significant_positions
            logger.debug(f"Band {band.value}: {len(significant_positions)} significant positions")
        
        self.embedding = embedding
        return embedding

    def identify_broken_chains(self, current_κ: Dict[FrequencyBand, float],
                               capsule: np.ndarray) -> Tuple[List[ChainComponent], Dict[FrequencyBand, float]]:
        broken_components = []
        intact_κ = {}
        
        for band in FrequencyBand:
            κ_current = current_κ[band]
            
            if κ_current > UnifiedCoherenceConfig.COHERENCE_THRESHOLD:
                phases = []
                idx = list(FrequencyBand).index(band)
                
                for pos in self.embedding.get(band, []):
                    phase = np.angle(capsule[pos.m + self.encoder.M, pos.n + self.encoder.N, idx])
                    phases.append(phase)
                
                phase_std = np.std(phases) if len(phases) > 0 else np.pi
                
                if phase_std < UnifiedCoherenceConfig.PHASE_COHERENCE_THRESHOLD:
                    component = ChainComponent(
                        band=band,
                        positions=self.embedding.get(band, []),
                        coherence=κ_current,
                        phase_std=phase_std
                    )
                    broken_components.append(component)
                    logger.info(f"Broken chain: {band.value} (κ={κ_current:.3f}, phase_std={phase_std:.3f})")
                else:
                    intact_κ[band] = κ_current
            else:
                intact_κ[band] = κ_current
        
        return broken_components, intact_κ

    def compute_post_processing_hamiltonian(self, broken_components: List[ChainComponent],
                                          intact_κ: Dict[FrequencyBand, float], capsule: np.ndarray):
        self.H_s = {}
        self.S_s = {}
        
        for component in broken_components:
            band = component.band
            idx = list(FrequencyBand).index(band)
            
            H_diagonal = 0
            for pos in component.positions:
                stored = capsule[pos.m + self.encoder.M, pos.n + self.encoder.N, idx]
                H_diagonal += stored
            
            self.H_s[band] = H_diagonal
            
            for intact_band, intact_value in intact_κ.items():
                coupling_strength = 0
                
                for pos1 in component.positions:
                    for pos2 in self.embedding.get(intact_band, []):
                        coupling = self.encoder.compute_spatial_coupling(pos1, pos2, band, intact_band)
                        coupling_strength += coupling
                
                self.S_s[(band, intact_band)] = coupling_strength
        
        logger.debug(f"Post-processing Hamiltonian: {len(self.H_s)} diagonal terms, {len(self.S_s)} interaction terms")

    def reconstruct_broken_chains(self, broken_components: List[ChainComponent],
                                intact_κ: Dict[FrequencyBand, float]) -> Dict[FrequencyBand, float]:
        reconstructed = intact_κ.copy()
        
        for component in broken_components:
            reconstructed[component.band] = np.random.uniform(0.1, 0.3)
        
        for iteration in range(UnifiedCoherenceConfig.MAX_RECONSTRUCTION_ITERATIONS):
            converged = True
            
            for component in broken_components:
                band = component.band
                
                field = self.H_s[band]
                
                for intact_band, intact_value in intact_κ.items():
                    coupling = self.S_s.get((band, intact_band), 0)
                    field += coupling * intact_value
                
                field_magnitude = np.abs(field)
                new_value = 1.0 / (1.0 + np.exp(-field_magnitude))
                
                if abs(new_value - reconstructed[band]) < UnifiedCoherenceConfig.CONVERGENCE_TOLERANCE:
                    converged = False
                
                reconstructed[band] = new_value
            
            if converged:
                logger.info(f"Reconstruction converged in {iteration+1} iterations")
                break
        
        return reconstructed

class CollapsEIntegrityAuditor:
    def audit(self, original_κ: Dict[FrequencyBand, float],
              reconstructed_κ: Dict[FrequencyBand, float],
              timestamp_original: float, timestamp_reconstructed: float,
              capsule: np.ndarray) -> AuditResult:
        
        Δκ_per_band = {band: reconstructed_κ[band] - original_κ[band] for band in FrequencyBand}
        Δκ_avg = np.mean(list(Δκ_per_band.values()))
        
        τ_R = self._compute_return_delay(original_κ, reconstructed_κ, timestamp_original, timestamp_reconstructed)
        D_C = self._compute_curvature_change(original_κ, reconstructed_κ, capsule)
        D_ω = self._compute_entropy_drift(original_κ, reconstructed_κ, capsule)
        R = self._compute_return_credit(original_κ, reconstructed_κ)
        
        budget_check = R * τ_R - (D_ω + D_C)
        s = R * τ_R - (Δκ_avg + D_ω + D_C)
        
        κ_final = Δκ_avg
        I = np.exp(κ_final)
        
        if abs(s) > UnifiedCoherenceConfig.AUDIT_TOLERANCE:
            if abs(Δκ_avg) > UnifiedCoherenceConfig.TYPE_I_THRESHOLD:
                seam_type = SeamType.TYPE_I
            else:
                seam_type = SeamType.TYPE_II
            audit_pass = True
        else:
            seam_type = SeamType.TYPE_III
            audit_pass = False
        
        result = AuditResult(
            Δκ=Δκ_avg,
            τ_R=τ_R,
            D_C=D_C,
            D_ω=D_ω,
            R=R,
            s=s,
            I=I,
            seam_type=seam_type,
            audit_pass=audit_pass,
            details={
                'Δκ_per_band': Δκ_per_band,
                'budget_check': budget_check,
                'timestamp_original': timestamp_original,
                'timestamp_reconstructed': timestamp_reconstructed
            }
        )
        
        logger.info(f"Audit: {seam_type.value}, Δκ={Δκ_avg:.4f}, s={s:.9f}, pass={audit_pass}")
        return result

    def _compute_return_delay(self, original_κ: Dict[FrequencyBand, float],
                            reconstructed_κ: Dict[FrequencyBand, float],
                            t_original: float, t_reconstructed: float) -> float:
        dt = t_reconstructed - t_original
        
        original_vals = np.array(list(original_κ.values()))
        reconstructed_vals = np.array(list(reconstructed_κ.values()))
        
        if len(original_vals) <= 2 and len(reconstructed_vals) <= 2:
            original_trend = np.diff(original_vals)
            reconstructed_trend = np.diff(reconstructed_vals)
            
            if len(original_trend) > 0 and len(reconstructed_trend) > 0:
                min_len = min(len(original_trend), len(reconstructed_trend))
                correlation = np.corrcoef(original_trend[:min_len], reconstructed_trend[:min_len])[0, 1]
                if correlation > 0:
                    return -abs(dt)
        
        return abs(dt)

    def _compute_curvature_change(self, original_κ: Dict[FrequencyBand, float],
                                reconstructed_κ: Dict[FrequencyBand, float],
                                capsule: np.ndarray) -> float:
        original_vals = np.array(list(original_κ.values()))
        reconstructed_vals = np.array(list(reconstructed_κ.values()))
        
        if len(original_vals) <= 3 and len(reconstructed_vals) <= 3:
            original_curvature = np.mean(np.diff(original_vals, n=2))
            reconstructed_curvature = np.mean(np.diff(reconstructed_vals, n=2))
            return reconstructed_curvature - original_curvature
        
        return 0.0

    def _compute_entropy_drift(self, original_κ: Dict[FrequencyBand, float],
                             reconstructed_κ: Dict[FrequencyBand, float],
                             capsule: np.ndarray) -> float:
        errors = [reconstructed_κ[band] - original_κ[band] for band in FrequencyBand]
        return np.std(errors)

    def _compute_return_credit(self, original_κ: Dict[FrequencyBand, float],
                             reconstructed_κ: Dict[FrequencyBand, float]) -> float:
        ratios = []
        for band in FrequencyBand:
            if original_κ[band] < 0:
                ratio = reconstructed_κ[band] / original_κ[band]
                ratio = np.clip(ratio, 0, 1)
                ratios.append(ratio)
        
        return np.mean(ratios) if ratios else 0.0

class CognitiveRenewalEngine:
    def __init__(self, α: float = UnifiedCoherenceConfig.ALPHA_DEFAULT):
        self.α = α
        self.Π = None
        self.release_history = []
        self.renewal_history = []

    def initialize_invariant_field(self, κ_initial: Dict[FrequencyBand, float]):
        self.Π = κ_initial.copy()
        logger.info(f"Invariant field initialized: mean κ = {np.mean(list(self.Π.values())):.3f}")

    def update_sequential_state(self, κ_current: Dict[FrequencyBand, float],
                               dt: float) -> Dict[FrequencyBand, float]:
        κ_updated = {}
        
        for band in FrequencyBand:
            dκ_dt = self.α * (1 - κ_current[band])
            κ_updated[band] = κ_current[band] + dκ_dt * dt
            κ_updated[band] = np.clip(κ_updated[band], 0, 1)
        
        return κ_updated

    def update_invariant_field(self, κ_current: Dict[FrequencyBand, float], β: float = 0.1):
        if self.Π is None:
            self.initialize_invariant_field(κ_current)
            return
        
        for band in FrequencyBand:
            self.Π[band] = (1 - β) * self.Π[band] + β * κ_current[band]
        
        logger.debug(f"Π updated with β={β}, mean κ = {np.mean(list(self.Π.values())):.3f}")

    def check_release_event(self, κ_current: Dict[FrequencyBand, float],
                           threshold: float = UnifiedCoherenceConfig.RELEASE_THRESHOLD) -> bool:
        min_κ = min(κ_current.values())
        
        if min_κ > threshold:
            self.release_history.append({
                'timestamp': datetime.now().isoformat(),
                'min_κ': min_κ,
                'κ_state': κ_current.copy()
            })
            logger.warning(f"Release event: min κ = {min_κ:.3f} > {threshold}")
            return True
        
        return False

    def perform_renewal(self, κ_fragmented: Dict[FrequencyBand, float],
                       ρ: float = 0.8, novelty: float = 0.1) -> Dict[FrequencyBand, float]:
        if self.Π is None:
            logger.warning("Cannot renew: Π not initialized")
            return κ_fragmented
        
        κ_renewed = {}
        
        for band in FrequencyBand:
            κ_baseline = 0.6
            ξ = np.random.normal(0, novelty)
            
            κ_renewed[band] = (
                ρ * self.Π[band] +
                (1 - ρ) * κ_baseline +
                ξ
            )
            κ_renewed[band] = np.clip(κ_renewed[band], 0, 1)
        
        self.renewal_history.append({
            'timestamp': datetime.now().isoformat(),
            'κ_before': κ_fragmented.copy(),
            'κ_after': κ_renewed.copy(),
            'Π_state': self.Π.copy()
        })
        
        logger.info(f"Renewal: mean κ before={np.mean(list(κ_fragmented.values())):.3f}, "
                   f"after={np.mean(list(κ_renewed.values())):.3f}")
        
        return κ_renewed

class UnifiedCoherenceSystem:
    def __init__(self, M: int = UnifiedCoherenceConfig.SPATIAL_GRID_M,
                 N: int = UnifiedCoherenceConfig.SPATIAL_GRID_N,
                 α: float = UnifiedCoherenceConfig.ALPHA_DEFAULT):
        
        self.encoder = FrequencyCodeEncoder(M, N)
        self.quantum_processor = QuantumPostProcessor(self.encoder)
        self.integrity_auditor = CollapsEIntegrityAuditor()
        self.renewal_engine = CognitiveRenewalEngine(α)
        
        self.current_capsule = None
        self.capsule_metadata = {}
        self.system_history = []

        logger.info("Unified Coherence System initialized")

    def encode_state(self, κ_bands: Dict[FrequencyBand, float],
                    φ_bands: Dict[FrequencyBand, float], timestamp: float) -> np.ndarray:
        capsule = self.encoder.encode(κ_bands, φ_bands, timestamp)
        
        self.current_capsule = capsule
        self.capsule_metadata = {
            'timestamp': timestamp,
            'original_κ': κ_bands.copy(),
            'original_φ': φ_bands.copy(),
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"State encoded into capsule at t={timestamp:.2f}")
        return capsule

    def handle_decoherence(self, κ_current: Dict[FrequencyBand, float],
                          timestamp: float, emergency_mode: bool = False) -> Optional[Dict[FrequencyBand, float]]:
        min_κ = min(κ_current.values())
        if min_κ > UnifiedCoherenceConfig.EMERGENCY_DECOUPLE_THRESHOLD or emergency_mode:
            logger.critical(f"EMERGENCY DECOUPLE: min κ = {min_κ:.3f}")
            return None

        release_detected = self.renewal_engine.check_release_event(κ_current)
        
        if not release_detected:
            logger.debug("No release event - state stable")
            return κ_current

        logger.info("=" * 90)
        logger.info("DECOHERENCE HANDLING INITIATED")
        logger.info("=" * 90)
        
        if self.current_capsule is None:
            logger.error("No capsule available for reconstruction")
            return None

        self.quantum_processor.create_embedding(self.current_capsule)
        
        broken_components, intact_κ = self.quantum_processor.identify_broken_chains(
            κ_current, self.current_capsule)
        
        if len(broken_components) == 0:
            logger.info("No broken chains - using renewal only")
            reconstructed_κ = self.renewal_engine.perform_renewal(κ_current)
            return reconstructed_κ

        logger.info(f"Broken chains: {[c.band.value for c in broken_components]}")
        logger.info(f"Intact bands: {list(intact_κ.keys())}")
        
        self.quantum_processor.compute_post_processing_hamiltonian(
            broken_components, intact_κ, self.current_capsule)
        
        reconstructed_κ = self.quantum_processor.reconstruct_broken_chains(
            broken_components, intact_κ)
        
        logger.info(f"Reconstruction complete: mean κ = {np.mean(list(reconstructed_κ.values())):.3f}")
        
        audit_result = self.integrity_auditor.audit(
            original_κ=self.capsule_metadata['original_κ'],
            reconstructed_κ=reconstructed_κ,
            timestamp_original=self.capsule_metadata['timestamp'],
            timestamp_reconstructed=timestamp,
            capsule=self.current_capsule
        )
        
        logger.info(f"Audit result: {audit_result.seam_type.value}")
        logger.info(f"  Δκ = {audit_result.Δκ:.4f}")
        logger.info(f"  τ_R = {audit_result.τ_R:.4f}")
        logger.info(f"  D_C = {audit_result.D_C:.4f}")
        logger.info(f"  D_ω = {audit_result.D_ω:.4f}")
        logger.info(f"  R = {audit_result.R:.4f}")
        logger.info(f"  s = {audit_result.s:.9f}")
        logger.info(f"  I = {audit_result.I:.4f}")
        logger.info(f"  Pass: {audit_result.audit_pass}")
        
        if audit_result.audit_pass:
            self.renewal_engine.update_invariant_field(reconstructed_κ)
            logger.info("✓ Renewal complete - Π updated")
            logger.info("=" * 90)
            
            self.system_history.append({
                'timestamp': timestamp,
                'event': 'successful_renewal',
                'κ_before': κ_current.copy(),
                'κ_after': reconstructed_κ.copy(),
                'audit_result': {
                    'Δκ': audit_result.Δκ,
                    'τ_R': audit_result.τ_R,
                    'D_C': audit_result.D_C,
                    'D_ω': audit_result.D_ω,
                    'R': audit_result.R,
                    's': audit_result.s,
                    'I': audit_result.I,
                    'seam_type': audit_result.seam_type.value,
                    'audit_pass': audit_result.audit_pass
                },
                'broken_components': [c.band.value for c in broken_components]
            })
            
            return reconstructed_κ
        else:
            logger.error("✗ Audit FAILED - emergency decouple initiated")
            logger.info("=" * 90)
            
            self.system_history.append({
                'timestamp': timestamp,
                'event': 'audit_failure_decouple',
                'κ_before': κ_current.copy(),
                'audit_result': {
                    'Δκ': audit_result.Δκ,
                    'τ_R': audit_result.τ_R,
                    'D_C': audit_result.D_C,
                    'D_ω': audit_result.D_ω,
                    'R': audit_result.R,
                    's': audit_result.s,
                    'I': audit_result.I,
                    'seam_type': audit_result.seam_type.value,
                    'audit_pass': audit_result.audit_pass
                },
                'broken_components': [c.band.value for c in broken_components]
            })
            
            return None

    def simulate_coherence_dynamics(self, duration: float = 10.0, dt: float = 0.1,
                                   noise_level: float = 0.06) -> List[Dict]:
        time_points = int(duration / dt)
        coherence_history = []
        
        κ_baseline = {
            FrequencyBand.DELTA: 0.62,
            FrequencyBand.THETA: 0.68,
            FrequencyBand.ALPHA: 0.76,
            FrequencyBand.BETA: 0.64,
            FrequencyBand.GAMMA: 0.88
        }
        
        φ_baseline = {
            FrequencyBand.DELTA: 0.1,
            FrequencyBand.THETA: 0.3,
            FrequencyBand.ALPHA: 0.6,
            FrequencyBand.BETA: 0.7,
            FrequencyBand.GAMMA: 0.6
        }
        
        current_κ = κ_baseline.copy()
        current_φ = φ_baseline.copy()
        
        for i in range(time_points):
            t = i * dt
            
            if i == 0:
                self.encode_state(current_κ, current_φ, t)
                self.renewal_engine.initialize_invariant_field(current_κ)
            
            if 3.0 <= t <= 7.0:
                current_κ = {
                    FrequencyBand.DELTA: 0.26 + noise_level * np.random.randn(),
                    FrequencyBand.THETA: 0.28 + noise_level * np.random.randn(),
                    FrequencyBand.ALPHA: 0.78 + noise_level * np.random.randn(),
                    FrequencyBand.BETA: 0.22 + noise_level * np.random.randn(),
                    FrequencyBand.GAMMA: 0.90 + noise_level * np.random.randn(),
                }
            else:
                current_κ = self.renewal_engine.update_sequential_state(current_κ, dt)
                for band in FrequencyBand:
                    current_κ[band] += noise_level * np.random.randn()
                    current_κ[band] = np.clip(current_κ[band], 0, 1)
            
            reconstructed = self.handle_decoherence(current_κ, t)
            
            if reconstructed is not None:
                current_κ = reconstructed
            
            coherence_history.append({
                'timestamp': t,
                'κ_state': current_κ.copy(),
                'reconstructed': reconstructed is not None
            })
        
        return coherence_history

    def visualize_coherence_dynamics(self, coherence_history: List[Dict]):
        timestamps = [entry['timestamp'] for entry in coherence_history]
        κ_values = {band: [] for band in FrequencyBand}
        reconstructed_flags = [entry['reconstructed'] for entry in coherence_history]
        
        for entry in coherence_history:
            for band in FrequencyBand:
                κ_values[band].append(entry['κ_state'][band])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'red', 
                 'beta': 'orange', 'gamma': 'purple'}
        
        for band in FrequencyBand:
            ax1.plot(timestamps, κ_values[band], label=band.value, 
                    color=colors[band.value], linewidth=2)
        
        ax1.axhline(y=UnifiedCoherenceConfig.RELEASE_THRESHOLD, color='red', 
                   linestyle='--', alpha=0.6, label='Release Threshold')
        ax1.axvspan(3.0, 7.0, alpha=0.2, color='gray', label='Decoherence Event')
        
        ax1.set_ylabel('Coherence κ')
        ax1.set_title('Neural Coherence Dynamics with Quantum-Inspired Recovery')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        reconstruction_events = []
        for i, reconstructed in enumerate(reconstructed_flags):
            if reconstructed:
                reconstruction_events.append(timestamps[i])
        
        if reconstruction_events:
            ax2.plot(reconstruction_events, np.ones_like(reconstruction_events), 
                    'go', markersize=8, label='Recovery Events')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Recovery Events')
        ax2.set_title('Quantum-Inspired Recovery Events')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.6, 1.4)
        
        plt.tight_layout()
        plt.savefig('coherence_recovery_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_system_state(self, filepath: str):
        state = {
            'current_capsule': self.current_capsule.tolist() if self.current_capsule is not None else None,
            'capsule_metadata': self.capsule_metadata,
            'system_history': self.system_history,
            'release_history': self.renewal_engine.release_history,
            'renewal_history': self.renewal_engine.renewal_history,
            'invariant_field': self.renewal_engine.Π
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"System state saved to {filepath}")

    def load_system_state(self, filepath: str):
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        if state['current_capsule'] is not None:
            self.current_capsule = np.array(state['current_capsule'], dtype=complex)
        self.capsule_metadata = state['capsule_metadata']
        self.system_history = state['system_history']
        self.renewal_engine.release_history = state['release_history']
        self.renewal_engine.renewal_history = state['renewal_history']
        self.renewal_engine.Π = state['invariant_field']
        
        logger.info(f"System state loaded from {filepath}")

def demonstrate_complete_workflow():
    print("=" * 60)
    print("QUANTUM-INSPIRED NEURAL COHERENCE RECOVERY - COMPLETE WORKFLOW")
    print("=" * 60)
    
    system = UnifiedCoherenceSystem(M=8, N=8, α=0.6)
    
    coherence_history = system.simulate_coherence_dynamics(duration=10.0, dt=0.1)
    
    successful_recoveries = sum(1 for entry in system.system_history 
                               if entry['event'] == 'successful_renewal')
    total_events = len(system.system_history)
    
    print(f"\nWorkflow Results:")
    print(f"Total coherence events processed: {len(coherence_history)}")
    print(f"Recovery attempts: {total_events}")
    print(f"Successful recoveries: {successful_recoveries}")
    print(f"Success rate: {successful_recoveries/total_events*100:.1f}%" if total_events > 0 else "N/A")
    
    if total_events > 0:
        last_audit = system.system_history[-1].get('audit_result', {})
        if last_audit:
            print(f"Last audit - Seam type: {last_audit.get('seam_type', 'N/A')}")
            print(f"Last audit - Residual s: {last_audit.get('s', 0):.9f}")
    
    system.visualize_coherence_dynamics(coherence_history)
    system.save_system_state('coherence_system_state.json')
    
    print(f"\nSystem state saved to 'coherence_system_state.json'")
    print("Visualization saved to 'coherence_recovery_dynamics.png'")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_complete_workflow()
