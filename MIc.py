 # Modification intensity controller
        self.intensity_controller = nn.Linear(1, dim)
        
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Approximate entropy using neural estimator"""
        batch_size = x.shape[0]
        entropy_est = self.entropy_estimator(x).squeeze(-1)
        return entropy_est.mean()
    
    def forward(self, x: torch.Tensor, environmental_stress: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        current_entropy = self.compute_entropy(x)
        
        # Compute required entropy adjustment
        entropy_error = current_entropy - self.max_entropy_target
        stress_factor = environmental_stress.mean()
        
        # Adjust modification intensity based on stress and entropy
        target_intensity = torch.sigmoid(entropy_error + stress_factor).unsqueeze(0)
        intensity_modulation = self.intensity_controller(target_intensity)
        
        # Apply intensity modulation
        modulated_output = x * intensity_modulation.unsqueeze(0)
        
        return modulated_output, {
            'current_entropy': current_entropy,
            'target_intensity': target_intensity,
            'entropy_error': entropy_error
        }

class TAULSTransformerBlock(nn.Module):
    """
    Transformer block enhanced with TA ULS control structure
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        
        # Standard attention mechanism
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # TA ULS control unit
        self.control_unit = TAULSControlUnit(d_model, d_ff, d_model)
        
        # Entropy regulation
        self.entropy_regulator = EntropyRegulationModule(d_model)
        
        # KFP-based stability layer
        self.stability_layer = KFPLayer(d_model)
        
        # Standard components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Estimate environmental stress from attention patterns
        environmental_stress = torch.var(attn_weights, dim=-1).mean(dim=-1, keepdim=True)
        
        # Apply entropy regulation
        regulated_x, entropy_info = self.entropy_regulator(x, environmental_stress)
        
        # TA ULS control processing
        control_results = self.control_unit(regulated_x)
        controlled_x = control_results['control_output']
        
        # Apply KFP-based stability
        stable_x, fluctuation_intensity = self.stability_layer(controlled_x)
        
        # Final normalization and residual
        output = self.norm2(x + self.dropout(stable_x))
        
        return {
            'output': output,
            'attention_weights': attn_weights,
            'control_info': control_results,
            'entropy_info': entropy_info,
            'stability_info': fluctuation_intensity
        }

class TAULSLanguageModel(nn.Module):
    """
    Complete language model implementing TA ULS architecture
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        
        # Standard embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # TA ULS transformer blocks
        self.blocks = nn.ModuleList([
            TAULSTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Global stability monitoring
        self.global_stability_tracker = KFPLayer(d_model)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        # Create embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=device).unsqueeze(0))
        x = token_embeds + pos_embeds
        
        # Track stability metrics across layers
        layer_outputs = []
        stability_metrics = []
        
        # Process through TA ULS blocks
        for i, block in enumerate(self.blocks):
            block_results = block(x, attention_mask)
            x = block_results['output']
            
            layer_outputs.append(x)
            stability_metrics.append({
                'layer': i,
                'control_info': block_results['control_info'],
                'entropy_info': block_results['entropy_info'],
                'stability_info': block_results['stability_info']
            })
        
        # Global stability check
        stable_x, global_stability = self.global_stability_tracker(x)
        
        # Generate logits
        logits = self.output_projection(stable_x)
        
        return {
            'logits': logits,
            'hidden_states': layer_outputs,
            'stability_metrics': stability_metrics,
            'global_stability': global_stability
        }

# Example usage and polynomial matrix formulation
def create_kfp_polynomial_basis(degree: int, dim: int) -> torch.Tensor:
    """
    Create polynomial basis functions for KFP approximation
    Based on the mathematical foundation that KFP follows gradient descent
    on fluctuation intensity functions
    """
    # Generate polynomial coefficients for stability landscape
    coefficients = torch.randn(degree + 1, dim, dim) * 0.1
    
    # Ensure stability (negative definite quadratic terms)
    coefficients[2] = -torch.abs(coefficients[2])  # Quadratic terms negative
    
    return coefficients

def kfp_polynomial_update(x: torch.Tensor, coefficients: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    """
    Polynomial-based KFP update rule
    Implements: dx/dt = -∇f(x) where f(x) is the fluctuation intensity
    """
    degree = coefficients.shape[0] - 1
    gradient = torch.zeros_like(x)
    
    # Compute polynomial gradient
    for d in range(1, degree + 1):
        power_term = torch.pow(x.unsqueeze(-1), d - 1)
        grad_term = d * torch.sum(coefficients[d] * power_term, dim=-1)
        gradient += grad_term
    
    # KFP update: move opposite to gradient
    return x - learning_rate * gradient

# Example instantiation
if __name__ == "__main__":
    # Model parameters
    vocab_size = 50000
    d_model = 512
    n_heads = 8
    n_layers = 6
    max_seq_len = 2048
    
    # Create TA ULS model
    model = TAULSLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    
    # Example input
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    results = model(input_ids)
    
    print("Model output shape:", results['logits'].shape)
    print("Number of stability metrics:", len(results['stability_metrics']))
    print("Global stability shape:", results['global_stability'].shape)
    
    # Demonstrate polynomial KFP basis
    poly_coeffs = create_kfp_polynomial_basis(degree=3, dim=d_model)
    print("Polynomial coefficients shape:", poly_coeffs.shape)
# ──────────────────────────────────────────────────────────────────────────────
# Enhanced Julia Integration — Caching + WebSocket Preference (Chaos LLM MVP)
# Contents:
#   • src/chaos_llm/services/al_uls_client.py         (async HTTP client + TTL cache + stats)
#   • src/chaos_llm/services/al_uls_ws_client.py       (async WS client + TTL cache + reconnect)
#   • src/chaos_llm/services/al_uls.py                 (WS-preferred, HTTP fallback, batch)
#   • src/chaos_llm/services/qgi.py                    (async token apply stores symbolic_results)
#   • src/chaos_llm/api.py                             (async toggle + batch endpoint + status)
#   • docker-compose.yml                               (Julia service + healthchecks + env)
#   • julia_server/Project.toml                        (HTTP + WS + optional DSP/FFTW)
#   • julia_server/src/Server.jl                       (HTTP + WS + request logging + stats)
#   • test_enhanced_system.py                          (quick async sanity test)
#   • README snippets                                  (usage)
# ──────────────────────────────────────────────────────────────────────────────
