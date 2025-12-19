import torch
import torch.nn as nn
from torch.nn import functional as F


import warnings



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2 activation to match standard MLP
        x = self.c_proj(x)
        return x

# moe cooked is not yet needed for current hypothesis
class MoECooked(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_expert)])

        self.n_expert = config.n_expert
        self.topk = config.max_expert_per_tok

    def forward(self, x, y=None, tau=0.5):
        b, l, h = x.size()
        x = x.view(-1, h)
        prior_logits = self.gate(x) # (b * l, n)
        if y is None:
            router_logits = prior_logits
            posterior_logits = None
        else:
            assert y.size() == (b, l, h)
            y = y.view(-1, h)
            router_logits = posterior_logits = self.gate(y) # (b * l, n)

        expert_weights = F.gumbel_softmax(router_logits.float(), tau, hard=False, dim=-1)

        selected_weights, selected_experts = expert_weights.topk(self.topk, dim=-1) # (b * l, k)
        selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
        selected_weights = selected_weights.to(router_logits.dtype)
        expert_mask = F.one_hot(selected_experts, self.n_expert).permute(2, 1, 0) # (n, k, b * l)

        hard_selected_weights = torch.zeros_like(selected_experts)
        hard_selected_weights.scatter_(dim=-1, index=selected_weights.argmax(-1, keepdim=True), value=1.0)
        hard_selected_weights = hard_selected_weights - selected_weights.detach() + selected_weights

        x_new = torch.zeros_like(x)
        y_new = torch.zeros_like(y) if y is not None else None
        for expert_idx in range(self.n_expert):
            if not expert_mask[expert_idx].any():
                continue

            expert_rank, tok_idx = torch.where(expert_mask[expert_idx])
            x_expert = self.experts[expert_idx](x[tok_idx]) * hard_selected_weights[tok_idx, expert_rank].unsqueeze(1)
            x_new.index_add_(0, tok_idx, x_expert.to(x_new.dtype))
            if y is not None:
                y_expert = self.experts[expert_idx](y[tok_idx]) * hard_selected_weights[tok_idx, expert_rank].unsqueeze(1)
                y_new.index_add_(0, tok_idx, y_expert.to(y_new.dtype))

        x_new = x_new.view(b, l, h)
        prior_logits = prior_logits.view(b, l, self.n_expert)
        if y is not None:
            y_new = y_new.view(b, l, h)
            posterior_logits = posterior_logits.view(b, l, self.n_expert)
        return (x_new, prior_logits), (y_new, posterior_logits)



class MoEClassical(nn.Module):
    """
    Classic MoE architecture as used in Switch Transformer, GShard, Mixtral, etc.
    Uses standard softmax routing with top-k expert selection.
    
    Compatible with old GPTMoE checkpoint format - uses 'gate' and 'experts' attributes.
    Returns (output, router_logits) for load balancing loss computation in GPT forward.
    """
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_expert)])

        self.n_expert = config.n_expert
        self.topk = config.max_expert_per_tok

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            output: Output tensor of shape (batch_size, seq_len, n_embd)
            router_logits: Router logits of shape (batch_size * seq_len, n_expert) for load balancing loss
        
        Note: MoE routing creates dynamic tensor shapes (variable-sized slices).
        When using torch.compile, set dynamic=True to avoid recompilation issues.
        """
        B, T, C = x.size()
        x = x.view(-1, C)
        router_logits = self.gate(x)

        expert_weights = F.softmax(router_logits.float(), dim=-1)
        # select top-k weights
        selected_weights, selected_experts = expert_weights.topk(self.topk, dim=-1)
        selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
        selected_weights = selected_weights.to(router_logits.dtype)
        expert_mask = F.one_hot(selected_experts, self.n_expert).permute(2, 1, 0)

        x_new = torch.zeros_like(x)
        for expert_idx in range(self.n_expert):
            if not expert_mask[expert_idx].any():
                continue

            expert_rank, tok_idx = torch.where(expert_mask[expert_idx])
            x_expert = self.experts[expert_idx](x[tok_idx]) * selected_weights[tok_idx, expert_rank].unsqueeze(1)
            x_new.index_add_(0, tok_idx, x_expert.to(x_new.dtype))

        x_new = x_new.view(B, T, C)
        # Return router_logits for load balancing loss computation in GPT forward
        return x_new, router_logits


class MoELookforward(nn.Module):
    """
    MoE with lookforward routing for multi-token prediction.
    
    Key idea:
    - Prior gate: p(expert | x[t]) - uses current token for routing
    - Posterior gate: q(expert | x[t], x[t+n]) - uses future token information
    
    During training:
    - Route tokens using posterior distribution (has access to future via multi-token pred)
    - Minimize KL(posterior || prior) to make prior learn good routing
    
    During inference:
    - Route tokens using prior distribution only (no access to future)
    
    This implements variational routing to optimize the ELBO.
    
    Returns:
    - During training: (output, (router_logits, prior_logits, posterior_logits))
    - During inference: (output, router_logits)
    
    All loss computation (load balancing + KL) happens in GPT forward.
    """
    def __init__(self, config):
        super().__init__()
        # Prior gate: uses current hidden state only
        self.gate_prior = nn.Linear(config.n_embd, config.n_expert, bias=False)
        # Posterior gate: uses future hidden state (from multi-token prediction)
        self.gate_posterior = nn.Linear(config.n_embd, config.n_expert, bias=False)
        
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_expert)])
        
        self.n_expert = config.n_expert
        self.topk = config.max_expert_per_tok
        self.n_predict_tokens = config.n_predict_tokens
    
    def forward(self, x):
        """
        Args:
            x: Current hidden states (batch, seq_len, n_embd)
        
        Returns:
            output: (batch, seq_len, n_embd)
            aux_info: During training: (router_logits, prior_logits, posterior_logits) for loss computation
                      During inference: router_logits only
        """
        b, l, h = x.size()
        x_flat = x.view(-1, h)  # (b * l, h)
        
        # Compute prior router logits (always computed)
        prior_logits = self.gate_prior(x_flat)  # (b * l, n_expert)
        
        # Determine which gate to use for routing
        if self.training and self.n_predict_tokens > 1:
            # Training mode with multi-token prediction: compute posterior using future states
            # Create x_future by shifting: x_future[t] = x[t + n_predict_tokens]
            n = self.n_predict_tokens
            x_future = torch.zeros_like(x)
            
            if l > n:
                # For positions [0, l-n), use actual future states: x_future[t] = x[t+n]
                x_future[:, :-n, :] = x[:, n:, :]
                
                # For last n positions [l-n, l), cannot compute true posterior (would need x[l], x[l+1], etc.)
                # Solution: use x[l-1] (last available position) as future for all last n positions
                # This means all last n positions share the same posterior routing
                x_future[:, -n:, :] = x[:, -1:, :].expand(-1, n, -1)
            else:
                # Sequence shorter than n: use last position for all
                # this actually should not happen
                warnings.warn(f"Sequence shorter than n_predict_tokens: {l} < {n}")
                x_future = x[:, -1:, :].expand(-1, l, -1)
                
            
            x_future_flat = x_future.view(-1, h)  # (b * l, h)
            posterior_logits = self.gate_posterior(x_future_flat)  # (b * l, n_expert)
            router_logits = posterior_logits  # Use posterior for routing
        else:
            # Inference mode or no multi-token prediction: use prior
            router_logits = prior_logits
            posterior_logits = None
        
        # Apply softmax to get routing probabilities
        expert_weights = F.softmax(router_logits, dim=-1)  # (b * l, n_expert)
        
        # Select top-k experts and normalize weights
        selected_weights, selected_experts = expert_weights.topk(self.topk, dim=-1)  # (b * l, k)
        selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
        
        # Create expert mask for efficient computation
        expert_mask = F.one_hot(selected_experts, self.n_expert).permute(2, 1, 0)  # (n_expert, k, b * l)
        
        # Compute weighted sum of expert outputs
        x_new = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_expert):
            if not expert_mask[expert_idx].any():
                continue
            
            # Find tokens assigned to this expert
            expert_rank, tok_idx = torch.where(expert_mask[expert_idx])
            
            # Get expert output and apply routing weights
            expert_output = self.experts[expert_idx](x_flat[tok_idx])
            weights = selected_weights[tok_idx, expert_rank].unsqueeze(1)
            weighted_output = expert_output * weights
            
            # Accumulate outputs
            x_new.index_add_(0, tok_idx, weighted_output.to(x_new.dtype))
        
        # Reshape back to original shape
        x_new = x_new.view(b, l, h)
        
        # Return all logits needed for loss computation in GPT forward
        if self.training and posterior_logits is not None:
            # Return tuple of (router_logits, prior_logits, posterior_logits) for loss computation
            return x_new, (router_logits, prior_logits, posterior_logits)
        else:
            # Inference: just return router_logits
            return x_new, router_logits
