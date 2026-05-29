import torch
import torch.nn as nn
import torch.nn.functional as F

class Debiased_Representation_Loss(nn.Module):
    """
    Debiased Representation Loss Module.
    Synergizes Soft Entropy Regularization (HAPPY-CGCD) and Soft Neighborhood Contrastive Loss (MetaGCD).
    """
    def __init__(self, feature_dim, hidden_dim, epsilon=0.6, tau=0.1):
        super(Debiased_Representation_Loss, self).__init__()
        self.epsilon = epsilon  # Lowered from 0.8 to 0.6: with few samples, fewer
                                # pairs exceed a 0.8 cosine threshold → contrastive
                                # loss collapses to 0 and provides no gradient signal.
        self.tau = tau
        
        # Attention Module linear projections
        self.f1 = nn.Linear(feature_dim, hidden_dim)
        self.f2 = nn.Linear(feature_dim, hidden_dim)

    def forward(self, z_u, logits, old_class_indices, new_class_indices, base_features):
        """
        Forward pass for the composite loss.
        
        Args:
            z_u: Unlabeled features, shape (B, D)
            logits: Model logits corresponding to z_u, shape (B, num_classes)
            old_class_indices: List or tensor of old class indices, shape (C_old,)
            new_class_indices: List or tensor of new class indices, shape (C_new,)
            base_features: Pre-trained ViT features used to construct stable neighborhood masks, shape (B, 768)
            
        Returns:
            total_loss: The combined debiased representation loss.
            loss_dict: Dictionary containing individual loss components for tracking.
        """
        device = z_u.device
        B, D = z_u.shape
        
        num_old = len(old_class_indices)
        num_new = len(new_class_indices)
        
        # ---------------------------------------------------------
        # Part A: Soft Entropy Regularization (HAPPY-CGCD)
        # ---------------------------------------------------------
        active_indices = old_class_indices + new_class_indices
        
        if len(active_indices) > 0:
            # Temperature scaling: weight-normed logits are in [-1,1], so unscaled
            # softmax over 60 classes is nearly uniform (entropy ≈ 0 always).
            # Dividing by tau sharpens the distribution so entropy has gradients.
            probs = F.softmax(logits[:, active_indices] / self.tau, dim=1)  # Shape: (B, len(active_indices))
            
            # Compute marginal probabilities over the batch
            mean_probs = probs.mean(dim=0)  # Shape: (len(active_indices),)
        else:
            # Edge case: no active classes (shouldn't happen in practice)
            mean_probs = torch.ones(1, device=device)
        
        import math
        
        if num_old > 0 and num_new > 0:
            # Standard inter-set entropy: encourage balanced old/new probability mass
            p_old = mean_probs[:num_old].sum()
            p_new = mean_probs[num_old:].sum()
            
            loss_entropy_inter = p_old * torch.log(p_old + 1e-8) + p_new * torch.log(p_new + 1e-8) + math.log(2)
            
            # Intra-set entropy for old classes
            p_old_in = mean_probs[:num_old] / (p_old + 1e-8)
            loss_entropy_old_in = torch.sum(p_old_in * torch.log(p_old_in + 1e-8)) + math.log(p_old_in.size(0))
            
            # Intra-set entropy for new classes
            p_new_in = mean_probs[num_old:] / (p_new + 1e-8)
            if p_new_in.size(0) > 1:
                loss_entropy_new_in = torch.sum(p_new_in * torch.log(p_new_in + 1e-8)) + math.log(p_new_in.size(0))
            else:
                # increment=1: single new class → intra-entropy is trivially 0.
                # Instead, add a term that encourages the model to assign non-trivial
                # probability mass to the novel class (prevents it from being ignored).
                loss_entropy_new_in = -0.5 * torch.log(p_new + 1e-8)
        
        elif num_old == 0 and num_new > 0:
            # Stage 0: only new classes, no old. Just maximize intra-new entropy.
            loss_entropy_inter = torch.tensor(0.0, device=device)
            loss_entropy_old_in = torch.tensor(0.0, device=device)
            
            p_new_in = mean_probs / (mean_probs.sum() + 1e-8)
            if p_new_in.size(0) > 1:
                loss_entropy_new_in = torch.sum(p_new_in * torch.log(p_new_in + 1e-8)) + math.log(p_new_in.size(0))
            else:
                loss_entropy_new_in = torch.tensor(0.0, device=device)
        
        else:
            # Fallback: no classes at all
            loss_entropy_inter = torch.tensor(0.0, device=device)
            loss_entropy_old_in = torch.tensor(0.0, device=device)
            loss_entropy_new_in = torch.tensor(0.0, device=device)
        
        # Total Soft Entropy Loss
        loss_entropy = loss_entropy_inter + loss_entropy_old_in + loss_entropy_new_in
        
        # ---------------------------------------------------------
        # Part B: Soft Neighborhood Contrastive Loss (MetaGCD)
        # ---------------------------------------------------------
        # L2 Normalize base features to construct stable neighborhood masks
        base_norm = F.normalize(base_features, p=2, dim=1)  # Shape: (B, 768)
        base_sim_matrix = torch.mm(base_norm, base_norm.t())  # Shape: (B, B)
        
        # Generate boolean mask of candidate nearest neighbors NN(Z_i) from BASE features!
        mask = base_sim_matrix > self.epsilon  # Shape: (B, B)
        mask.fill_diagonal_(False)        # Avoid self-matching
        
        # L2 Normalize student features for InfoNCE
        z_norm = F.normalize(z_u, p=2, dim=1)  # Shape: (B, D)
        sim_matrix = torch.mm(z_norm, z_norm.t())  # Shape: (B, B)
        
        # Attention Module: linear projections f1 and f2
        f1_z = self.f1(z_u)  # Shape: (B, hidden_dim)
        f2_z = self.f2(z_u)  # Shape: (B, hidden_dim)
        
        # Compute raw attention scores
        attn_logits = torch.mm(f1_z, f2_z.t())  # Shape: (B, B)
        
        # Apply mask to attention logits (clamp to very small number where mask is False)
        # We ensure soft positiveness weights are only distributed among neighbors
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        
        # Compute soft positiveness weights w_{ik}
        w_ik = F.softmax(attn_logits, dim=1)  # Shape: (B, B)
        
        # Mask out w_ik where there are no neighbors (softmax might give non-zero if all row is -1e9)
        w_ik = w_ik * mask.float() # Ensure exact 0 for non-neighbors
        
        exp_sim_tau = torch.exp(sim_matrix / self.tau)  # Shape: (B, B)
        
        # Denominator: sum_{n \neq i} exp(Z_i * Z_n / tau)
        exp_sim_tau_no_self = exp_sim_tau.clone()
        exp_sim_tau_no_self.fill_diagonal_(0)
        denominator = exp_sim_tau_no_self.sum(dim=1, keepdim=True)  # Shape: (B, 1)
        
        # Compute log components (InfoNCE standard log formulation)
        # Note: MetaGCD uses SupConLoss formulation where w_ik multiplies the log term outside.
        log_prob = (sim_matrix / self.tau) - torch.log(denominator + 1e-8)
        
        # Formulate the loss: -(1 / |NN|) * sum_k (w_ik * log_prob)
        num_neighbors = mask.sum(dim=1).float()  # Shape: (B,)
        
        # Only compute loss for samples that have at least one neighbor to avoid division by zero
        valid_samples = num_neighbors > 0
        
        if valid_samples.any():
            loss_soft_i = -(w_ik * mask.float() * log_prob).sum(dim=1)[valid_samples] / num_neighbors[valid_samples]
            loss_contrastive = loss_soft_i.mean()
        else:
            loss_contrastive = torch.tensor(0.0, device=device)
        
        # ---------------------------------------------------------
        # Final Composite Loss
        # ---------------------------------------------------------
        # When increment is small (1-2 classes), the entropy terms provide less
        # gradient signal.  Upweight the contrastive loss to compensate.
        if num_new <= 2 and num_old > 0:
            contrastive_weight = 2.0
        else:
            contrastive_weight = 1.0
        
        total_loss = loss_entropy + contrastive_weight * loss_contrastive
        
        loss_dict = {
            'loss_entropy_inter': loss_entropy_inter.item(),
            'loss_entropy_old_in': loss_entropy_old_in.item(),
            'loss_entropy_new_in': loss_entropy_new_in.item(),
            'loss_contrastive': loss_contrastive.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
