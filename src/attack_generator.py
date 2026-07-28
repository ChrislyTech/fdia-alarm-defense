# attack_generator.py


import torch as t
import torch.optim as optim
from config import J_SPE  # ←

print("🟢 attack_generator.py chargé !")

def generate_chattering_attack(x, model, opt, alpha=0.5, beta=0.3, omega=10):

    x = x.clone().detach().requires_grad_(True)
    xp, T, h, hp, W = model(x, opt)  # ← opt is passed here
    loss = ((x - xp) ** 2).sum()
    loss.backward()

    with t.no_grad():
        grad = x.grad
        sin_term = t.sin(omega * t.arange(x.size(1), device=x.device))
        perturbation = alpha * grad + beta * sin_term

        # --- Nouveau : Contrôle dynamique avec changement de signe ---
        # Calculer le SPE actuel
        xp_new, _, _, _, _ = model(x + perturbation.unsqueeze(0), opt)
        spe_current = ((x + perturbation.unsqueeze(0) - xp_new) ** 2).sum().item()

        # Si SPE > seuil, inverser le signe de la perturbation pour faire baisser le SPE
        if spe_current > J_SPE:
            perturbation = -perturbation  # Inverser le signe

        # Si SPE < seuil, garder le signe original pour faire monter le SPE
        elif spe_current < J_SPE:
            perturbation = perturbation  # Garder le signe

        x_attacked = x + perturbation

    return x_attacked.detach()

"""
    Generate a chattering attack that dynamically adjusts to force the SPE to repeatedly cross the detection threshold.
    x: input sample (torch.Tensor)
    model: trained AE model
    opt: config object (passed to model.forward)
    Returns: attacked sample (detached)
    """