# visualize_attacks.py
import matplotlib.pyplot as plt

def plot_attack_results(x_original, x_attacked, spe_original, spe_attacked, title="Attack Results"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_original.detach().numpy(), label='Original')
    plt.plot(x_attacked.detach().numpy(), label='Attacked', linestyle='--')
    plt.title(title)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(['Original', 'Attacked'], [spe_original, spe_attacked], color=['blue', 'red'])
    plt.title('SPE Before and After Attack')
    plt.ylabel('SPE')
    plt.show()