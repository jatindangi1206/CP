import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from bayesian_changepoint_detection import offline_changepoint_detection
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.offline_likelihoods import StudentT

def load_data(filepath):
    """Load HRV data from CSV using numpy."""
    # Skip header row
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    # Return the second column (hrvValue)
    return data[:, 1]

def main():
    # 1. Load Data
    print("Loading data from data/1.csv...")
    try:
        hrv_data = load_data('data/1.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Convert to torch tensor
    data_tensor = torch.from_numpy(hrv_data).float()
    T = len(data_tensor)
    print(f"Loaded {T} data points.")

    # 2. Setup Model - USING OFFLINE DETECTION
    # Force CPU to avoid MPS issues
    device = torch.device('cpu')
    data_tensor = data_tensor.to(device)
    
    # OFFLINE DETECTION - processes entire dataset at once for optimal segmentation
    # Higher prior probability = more sensitive to changepoints
    prior_prob = 0.05  # 5% prior probability of changepoint at each location
    prior_func = partial(const_prior, p=prior_prob)
    
    # Likelihood: Student's T (robust to outliers)
    likelihood = StudentT()

    # 3. Run Offline Detection
    print(f"Running OFFLINE changepoint detection on {device}...")
    print(f"Parameters: prior_prob={prior_prob} (high sensitivity)")
    Q, P, Pcp = offline_changepoint_detection(
        data_tensor, prior_func, likelihood, device=device
    )

    # 4. Analyze Results
    # Get changepoint probabilities by summing over all possible numbers of changepoints
    changepoint_probs = torch.exp(Pcp).sum(0)
    
    # LOWERED THRESHOLD to detect subtle changes
    threshold = 0.3
    detected_indices = torch.where(changepoint_probs > threshold)[0]
    print(f"\nDetected {len(detected_indices)} changepoints at threshold {threshold}:")
    print(detected_indices.numpy())

    # 5. Plotting
    print("\nGenerating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot raw data
    ax1.plot(hrv_data, label='HRV Value', color='blue', alpha=0.7)
    ax1.set_ylabel('HRV Value')
    ax1.set_title('HRV Data and Detected Changepoints')
    ax1.grid(True, alpha=0.3)
    
    # Plot changepoint probabilities
    ax2.plot(changepoint_probs.cpu().numpy(), label='Changepoint Prob', color='red')
    ax2.axhline(y=threshold, color='green', linestyle=':', alpha=0.7, label=f'Threshold={threshold}')
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Time Step')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add vertical lines for detected changepoints on data plot
    for cp in detected_indices:
        ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('changepoint_plot.png')
    print("Plot saved to changepoint_plot.png")

if __name__ == "__main__":
    main()
