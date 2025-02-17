import torch
import numpy as np
import matplotlib.pyplot as plt

from PANDABetaO import PANDABetaO
# from PANDAFlash import PANDAFlash
from MAX import PANDA
# from PANDAMaster import PANDAMaster
from PANDABeta import PANDABeta
from AdEMAMix import AdEMAMix
from AdaBelief import AdaBelief
from Lion import Lion
# Define the complex loss function
def loss_function(x, y):

    return 0.5 * ((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y))


# Find the minimum (numerically)
def find_target_point(start_point=(2.5, -2.5), lr=0.01, steps=500):
    point = torch.tensor(start_point, requires_grad=True)
    optimizer = torch.optim.Adam([point], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_function(point[0], point[1])
        loss.backward()
        optimizer.step()
    return point.detach().numpy()

# Function to record optimizer steps
def optimizer_steps_complex(optimizer_name, initial_lr=0.01, steps=50, start_point=(2.5, -2.5)):
    initial_point = torch.tensor(start_point, requires_grad=True)
    optimizer = optimizer_name([initial_point], lr=initial_lr)
    path = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_function(initial_point[0], initial_point[1])
        path.append(initial_point.detach().numpy().copy())
        loss.backward()
        optimizer.step()
    return np.array(path)

# Function to automatically set plotting range
def determine_plot_range(paths, target_point, margin=0.5):
    all_points = np.concatenate(paths, axis=0)
    x_min, x_max = min(all_points[:, 0].min(), target_point[0]) - margin, max(all_points[:, 0].max(), target_point[0]) + margin
    y_min, y_max = min(all_points[:, 1].min(), target_point[1]) - margin, max(all_points[:, 1].max(), target_point[1]) + margin
    return x_min, x_max, y_min, y_max
COLOS = [
    'Blues', 'Reds', 'Greens', 'Purples',
    'Oranges', 'Pink', 'Greys', 'YlGn',
    'YlGnBu', 'PuBu', 'PuRd', 'GnBu',
    'BuGn', 'BuPu', 'OrRd', 'YlOrBr'
]

# Function to compare multiple optimizers
def compare_optimizers(optimizers, labels, initial_lr=0.01, steps=50, start_point=(2.5, -2.5), target_point=None):
    if target_point is None:
        target_point = find_target_point(start_point)

    paths = []
    for i, optimizer in enumerate(optimizers):
        lr = initial_lr

        lr = 0.001 if labels[i] == "PANDA-Focus" else lr
        lr = 0.008 if labels[i] == "Lion" else lr
        lr = 0.037 if labels[i] == "PANDA-Master" else lr
        lr = 0.03 if labels[i] == "Adam" else lr
        lr = 0.03 if labels[i] == "AdamW" else lr
        lr = 0.03 if labels[i] == "AdEMAMix" else lr
        lr = 0.001 if labels[i] == "AdaBelief" else lr
        lr = 0.0037 if labels[i] == "SGD" else lr
        path = optimizer_steps_complex(optimizer, lr, steps, start_point)
        paths.append(path)

    # Determine dynamic plot range
    x_min, x_max, y_min, y_max = determine_plot_range(paths, target_point)

    # Create grid for contour plotting
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    X_torch = torch.tensor(X, dtype=torch.float32)
    Y_torch = torch.tensor(Y, dtype=torch.float32)
    Z_torch = loss_function(X_torch, Y_torch).detach().numpy()

    # Plot 1: Trajectories on the contour map
    plt.figure(figsize=(12, 6))
    plt.contour(X, Y, Z_torch, levels=100, cmap='viridis')
    for path, label in zip(paths, labels):
        plt.plot(path[:, 0], path[:, 1], 'o-', label=f'{label} Trajectory', markersize=2, alpha=0.8)
    plt.scatter(*target_point, color='red', marker='*', s=150, label='Target Point')  # Mark the target point
    plt.title('Optimizer Trajectories on Complex Loss Landscape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Plot 2: Loss value over iterations
    plt.figure(figsize=(12, 6))
    for path, label in zip(paths, labels):
        losses = [loss_function(torch.tensor(p[0]), torch.tensor(p[1])).item() for p in path]
        plt.plot(losses, label=f'{label}', marker='o', markersize=2)
    plt.title('Loss Value Over Iterations on Complex Landscape')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

    # Plot 3: Density heatmap of paths
    plt.figure(figsize=(12, 6))
    for path, label, cmap in zip(paths, labels, COLOS[:5]):
        plt.hist2d(path[:, 0], path[:, 1], bins=50, alpha=0.6, cmap=cmap, label=f'{label} Density')
    # plt.scatter(*target_point, color='red', marker='*', s=150, label='Target Point')  # Mark the target point
    plt.title('Optimizer Path Density on Complex Landscape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Density')
    plt.legend()
    plt.show()



# Example usage: Compare Adam and AdamW
optimizers = [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, AdEMAMix, Lion, AdaBelief, PANDA]
labels = ["SGD", "Adam", "AdamW", "AdEMAMix", "Lion", "AdaBelief", "PANDA"]
compare_optimizers(optimizers, labels, initial_lr=0.001, steps=100, start_point=(2.5, -2.5))
