import numpy as np
import matplotlib.pyplot as plt
import time

# Function for generating synthetic quadratic data
def generate_data(x, m1, m2, m0):
    """
    Generate quadratic data with additive Gaussian noise.
    
    Args:
        x (np.ndarray): Input data points.
        m1 (float): Coefficient for x^2 term.
        m2 (float): Coefficient for x term.
        m0 (float): Constant offset.
        
    Returns:
        np.ndarray: Quadratic data with noise.
    """
    noise = np.random.normal(0, 1, len(x))
    return m1 * x**2 + m2 * x + m0 + noise

# Function for calculating loss values
def loss_function(m1, m2, m0, x, y):
    """
    Compute the Mean Squared Error (MSE) loss for quadratic predictions.
    
    Args:
        m1, m2, m0 (float): Parameters for the quadratic model.
        x (np.ndarray): Input data points.
        y (np.ndarray): True output data points.
        
    Returns:
        float: MSE loss value.
    """
    y_pred = m1 * x**2 + m2 * x + m0
    return np.mean((y - y_pred)**2)

# Function for computing gradient values
def compute_gradients(m1, m2, m0, x, y):
    """
    Compute gradients of the MSE loss with respect to m1, m2, and m0.
    
    Args:
        m1, m2, m0 (float): Parameters for the quadratic model.
        x (np.ndarray): Input data points.
        y (np.ndarray): True output data points.
        
    Returns:
        tuple: Gradients (grad_m1, grad_m2, grad_m0) for each parameter.
    """
    y_pred = m1 * x**2 + m2 * x + m0
    error = y - y_pred
    n = len(x)
    grad_m1 = (-2 / n) * np.sum(x**2 * error)
    grad_m2 = (-2 / n) * np.sum(x * error)
    grad_m0 = (-2 / n) * np.sum(error)
    return grad_m1, grad_m2, grad_m0

# Function for implementing linear search
def linear_search(m1_range, m2, m0, x, y):
    """
    Perform a linear search over a range of m1 values to minimize the loss.
    
    Args:
        m1_range (np.ndarray): Array of candidate m1 values.
        m2 (float): Fixed parameter m2.
        m0 (float): Fixed parameter m0.
        x (np.ndarray): Input data points.
        y (np.ndarray): True output data points.
        
    Returns:
        tuple: Best m1 value and its corresponding loss.
    """
    losses = list(map(lambda m1: loss_function(m1, m2, m0, x, y), m1_range))
    best_index = np.argmin(losses)
    return m1_range[best_index], losses[best_index]

# Function for iterative gradient descent
def gradient_descent(x, y, lr=1e-4, max_epochs=500, stop_threshold=1e-2, patience=10):
    """
    Iterative gradient descent to optimize m1, m2, m0.
    
    Args:
        x, y (np.ndarray): Data points.
        lr (float): Learning rate.
        max_epochs (int): Maximum iterations.
        stop_threshold (float): Convergence threshold for loss.
        patience (int): Number of epochs to wait for convergence before stopping.
        
    Returns:
        tuple: Optimized parameters (m1, m2, m0) and loss history.
    """
    np.random.seed(42)
    m1, m2, m0 = np.random.rand(), np.random.rand(), np.random.rand()
    loss_history = [loss_function(m1, m2, m0, x, y)]
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        # Compute gradients
        g1, g2, g0 = compute_gradients(m1, m2, m0, x, y)
        
        # Update parameters
        m1 -= lr * g1
        m2 -= lr * g2
        m0 -= lr * g0
        
        # Compute new loss
        current_loss = loss_function(m1, m2, m0, x, y)
        loss_history.append(current_loss)
        
        # Early stopping based on loss change
        if abs(loss_history[-2] - current_loss) < stop_threshold:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Converged after {epoch} epochs")
                break
        else:
            patience_counter = 0
    
    return m1, m2, m0, loss_history

def main():
    # Generate synthetic quadratic data
    x = np.linspace(0, 10, 100)
    m1_true, m2_true, m0_true = 3, 2, 1
    y = generate_data(x, m1_true, m2_true, m0_true)
    
    # Linear search 
    start_time = time.perf_counter()
    m1_range = np.linspace(-5, 5, 21)
    best_m1_ls, min_loss_ls = linear_search(m1_range, m2_true, m0_true, x, y)
    ls_time = time.perf_counter() - start_time
    print(f"Linear Search: Best m1 = {best_m1_ls}, Loss = {min_loss_ls:.6f}, Time = {ls_time:.6f}")
    
    # Gradient descent
    start_time_gd = time.perf_counter()
    m1_gd, m2_gd, m0_gd, loss_history = gradient_descent(x, y)
    gd_time = time.perf_counter() - start_time_gd
    print(f"Gradient Descent: Best m1 = {m1_gd:.4f}, Loss = {loss_history[-1]:.6f}, Time = {gd_time:.6f}")
    
    # Plot linear search loss vs. m1 values
    plt.figure()
    ls_losses = list(map(lambda m1: loss_function(m1, m2_true, m0_true, x, y), m1_range))
    plt.plot(m1_range, ls_losses, label='Linear Search Loss')
    plt.axvline(best_m1_ls, color='g', linestyle='--', label=f'Best m1 (LS): {best_m1_ls}')
    plt.axvline(m1_gd, color='b', linestyle='--', label=f'Best m1 (GD): {m1_gd:.4f}')
    plt.xlabel('m1 values')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title('Linear Search Loss')
    plt.legend()
    plt.show()
    
    # Plot gradient descent loss convergence
    plt.figure()
    plt.plot(loss_history, color='b', label='Gradient Descent Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Gradient Descent Convergence')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
