import numpy as np
import matplotlib.pyplot as plt
import time

#Generate data 
x = np.linspace(0, 10, 100)
m1_true, m2_true = 3, 2
m0 = 1
noise = np.random.normal(0, 1, len(x))
y = m1_true * x**2 + m2_true * x + noise  

plt.scatter(x, y, label='Data')
plt.title("Generated Data plot")
plt.legend()
plt.show()

#Loss Function
def loss_function(m1, m2, m0, x, y):
    y_pred = m1*x**2 + m2*x + m0
    return np.mean((y - y_pred)**2)

#Linear Search
m1_range = np.linspace(-5, 5, 21)
best_m1_ls, min_loss = None, float('inf')
start_ls = time.time()
for m1 in m1_range:
    curr_loss = loss_function(m1, m2_true, m0, x, y)
    if curr_loss < min_loss:
        min_loss = curr_loss
        best_m1_ls = m1
end_ls = time.time()
print(f"Linear Search: Best m1 = {best_m1_ls}, Loss = {min_loss:.6f}, Time = {end_ls - start_ls:.4f}s")

#Gradient Descent for Quadratic Function
def gradients(m1, m2, m0, x, y):
    y_pred = m1*x**2 + m2*x + m0
    error = y - y_pred
    grad_m1 = (-2/len(x)) * np.sum(x**2 * error)
    grad_m2 = (-2/len(x)) * np.sum(x * error)
    return grad_m1, grad_m2

start_gd = time.time()
np.random.seed(0)
m1_gd, m2_gd = np.random.rand(), np.random.rand()
lr = 1e-5
epochs = 500
loss_values = []
prev_loss = float('inf')

for i in range(epochs):
    g1, g2 = gradients(m1_gd, m2_gd, m0, x, y)
    m1_gd -= lr * g1
    m2_gd -= lr * g2
    
    curr_loss = loss_function(m1_gd, m2_gd, m0, x, y)
    loss_values.append(curr_loss)
    
    #Early stopper
    if abs(prev_loss - curr_loss) < 1e-6:
        print(f"Converged at epoch {i+1}")
        break
    prev_loss = curr_loss

end_gd = time.time()
print(f"Gradient Descent: Best m1 = {m1_gd:.4f} "
      f"Loss = {loss_values[-1]:.6f}, Time = {end_gd - start_gd:.4f}s")

# Plot LS loss
plt.plot(m1_range, [loss_function(m1, m2_true, m0, x, y) for m1 in m1_range])
plt.axvline(best_m1_ls, color='g', linestyle='--', label=f'Best m1 (LS): {best_m1_ls}')
plt.axvline(m1_gd, color='b', linestyle='--', label=f'Best m1 (GD): {m1_gd:.4f}')
plt.xlabel('m1 values')
plt.ylabel('Loss')
plt.title('Linear Search Loss')
plt.legend()
plt.show()


# Plot GD loss
plt.plot(loss_values, marker='o', color='b', label='GD Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Gradient Descent Convergence (No Intercept)')
plt.legend()
plt.show()
