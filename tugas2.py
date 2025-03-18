import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definisikan fungsi potensial
def V(x, y, a=10, b=8, q1=1, q2=2):
    return q1 / np.sqrt(x**2 + y**2) + q2 / np.sqrt((x - a)**2 + (y - b)**2)

# Definisikan turunan pertama
def dV_dx(x, y, a=10, b=8, q1=1, q2=2):
    return -q1 * x / (x**2 + y**2)**(3/2) - q2 * (x - a) / ((x - a)**2 + (y - b)**2)**(3/2)

def dV_dy(x, y, a=10, b=8, q1=1, q2=2):
    return -q1 * y / (x**2 + y**2)**(3/2) - q2 * (y - b) / ((x - a)**2 + (y - b)**2)**(3/2)

# Definisikan turunan kedua
def d2V_dx2(x, y, a=10, b=8, q1=1, q2=2):
    return q1 * (2*x**2 - y**2) / (x**2 + y**2)**(5/2) + q2 * (2*(x - a)**2 - (y - b)**2) / ((x - a)**2 + (y - b)**2)**(5/2)

def d2V_dy2(x, y, a=10, b=8, q1=1, q2=2):
    return q1 * (2*y**2 - x**2) / (x**2 + y**2)**(5/2) + q2 * (2*(y - b)**2 - (x - a)**2) / ((x - a)**2 + (y - b)**2)**(5/2)

def d2V_dxdy(x, y, a=10, b=8, q1=1, q2=2):
    return 3 * q1 * x * y / (x**2 + y**2)**(5/2) + 3 * q2 * (x - a) * (y - b) / ((x - a)**2 + (y - b)**2)**(5/2)

# Implementasi metode Newton-Raphson 2D
def newton_raphson_2d(x0, y0, tol=1e-6, max_iter=100):
    x, y = x0, y0
    history = [(x, y)]
    for _ in range(max_iter):
        # Hitung turunan pertama
        fx = dV_dx(x, y)
        fy = dV_dy(x, y)
        
        # Hitung turunan kedua
        fxx = d2V_dx2(x, y)
        fyy = d2V_dy2(x, y)
        fxy = d2V_dxdy(x, y)
        
        # Bentuk matriks Hessian
        H = np.array([[fxx, fxy], [fxy, fyy]])
        
        # Hitung invers Hessian
        H_inv = np.linalg.inv(H)
        
        # Update nilai x dan y
        delta = H_inv @ np.array([fx, fy])
        x -= delta[0]
        y -= delta[1]
        
        history.append((x, y))
        
        # Cek konvergensi
        if np.linalg.norm(delta) < tol:
            break
    
    return x, y, history

# Inisialisasi titik awal
x0, y0 = 5, 5

# Cari titik kritis
x_crit, y_crit, history = newton_raphson_2d(x0, y0)

# Uji apakah titik kritis adalah saddle point
H = np.array([[d2V_dx2(x_crit, y_crit), d2V_dxdy(x_crit, y_crit)], [d2V_dxdy(x_crit, y_crit), d2V_dy2(x_crit, y_crit)]])
det_H = np.linalg.det(H)

if det_H < 0:
    print(f"Titik ({x_crit}, {y_crit}) adalah saddle point.")
else:
    print(f"Titik ({x_crit}, {y_crit}) bukan saddle point.")

# Visualisasi grafik 3D
x = np.linspace(-5, 15, 400)
y = np.linspace(-5, 15, 400)
X, Y = np.meshgrid(x, y)
Z = V(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.plot([x_crit], [y_crit], [V(x_crit, y_crit)], markerfacecolor='r', markeredgecolor='r', marker='o', markersize=5, alpha=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('V(x, y)')
ax.set_title('Grafik 3D Potensial Elektrostatik dan Titik Kritis')
plt.show()