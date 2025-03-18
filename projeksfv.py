import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Parameter grid
nx, ny = 50, 50
dx, dy = 1.0, 1.0
alpha = 1  # Difusivitas termal
dt = 0.01  # Reduced time step for stability
decay_factor = 0.99  # Decay factor for the source energy

# Inisialisasi grid suhu
u_current = np.zeros((nx, ny))

# Fungsi untuk memperbarui suhu
def update_heat(u, source_x, source_y, source_energy, alpha):
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )
    u_new[source_x, source_y] += source_energy
    return u_new

# Fungsi untuk memperbarui visualisasi
def update_plot(frame, img, u, source_x, source_y, source_energy, alpha):
    global u_current
    u_current = update_heat(u_current, source_x, source_y, source_energy, alpha)
    img.set_array(u_current)
    return img,

# Fungsi untuk menjalankan simulasi
def run_simulation():
    global u_current, source_x_slider, source_y_slider, source_energy_slider, alpha_slider, ani, source_energy
    u_current = np.zeros((nx, ny))
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.4)
    img = ax.imshow(u_current, cmap='hot', interpolation='nearest', vmin=0, vmax=1, origin='lower')
    plt.colorbar(img)

    # Slider untuk mengatur posisi dan energi titik panas
    ax_source_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_source_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_source_energy = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_alpha = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    source_x_slider = Slider(ax_source_x, 'X Pos', 0, nx-1, valinit=25, valstep=1)
    source_y_slider = Slider(ax_source_y, 'Y Pos', 0, ny-1, valinit=25, valstep=1)
    source_energy_slider = Slider(ax_source_energy, 'Energy', 1, 100, valinit=1, valstep=1)
    alpha_slider = Slider(ax_alpha, 'Alpha', 0.1, 10.0, valinit=1.0, valstep=0.1)

    # Button untuk menerapkan perubahan
    ax_apply = plt.axes([0.7, 0.025, 0.1, 0.04])
    apply_button = Button(ax_apply, 'Apply', color='lightgoldenrodyellow', hovercolor='0.975')

    # Button untuk mereset simulasi
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(ax_reset, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

    def apply_changes(event):
        global ani, source_energy
        source_x = int(source_x_slider.val)
        source_y = int(source_y_slider.val)
        source_energy = source_energy_slider.val
        alpha = alpha_slider.val
        logging.info(f'Applying changes: source_x={source_x}, source_y={source_y}, source_energy={source_energy}, alpha={alpha}')
        if 'ani' in globals():
            ani.event_source.stop()
        ani = FuncAnimation(fig, update_plot, fargs=(img, u_current, source_x, source_y, source_energy, alpha), frames=200, interval=50, blit=True)
        ani.event_source.start()

    def reset_simulation(event):
        global u_current, source_energy
        u_current = np.zeros((nx, ny))
        img.set_array(u_current)
        source_x_slider.reset()
        source_y_slider.reset()
        source_energy_slider.reset()
        alpha_slider.reset()
        source_energy = source_energy_slider.val
        logging.info('Simulation reset')
        if 'ani' in globals():
            ani.event_source.stop()

    apply_button.on_clicked(apply_changes)
    reset_button.on_clicked(reset_simulation)

    ani = FuncAnimation(fig, update_plot, fargs=(img, u_current, int(source_x_slider.val), int(source_y_slider.val), source_energy_slider.val, alpha_slider.val), frames=200, interval=50, blit=True)
    plt.show()

# Run the simulation
run_simulation()