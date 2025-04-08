import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

# System Parameters
M = 1.0       # Cart mass (kg)
m = 0.1       # Pendulum mass (kg)
l = 0.5       # Pendulum length (m)
g = 9.81      # Gravity (m/s²)

# Simulation Parameters
dt = 0.01     # Time step (s)
T_sim = 10.0  # Total simulation time (s)
steps = int(T_sim / dt)

# LQR Parameters
# State vector: [p, p_dot, theta, theta_dot]
# Higher values in Q penalize deviations more heavily
Q = np.diag([1.0, 1.0, 10.0, 1.0])  # State cost matrix (higher for theta)
R = np.array([[1.0]])               # Control cost matrix

# Nonlinear Dynamics for Cart-Pole System
def cart_pole_dynamics(t, state, u=0):
    """
    Compute the derivatives for the cart-pole system.
    
    Args:
        t: Time (not used, required for solve_ivp)
        state: State vector [p, p_dot, theta, theta_dot]
        u: Control input (force applied to cart)
    
    Returns:
        state_dot: Derivative of state vector
    """
    p, p_dot, theta, theta_dot = state
    
    # Intermediate calculations
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Denominator term
    den = M + m * sin_theta**2
    
    # Accelerations
    theta_ddot = (g * sin_theta - cos_theta * (u + m * l * theta_dot**2 * sin_theta) / (M + m)) / (l * (1 - m * cos_theta**2 / (M + m)))
    p_ddot = (u + m * l * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / den
    
    return np.array([p_dot, p_ddot, theta_dot, theta_ddot])

# Linearize the Cart-Pole System around the upright equilibrium point
def linearize_system():
    """
    Linearize the cart-pole system around the upright equilibrium point.
    
    Returns:
        A: State matrix
        B: Input matrix
    """
    # Linearized model: x_dot = Ax + Bu where x = [p, p_dot, theta, theta_dot]
    # At equilibrium [p_eq, 0, 0, 0]
    
    # Linearized state matrix
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, -m*g/M, 0],
        [0, 0, 0, 1],
        [0, 0, (M+m)*g/(M*l), 0]
    ])
    
    # Linearized input matrix
    B = np.array([
        [0],
        [1/M],
        [0],
        [-1/(M*l)]
    ])
    
    return A, B

# Discretize the system
def discretize_system(A, B, dt):
    """
    Convert continuous-time system to discrete-time.
    
    Args:
        A: Continuous-time state matrix
        B: Continuous-time input matrix
        dt: Time step
    
    Returns:
        Ad: Discrete-time state matrix
        Bd: Discrete-time input matrix
    """
    # Using scipy's cont2discrete 
    # Method 'zoh' stands for zero-order hold
    disc_sys = cont2discrete((A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], 1))), dt, method='zoh')
    Ad, Bd = disc_sys[0], disc_sys[1]
    
    return Ad, Bd

# Design LQR Controller
def design_lqr_controller(A, B, Q, R, discrete=False):
    """
    Design an LQR controller for the system.
    
    Args:
        A: State matrix
        B: Input matrix
        Q: State cost matrix
        R: Control cost matrix
        discrete: Whether system is discrete or continuous
    
    Returns:
        L: Optimal gain matrix
        P: Solution to Riccati equation
    """
    if discrete:
        P = solve_discrete_are(A, B, Q, R)
        BT_P_B = B.T @ P @ B
        R_BT_P_B = R + BT_P_B
        L = np.linalg.inv(R_BT_P_B) @ B.T @ P @ A
    else:
        P = solve_continuous_are(A, B, Q, R)
        L = np.linalg.inv(R) @ B.T @ P
    
    return L, P

# Simulate the system with the LQR controller
def simulate_system(x0, L, add_noise=False, noise_std_dev=0.01, add_disturbance=False, disturbance_times=None, disturbance_magnitude=5.0):
    """
    Simulate the nonlinear cart-pole system with LQR control.
    
    Args:
        x0: Initial state
        L: LQR gain matrix
        add_noise: Whether to add measurement noise
        noise_std_dev: Standard deviation of measurement noise
        add_disturbance: Whether to add disturbances
        disturbance_times: Times at which to apply disturbances
        disturbance_magnitude: Magnitude of disturbances
    
    Returns:
        t_history: Time history
        x_history: State history
        u_history: Control input history
    """
    # Initialize state and history arrays
    x = x0.copy()
    
    t_history = np.arange(0, T_sim, dt)
    x_history = np.zeros((len(t_history), len(x0)))
    u_history = np.zeros(len(t_history))
    
    # Equilibrium state
    x_eq = np.array([0, 0, 0, 0])
    
    # Simulation loop
    for i in range(len(t_history)):
        t = t_history[i]
        
        # Store current state
        x_history[i] = x
        
        # Calculate deviation from equilibrium
        delta_x = x - x_eq
        
        # Add measurement noise if requested
        if add_noise:
            delta_x_measured = delta_x + np.random.normal(0, noise_std_dev, size=delta_x.shape)
        else:
            delta_x_measured = delta_x
        
        # Calculate control input
        u = -L @ delta_x_measured
        # Extract the scalar value from the control input matrix
        u_scalar = float(u[0])
        
        # Add disturbance if requested
        if add_disturbance and disturbance_times is not None:
            for d_time in disturbance_times:
                if abs(t - d_time) < dt:
                    u_scalar += disturbance_magnitude
        
        # Store control input
        u_history[i] = u_scalar
        
        # Simulate one step using solve_ivp for accuracy
        solution = solve_ivp(lambda t, y: cart_pole_dynamics(t, y, u_scalar), [0, dt], x, method='RK45')
        x = solution.y[:, -1]
    
    return t_history, x_history, u_history

# Visualization functions
def plot_state_trajectories(t, x_history, u_history, title=""):
    """Plot the state trajectories and control input."""
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(t, x_history[:, 0])
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title(title)
    axes[0].grid(True)
    
    axes[1].plot(t, x_history[:, 1])
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].grid(True)
    
    axes[2].plot(t, x_history[:, 2])
    axes[2].set_ylabel('Angle (rad)')
    axes[2].grid(True)
    
    axes[3].plot(t, x_history[:, 3])
    axes[3].set_ylabel('Angular Velocity (rad/s)')
    axes[3].grid(True)
    
    axes[4].plot(t, u_history)
    axes[4].set_ylabel('Force (N)')
    axes[4].set_xlabel('Time (s)')
    axes[4].grid(True)
    
    plt.tight_layout()
    return fig

def plot_phase_portrait(x_history, title=""):
    """Plot the phase portrait (theta_dot vs. theta)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x_history[:, 2], x_history[:, 3])
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title(title)
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def animate_pendulum(x_history, filename=None):
    """Create an animation of the cart-pole system."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the limits of the plot
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 1.5)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Create artists for the cart and pendulum
    cart_width, cart_height = 0.3, 0.2
    cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, 
                         fill=True, color='blue', ec='black')
    pendulum, = ax.plot([], [], 'k-', lw=2)
    pendulum_bob = plt.Circle((0, 0), 0.05, fill=True, color='red')
    
    # Add the artists to the axes
    ax.add_patch(cart)
    ax.add_patch(pendulum_bob)
    
    # Vertical reference line for the upright position
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Text displays
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        """Initialize the animation."""
        pendulum.set_data([], [])
        time_text.set_text('')
        cart.set_xy((-cart_width/2, -cart_height/2))
        pendulum_bob.center = (0, 0)
        return pendulum, cart, pendulum_bob, time_text
    
    def animate(i):
        """Update the animation for frame i."""
        p = x_history[i, 0]
        theta = x_history[i, 2]
        
        # Cart position
        cart.set_xy((p - cart_width/2, -cart_height/2))
        
        # Pendulum position
        x_pendulum = [p, p + l * np.sin(theta)]
        y_pendulum = [0, -l * np.cos(theta)]
        pendulum.set_data(x_pendulum, y_pendulum)
        
        # Pendulum bob position
        pendulum_bob.center = (x_pendulum[1], y_pendulum[1])
        
        time_text.set_text(f'Time: {i*dt:.2f}s')
        
        return pendulum, cart, pendulum_bob, time_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=len(x_history),
                         init_func=init, blit=True, interval=dt*1000)
    
    # Save animation if filename provided
    if filename:
        anim.save(filename, writer='ffmpeg', fps=1/dt)
    
    plt.close()
    return anim

def compare_q_r_tuning():
    """Compare different Q and R tuning parameters."""
    A, B = linearize_system()
    
    # Initial state (slight angle perturbation)
    x0 = np.array([0, 0, 0.1, 0])
    
    # Different Q and R values to compare
    Q_values = [
        np.diag([1.0, 1.0, 10.0, 1.0]),   # Base case
        np.diag([1.0, 1.0, 100.0, 1.0]),  # Higher penalty on angle
        np.diag([1.0, 1.0, 10.0, 10.0])   # Higher penalty on angular velocity
    ]
    
    R_values = [
        np.array([[1.0]]),   # Base case
        np.array([[10.0]]),  # Higher penalty on control
        np.array([[0.1]])    # Lower penalty on control
    ]
    
    # Combine Q and R variations
    labels = [
        "Base Case (Q angle=10, R=1)",
        "High Angle Penalty (Q angle=100, R=1)",
        "High Angular Velocity Penalty (Q angle_dot=10, R=1)",
        "High Control Penalty (Q angle=10, R=10)",
        "Low Control Penalty (Q angle=10, R=0.1)"
    ]
    
    params = [
        (Q_values[0], R_values[0]),
        (Q_values[1], R_values[0]),
        (Q_values[2], R_values[0]),
        (Q_values[0], R_values[1]),
        (Q_values[0], R_values[2])
    ]
    
    # Run simulations and collect results
    t_history = None
    angle_histories = []
    control_histories = []
    
    for Q, R in params:
        L, _ = design_lqr_controller(A, B, Q, R)
        t, x_history, u_history = simulate_system(x0, L)
        
        if t_history is None:
            t_history = t
        
        angle_histories.append(x_history[:, 2])  # Pendulum angle
        control_histories.append(u_history)      # Control input
    
    # Plot angle comparison
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(angle_histories):
        plt.plot(t_history, history, label=labels[i])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum Angle (rad)')
    plt.title('Effect of Q and R Tuning on Pendulum Angle')
    plt.legend()
    plt.grid(True)
    plt.savefig('tuning_comparison_angle.png')
    
    # Plot control comparison
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(control_histories):
        plt.plot(t_history, history, label=labels[i])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Control Force (N)')
    plt.title('Effect of Q and R Tuning on Control Effort')
    plt.legend()
    plt.grid(True)
    plt.savefig('tuning_comparison_control.png')

def test_robustness():
    """Test controller robustness to noise and disturbances."""
    A, B = linearize_system()
    L, _ = design_lqr_controller(A, B, Q, R)
    
    # Initial state (slight angle perturbation)
    x0 = np.array([0, 0, 0.1, 0])
    
    # Baseline (no noise or disturbance)
    t, x_baseline, u_baseline = simulate_system(x0, L)
    
    # With measurement noise
    t, x_noise, u_noise = simulate_system(
        x0, L, add_noise=True, noise_std_dev=0.02
    )
    
    # With disturbances at specific times
    disturbance_times = [2.0, 5.0, 8.0]
    t, x_disturbance, u_disturbance = simulate_system(
        x0, L, add_disturbance=True, 
        disturbance_times=disturbance_times,
        disturbance_magnitude=5.0
    )
    
    # Visualization of robustness
    # Angle comparison
    plt.figure(figsize=(12, 6))
    plt.plot(t, x_baseline[:, 2], label='Baseline')
    plt.plot(t, x_noise[:, 2], label='With Measurement Noise')
    plt.plot(t, x_disturbance[:, 2], label='With Disturbances')
    
    # Mark disturbance times
    for d_time in disturbance_times:
        plt.axvline(x=d_time, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum Angle (rad)')
    plt.title('Robustness of LQR Controller to Noise and Disturbances')
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_angle.png')
    
    # Control input comparison
    plt.figure(figsize=(12, 6))
    plt.plot(t, u_baseline, label='Baseline')
    plt.plot(t, u_noise, label='With Measurement Noise')
    plt.plot(t, u_disturbance, label='With Disturbances')
    
    # Mark disturbance times
    for d_time in disturbance_times:
        plt.axvline(x=d_time, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Control Force (N)')
    plt.title('Controller Response to Noise and Disturbances')
    plt.legend()
    plt.grid(True)
    plt.savefig('robustness_control.png')

def simulate_uncontrolled():
    """Simulate the system without control for comparison."""
    # Initial state (slight angle perturbation)
    x0 = np.array([0, 0, 0.1, 0])
    
    # Initialize state and history arrays
    x = x0.copy()
    
    t_history = np.arange(0, T_sim, dt)
    x_history = np.zeros((len(t_history), len(x0)))
    u_history = np.zeros(len(t_history))  # No control, all zeros
    
    # Simulation loop
    for i in range(len(t_history)):
        # Store current state
        x_history[i] = x
        
        # Simulate one step with zero control (u=0)
        solution = solve_ivp(lambda t, y: cart_pole_dynamics(t, y, 0.0), [0, dt], x, method='RK45')
        x = solution.y[:, -1]
        
        # Stop if the pendulum falls too far
        if abs(x[2]) > np.pi/2:  # More than 90 degrees
            # Fill remaining history with the last state
            x_history[i+1:] = x_history[i]
            break
    
    return t_history, x_history, u_history

def analyze_real_data(data_file, estimate_params=False):
    """
    Analyze real pendulum data and compare with model predictions.
    
    Args:
        data_file: CSV file with real pendulum data (time, position, angle)
        estimate_params: Whether to estimate system parameters from the data
        
    Returns:
        Comparison plots and analysis results
    """
    # Load real data
    print(f"Loading real data from {data_file}...")
    try:
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        # Expected format: time, cart_position, cart_velocity, pendulum_angle, pendulum_angular_velocity
        
        t_real = data[:, 0]
        x_real = data[:, 1:5]  # All state variables
        
        # Check if control input is included in data
        if data.shape[1] > 5:
            u_real = data[:, 5]
        else:
            u_real = np.zeros(len(t_real))
            
        print(f"Loaded {len(t_real)} data points")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # If requested, estimate system parameters from the data
    if estimate_params:
        print("Estimating system parameters from data...")
        estimated_params = estimate_system_parameters(t_real, x_real, u_real)
        M_est, m_est, l_est, g_est = estimated_params
        print(f"Estimated parameters: M={M_est:.3f}, m={m_est:.3f}, l={l_est:.3f}, g={g_est:.3f}")
    
    # Run simulation using the same initial state and control inputs as the real data
    print("Simulating with model using real initial conditions and inputs...")
    x0 = x_real[0]  # Use initial state from real data
    
    # Initialize state and history arrays for simulation
    x = x0.copy()
    t_model = t_real.copy()
    dt_real = t_real[1] - t_real[0] if len(t_real) > 1 else dt
    x_model = np.zeros((len(t_real), len(x0)))
    
    # Simulation loop
    for i in range(len(t_real)):
        # Store current state
        x_model[i] = x
        
        # Get control input from real data
        u_i = u_real[i]
        
        # Only simulate to next time step if not at the end
        if i < len(t_real) - 1:
            # Calculate time step
            dt_i = t_real[i+1] - t_real[i]
            
            # Simulate one step using real control input
            solution = solve_ivp(lambda t, y: cart_pole_dynamics(t, y, u_i), 
                                [0, dt_i], x, method='RK45')
            x = solution.y[:, -1]
    
    # Compare real data with model predictions
    print("Creating comparison plots...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    state_labels = ['Cart Position (m)', 'Cart Velocity (m/s)', 
                   'Pendulum Angle (rad)', 'Pendulum Angular Velocity (rad/s)']
    
    for i in range(4):
        axes[i].plot(t_real, x_real[:, i], 'b-', label='Real Data')
        axes[i].plot(t_model, x_model[:, i], 'r--', label='Model Prediction')
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)
        
        # Add error metrics
        rmse = np.sqrt(np.mean((x_real[:, i] - x_model[:, i])**2))
        axes[i].text(0.02, 0.9, f'RMSE: {rmse:.4f}', transform=axes[i].transAxes)
    
    axes[0].set_title('Real Data vs. Model Prediction')
    axes[-1].set_xlabel('Time (s)')
    axes[0].legend()
    
    plt.tight_layout()
    fig.savefig('real_data_comparison.png')
    print("Comparison plot saved as 'real_data_comparison.png'")
    
    return fig

def estimate_system_parameters(t_data, x_data, u_data):
    """
    Estimate system parameters (M, m, l, g) from real data using optimization.
    
    Args:
        t_data: Time series
        x_data: State data
        u_data: Control input data
        
    Returns:
        Estimated parameters [M, m, l, g]
    """
    from scipy.optimize import minimize
    
    def simulation_error(params):
        """Objective function: error between real data and simulated data with given params."""
        global M, m, l, g
        
        # Save original parameters
        M_orig, m_orig, l_orig, g_orig = M, m, l, g
        
        # Set parameters to the test values
        M, m, l, g = params
        
        # Run simulation
        x0 = x_data[0]
        x = x0.copy()
        x_sim = np.zeros((len(t_data), len(x0)))
        
        for i in range(len(t_data)):
            x_sim[i] = x
            
            if i < len(t_data) - 1:
                dt_i = t_data[i+1] - t_data[i]
                solution = solve_ivp(lambda t, y: cart_pole_dynamics(t, y, u_data[i]), 
                                    [0, dt_i], x, method='RK45')
                x = solution.y[:, -1]
        
        # Restore original parameters
        M, m, l, g = M_orig, m_orig, l_orig, g_orig
        
        # Calculate error (using weighted RMSE)
        error = np.sqrt(np.mean((x_data - x_sim)**2, axis=0))
        # Weight angle error more heavily
        weighted_error = error[0] + error[1] + 5*error[2] + 2*error[3]
        
        return weighted_error
    
    # Initial guess
    initial_params = [M, m, l, g]
    
    # Parameter bounds (realistic physical constraints)
    bounds = [
        (0.1, 10.0),  # M: 0.1 to 10 kg
        (0.01, 1.0),  # m: 0.01 to 1 kg
        (0.1, 2.0),   # l: 0.1 to 2 meters
        (9.7, 9.9)    # g: 9.7 to 9.9 m/s² (narrow range around Earth gravity)
    ]
    
    # Run optimization
    result = minimize(simulation_error, initial_params, bounds=bounds, method='L-BFGS-B')
    
    return result.x

def main():
    """Main function to run the simulation and generate visualizations."""
    # Linearize the system
    print("Linearizing the system...")
    A, B = linearize_system()
    print("A =\n", A)
    print("B =\n", B)
    
    # Design LQR controller
    print("\nDesigning LQR controller...")
    L, P = design_lqr_controller(A, B, Q, R)
    print("L =", L)
    
    # Initial state with a slight angle perturbation
    x0 = np.array([0, 0, 0.1, 0])
    
    # Simulate the controlled system
    print("\nSimulating the controlled system...")
    t, x_controlled, u_controlled = simulate_system(x0, L)
    
    # Simulate the uncontrolled system
    print("Simulating the uncontrolled system...")
    t, x_uncontrolled, u_uncontrolled = simulate_uncontrolled()
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # VIZ 1: State Trajectories (controlled vs uncontrolled)
    fig_controlled = plot_state_trajectories(t, x_controlled, u_controlled, 
                                            "Controlled Inverted Pendulum - State Trajectories")
    fig_controlled.savefig('controlled_state_trajectories.png')
    
    fig_uncontrolled = plot_state_trajectories(t, x_uncontrolled, u_uncontrolled,
                                              "Uncontrolled Inverted Pendulum - State Trajectories")
    fig_uncontrolled.savefig('uncontrolled_state_trajectories.png')
    
    # VIZ 3: Phase Plot
    fig_phase = plot_phase_portrait(x_controlled, "Phase Portrait (theta_dot vs. theta)")
    fig_phase.savefig('phase_portrait.png')
    
    # VIZ 4: Animation
    print("Creating animation...")
    anim = animate_pendulum(x_controlled, 'pendulum_animation.mp4')
    
    # VIZ 5: Compare Q/R tuning
    print("Comparing different Q/R tunings...")
    compare_q_r_tuning()
    
    # VIZ 6-7: Test robustness
    print("Testing controller robustness...")
    test_robustness()
    
    # Check if real data file is provided as command line argument
    import sys
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"\nAnalyzing real data from {data_file}...")
        analyze_real_data(data_file, estimate_params=True)
    
    print("\nSimulation complete. Visualizations saved.")

if __name__ == "__main__":
    main() 