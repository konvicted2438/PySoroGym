import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numba import njit, float64

# ---------------------------------------------------------------
@njit(float64[:, :](float64[:]), cache=True, fastmath=True)
def hat_nb(v):
    H = np.zeros((3, 3), dtype=np.float64)
    H[0,1] = -v[2]; H[0,2] =  v[1]
    H[1,0] =  v[2]; H[1,2] = -v[0]
    H[2,0] = -v[1]; H[2,1] =  v[0]
    return H

@njit(cache=True, fastmath=True)
def stiff_flop_odes_nb(y, pressure_chamber, Kse, Kbt,
                       A_chambers, e3, rhoAg, r_chamber):
    fp = np.zeros(3)
    lp = np.zeros(3)

    # unpack state
    p  = y[:3]
    R  = y[3:12].reshape((3, 3))
    n  = y[12:15]
    m  = y[15:18]

    # constitutive
    u = np.linalg.solve(Kbt, R.T @ m)        # Kbt⁻¹
    v = np.linalg.solve(Kse, R.T @ n) + e3   # Kse⁻¹ + e3

    ps = R @ v
    Rs = R @ hat_nb(u)

    # chamber forces
    for i in range(6):
        coeff = 1e5 * pressure_chamber[i] * A_chambers
        fp -= coeff * (Rs @ e3)
        lp -= coeff * R @ (hat_nb(v + hat_nb(u) @ r_chamber[:, i]) @ e3 +
                       hat_nb(r_chamber[:, i]) @ (hat_nb(u) @ e3))

    ns = -rhoAg - fp
    ms = -hat_nb(ps) @ n - lp

    out = np.empty(18)
    out[:3]   = ps
    out[3:12] = Rs.ravel()
    out[12:15] = ns
    out[15:18] = ms
    return out
# ---------------------------------------------------------------

class SoftRobotKinematics:
    def __init__(self, L=0.042, rad_robot=0.0075, rad_inner=5.4/2000, 
                 rad_chamber=2.5/2000, rad_chamber_pos=0.0051, 
                 num_chambers=6, rho=1040, n_element=5):
        """Initialize the soft robot kinematics solver with specific parameters."""
        # Global variables equivalent
        self.last_guess = np.zeros(9)
        self.compu_time = []
        self.y_tip_all = []
        
        # Robot parameters
        self.L = L
        self.rad_chamber_position = rad_chamber_pos
        self.rho = rho
        self.rad_inner = rad_inner
        self.rad_chamber = rad_chamber
        self.rad_robot = rad_robot
        self.n_element = n_element
        self.num_chambers = num_chambers
        
        # Physical constants
        self.g = np.array([0, 0, 9.81 * 1])  # gravity
        self.ee_mass = 0  # end effector mass
        self.e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        
        # Calculate geometric properties
        self.A_chambers = np.pi * self.rad_chamber**2  # chamber cross-sectional area
        self.A_cross_section = (np.pi * self.rad_robot**2 - 
                               np.pi * self.rad_inner**2 - 
                               6 * np.pi * self.rad_chamber**2)  # robot cross-sectional area
        
        # Moment of inertia calculation
        self.I = (np.pi * (2*self.rad_robot)**4/64 - 
                 np.pi * (2*self.rad_inner)**4/64 -
                 (2*np.pi * (2*self.rad_chamber)**4/64) -
                 2*(2*np.pi * (2*self.rad_chamber)**4/64 +
                 2*3/4*np.pi*self.rad_chamber**2*(self.rad_chamber_position)**2))
        
        self.J = 2 * self.I
        self.rhoAg = self.rho * self.A_cross_section * self.g
        
        # Initial conditions
        self.R0 = np.eye(3)
        self.p0 = np.zeros(3)
        
        # Chamber positions (hole pattern)
        alpha1 = 60 * np.pi / 180
        alpha2 = 120 * np.pi / 180 - alpha1
        i_chamber = np.arange(1, 7)  # 1 to 6
        theta_B = (-alpha2 + (i_chamber - i_chamber % 2) * alpha2 + 
                  (i_chamber - 1 - (i_chamber - 1) % 2) * alpha1) / 2
        
        self.r_chamber = self.rad_chamber_position * np.array([
            np.cos(theta_B),
            np.sin(theta_B),
            np.zeros(6)
        ])
        
        # Tip forces and moments
        self.F = self.ee_mass * self.g + np.array([0.0, 0.0, 0])
        self.M = np.array([0, 0, 0])
        
        # -----------------------------------------------------------------
        # Numba warm-up (compile kernels once with dummy data – <1 ms run-time)
        y_dummy   = np.zeros(18, dtype=np.float64)
        p_dummy   = np.zeros(6,  dtype=np.float64)
        Kse_dummy = np.eye(3)
        Kbt_dummy = np.eye(3)
        stiff_flop_odes_nb(y_dummy, p_dummy, Kse_dummy, Kbt_dummy,
                           self.A_chambers, self.e3,
                           self.rhoAg, self.r_chamber)
        hat_nb(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        # -----------------------------------------------------------------

    # --- JIT hat ----------------------------------------------------
    def hat(self, v):              # << replaces the old @staticmethod
        return hat_nb(v)
    # ---------------------------------------------------------------

    # --- JIT stiff ODE ---------------------------------------------
    def stiff_flop_odes(self, s, y, pressure_chamber, Kse, Kbt):
        return stiff_flop_odes_nb(y, pressure_chamber, Kse, Kbt,
                                  self.A_chambers, self.e3,
                                  self.rhoAg, self.r_chamber)
    
    def pressure_dependent_modulus(self, pressure_chamber):
        """Calculate pressure-dependent elastic modulus"""
        sum_pressure = np.sum(pressure_chamber) / 6
        
        # Polynomial coefficients for pressure-dependent modulus
        p1 = 8727
        p2 = -4.468e+04
        p3 = 9.663e+04
        p4 = -1.074e+05
        p5 = 9e+04
        
        E_y = (p1 * sum_pressure**4 + p2 * sum_pressure**3 + 
               p3 * sum_pressure**2 + p4 * sum_pressure + p5)
        G_p = E_y / 3
        
        return E_y, G_p
    
    def get_stiffness_matrices(self, E_y, G_p):
        """Calculate stiffness matrices"""
        Kse = np.diag([G_p * self.A_cross_section, 
                       G_p * self.A_cross_section, 
                       E_y * self.A_cross_section])
        Kbt = np.diag([E_y * self.I, E_y * self.I, G_p * self.J])
        return Kse, Kbt
    
    def ode4(self, func, t0, h, tfinal, y0):
        """4th order Runge-Kutta integration"""
        y = y0.copy()
        yout = [y.copy()]
        s = [t0]
        
        t = t0
        while t < tfinal - h:
            s1 = func(t, y)
            s2 = func(t + h/2, y + h * s1/2)
            s3 = func(t + h/2, y + h * s2/2)
            s4 = func(t + h, y + h * s3)
            y = y + h * (s1 + 2*s2 + 2*s3 + s4) / 6
            yout.append(y.copy())
            s.append(t + h)
            t += h
            
        return np.array(yout), np.array(s)
    
    def shooting_method_forward(self, G, pressure_chamber,
                                return_jac: bool = False,
                                n_elem: int | None = None):
        """Shooting method for forward kinematics
        Parameters
        ----------
        G : (6,) array
            Initial internal force/torque guess [n0, m0].
        pressure_chamber : (6,) array
            Chamber pressures (each chamber duplicated).
        return_jac : bool, optional
            If True also return ∂residual/∂G.  Default False.
        n_elem : int or None
            Number of backbone elements used in integration.
            If None, fall back to self.n_element (=10).
        """
        # ----- setup -------------------------------------------------
        if n_elem is None:
            n_elem = self.n_element                      # default 10
        h = self.L / n_elem

        n0, m0 = G[:3], G[3:6]

        # Elastic parameters
        E_y, G_p = self.pressure_dependent_modulus(pressure_chamber)
        Kse, Kbt  = self.get_stiffness_matrices(E_y, G_p)

        # Initial state vector
        y0 = np.concatenate([self.p0, self.R0.flatten(), n0, m0])

        # ODE
        def ode_func(s, y):
            return self.stiff_flop_odes(s, y, pressure_chamber, Kse, Kbt)

        y_out, _ = self.ode4(ode_func, 0.0, h, self.L, y0)

        # ----- residual at tip --------------------------------------
        RL = y_out[-1, 3:12].reshape(3, 3)
        nL = y_out[-1, 12:15]
        mL = y_out[-1, 15:18]

        force_err  = -nL + self.F
        torque_err = -mL + self.M

        for i in range(self.num_chambers):
            coeff = 1e5 * pressure_chamber[i] * self.A_chambers
            force_err  += RL @ (coeff * self.e3)
            torque_err += RL @ (coeff * self.hat(self.r_chamber[:, i]) @ self.e3)

        residual = np.concatenate([force_err, torque_err])

        # ----- simple finite-difference Jacobian --------------------
        if not return_jac:
            return residual

        eps   = 1e-6
        J     = np.zeros((6, 6))
        for k in range(6):
            G_pert          = G.copy()
            G_pert[k]      += eps
            res_plus        = self.shooting_method_forward(
                                   G_pert, pressure_chamber,
                                   return_jac=False, n_elem=n_elem)
            J[:, k] = (res_plus - residual) / eps

        return residual, J
    
    def shooting_method_inverse(self, G, desired_position):
        """Shooting method for inverse kinematics (9 variables: n0, m0, pressures)"""
        n0 = G[:3]
        m0 = G[3:6]
        # MATLAB: pressure_chamber = [G(7);G(7);G(8);G(8);G(9);G(9)];
        pressure_chamber = np.array([G[6], G[6], G[7], G[7], G[8], G[8]])
        
        # Calculate pressure-dependent modulus
        E_y, G_p = self.pressure_dependent_modulus(pressure_chamber)
        Kse, Kbt = self.get_stiffness_matrices(E_y, G_p)
        
        # Initial conditions
        y0 = np.concatenate([self.p0, self.R0.flatten(), n0, m0])
        
        # Create ODE function with fixed parameters
        def ode_func(s, y):
            return self.stiff_flop_odes(s, y, pressure_chamber, Kse, Kbt)
        
        # Integrate along the robot (inverse uses n_element = 6)
        y_out, s = self.ode4(ode_func, 0, self.L/6, self.L, y0)
        
        # Extract final values
        pL_shot = y_out[-1, :3]
        RL_shot = y_out[-1, 3:12].reshape(3, 3)
        nL_shot = y_out[-1, 12:15]
        mL_shot = y_out[-1, 15:18]
        
        # MATLAB: position_error = pL_shot - pL;
        position_error = pL_shot - desired_position
        
        # MATLAB: rotation_error = inv_hat(RL_shot'*RL-RL_shot*RL');
        # For now, we'll use identity for RL (bend = 0)
        RL = np.eye(3)  # MATLAB has bend = 0*pi/180
        rotation_error = self.inv_hat(RL_shot.T @ RL - RL_shot @ RL.T)
        
        # Calculate errors at tip
        force_error = -nL_shot + self.F
        torque_error = -mL_shot + self.M
        
        # Add chamber pressure contributions at tip
        for i in range(self.num_chambers):
            force_error = force_error + RL_shot @ (1e5 * pressure_chamber[i] * self.A_chambers * self.e3)
            torque_error = torque_error + RL_shot @ (1e5 * pressure_chamber[i] * self.A_chambers * 
                                                   self.hat(self.r_chamber[:, i]) @ self.e3)
        
        # Return error vector (MATLAB: E = [force_error; 1*torque_error; 2*position_error])
        return np.concatenate([force_error, 1.0 * torque_error, 2.0 * position_error])
    
    def forward_kinematics(self, act_pressure):
        """
        Forward kinematics with detailed timing information
        """
        import time
        
        # Overall timing
        total_start = time.time()
        
        # STEP 1: Setup - timing
        setup_start = time.time()
        pressure_chamber = np.array([
            act_pressure[0], act_pressure[0],
            act_pressure[1], act_pressure[1], 
            act_pressure[2], act_pressure[2]
        ])
        init_guess = np.zeros(6)
        setup_time = time.time() - setup_start
        
        # STEP 2: Optimization - timing
        optimize_start = time.time()
        
        # ---------------- objective & Jacobian -------------------------
        eval_count = [0]
        eval_times = []

        def residual_fun(G):
            t0 = time.time()
            res = self.shooting_method_forward(G, pressure_chamber,
                                               return_jac=False,  # residual only
                                               n_elem=3)          # cheap model
            eval_count[0] += 1
            eval_times.append(time.time() - t0)
            return res                         # <-- only the residual vector

        def jac_fun(G):
            # analytic (or finite-diff) Jacobian
            _, J = self.shooting_method_forward(G, pressure_chamber,
                                                return_jac=True,
                                                n_elem=3)
            return J

        result = least_squares(residual_fun, init_guess,
                               jac=jac_fun,          # <-- pass callable
                               method='trf',
                               ftol=1e-3, xtol=1e-3,
                               max_nfev=15)
        
        optimize_time = time.time() - optimize_start
        
        # STEP 3: Final shape computation - timing
        shape_start = time.time()
        
        # Get final robot shape
        n0 = result.x[:3]
        m0 = result.x[3:6]
        
        # Time the modulus calculation
        modulus_start = time.time()
        E_y, G_p = self.pressure_dependent_modulus(pressure_chamber)
        Kse, Kbt = self.get_stiffness_matrices(E_y, G_p)
        modulus_time = time.time() - modulus_start
        
        y0 = np.concatenate([self.p0, self.R0.flatten(), n0, m0])
        
        def ode_func(s, y):
            return self.stiff_flop_odes(s, y, pressure_chamber, Kse, Kbt)
        
        # Time the ODE solver
        ode_start = time.time()
        y_all, s = self.ode4(ode_func, 0, self.L/10, self.L, y0)
        ode_time = time.time() - ode_start
        
        y_tip = y_all[-1, :3]
        self.y_tip_all.append(y_tip.copy())
        
        shape_time = time.time() - shape_start
        total_time = time.time() - total_start
        
        # Record computation time
        self.compu_time.append(total_time)
        
        # Print timing information
        # print("\n--- FORWARD KINEMATICS TIMING ANALYSIS ---")
        # print(f"Total execution time: {total_time:.6f} seconds")
        # print(f"  1. Setup time:       {setup_time:.6f} seconds ({setup_time/total_time*100:.1f}%)")
        # print(f"  2. Optimization:     {optimize_time:.6f} seconds ({optimize_time/total_time*100:.1f}%)")
        # print(f"     - Function evaluations: {eval_count[0]}")
        # if eval_count[0] > 0:
        #     print(f"     - Average per evaluation: {sum(eval_times)/len(eval_times):.6f} seconds")
        #     print(f"     - First evaluation: {eval_times[0]:.6f} seconds")
        #     print(f"     - Last evaluation:  {eval_times[-1]:.6f} seconds")
        # print(f"  3. Final shape calc: {shape_time:.6f} seconds ({shape_time/total_time*100:.1f}%)")
        # print(f"     - Modulus calculation: {modulus_time:.6f} seconds")
        # print(f"     - ODE integration:     {ode_time:.6f} seconds")
        # print(f"  4. Computation in shooting_method_forward:")
        
        # Run a single evaluation of shooting_method to profile it
        #profile_start = time.time()
        _ = self.shooting_method_forward(result.x, pressure_chamber)
        #profile_time = time.time() - profile_start
        #print(f"     - Single evaluation:    {profile_time:.6f} seconds")
        
        # Add a call to profile just the ODE part of shooting_method
        ode_profile_start = time.time()
        def temp_ode_func(s, y):
            return self.stiff_flop_odes(s, y, pressure_chamber, Kse, Kbt)
        
        y0_temp = np.concatenate([self.p0, self.R0.flatten(), result.x[:3], result.x[3:6]])
        _, _ = self.ode4(temp_ode_func, 0, self.L/10, self.L, y0_temp)
        #ode_profile_time = time.time() - ode_profile_start
        #print(f"     - Just ODE part:        {ode_profile_time:.6f} seconds")
        
        return result.x, y_tip, y_all
    
    def inverse_kinematics(self, desired_position):
        """
        Inverse kinematics: given desired tip position, find required pressures
        Follows MATLAB Static_Inverse_one_segment.m exactly
        
        Args:
            desired_position: desired [x, y, z] tip position
            
        Returns:
            pressures: optimized pressure values [p1, p2, p3]
            y_tip: achieved tip position  
            y_all: complete robot shape
        """
        # MATLAB: init_guess = last_guess; %Initial guess
        init_guess = self.last_guess.copy()
        
        start_time = time.time()
        
        # MATLAB: [x,res] = lsqnonlin(@STIFF_FLOP_Shooting,init_guess)
        def objective(G):
            return self.shooting_method_inverse(G, desired_position)
        
        # Solve with 9 variables: [n0(3), m0(3), p1, p2, p3]
        result = least_squares(objective, init_guess, 
                          bounds=([-np.inf]*6 + [0]*3, [np.inf]*6 + [2.5]*3),
                          method='trf', ftol=1e-4, xtol=1e-4, max_nfev=20)
        
        computation_time = time.time() - start_time
        self.compu_time.append(computation_time)
        
        # MATLAB: last_guess = x;
        self.last_guess = result.x.copy()
        
        # Extract pressure values from solution
        # MATLAB: pressure_chamber = [G(7);G(7);G(8);G(8);G(9);G(9)];
        pressures = result.x[6:9]  # [p1, p2, p3]
        pressure_chamber = np.array([pressures[0], pressures[0], 
                                    pressures[1], pressures[1], 
                                    pressures[2], pressures[2]])
        
        # Get final robot shape using the optimized solution
        n0 = result.x[:3]
        m0 = result.x[3:6]
        E_y, G_p = self.pressure_dependent_modulus(pressure_chamber)
        Kse, Kbt = self.get_stiffness_matrices(E_y, G_p)
        
        y0 = np.concatenate([self.p0, self.R0.flatten(), n0, m0])
        
        def ode_func(s, y):
            return self.stiff_flop_odes(s, y, pressure_chamber, Kse, Kbt)
        
        # MATLAB inverse uses n_element = 6
        y_all, s = self.ode4(ode_func, 0, self.L/6, self.L, y0)
        y_tip = y_all[-1, :3]
        
        # MATLAB: y_tip_all = [y_tip_all;y_tip];
        self.y_tip_all.append(y_tip.copy())
        
        return pressures, y_tip, y_all
    
    def plot_tube(self, ax, y_all, radius=None, color='orange', alpha=0.3):
        """
        Simple function to plot a cylindrical tube around the robot backbone with orientation
        
        Args:
            ax: matplotlib 3D axis
            y_all: Nx18 array containing position and orientation data
            radius: tube radius (default: robot radius)
            color: tube color
            alpha: transparency
        """
        if radius is None:
            radius = self.rad_robot
        
        n_points = len(y_all)
        n_sides = 12  # number of sides for cylinder
        
        # Create parameter arrays
        theta = np.linspace(0, 2*np.pi, n_sides)
        
        # Initialize surface arrays
        X = np.zeros((n_points, n_sides))
        Y = np.zeros((n_points, n_sides))
        Z = np.zeros((n_points, n_sides))
        
        # Create cylindrical surface with proper orientation
        for i in range(n_points):
            # Extract position and rotation matrix
            position = y_all[i, :3]
            R = y_all[i, 3:12].reshape(3, 3)
            
            for j, th in enumerate(theta):
                # Create circular cross-section in local coordinates
                local_point = np.array([
                    radius * np.cos(th),
                    radius * np.sin(th),
                    0
                ])
                
                # Transform to global coordinates using rotation matrix
                global_point = position + R @ local_point
                
                X[i, j] = global_point[0]
                Y[i, j] = global_point[1]
                Z[i, j] = global_point[2]
        
        # Plot the tube surface
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, shade=True)

    def visualize_robot(self, y_all, title="Soft Robot Shape", desired_position=None, show_tube=False):
        """Visualize the robot shape in 3D"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot robot backbone
        ax.plot(y_all[:, 0], y_all[:, 1], y_all[:, 2], 'b-', linewidth=3, label='Robot backbone')
        
        # Plot tube if requested
        if show_tube:
            self.plot_tube(ax, y_all)
        
        # Plot ONLY current tip position (end of y_all)
        current_tip = y_all[-1, :3]  # Get the last position from y_all
        ax.scatter(current_tip[0], current_tip[1], current_tip[2], 
                   c='red', s=50, label='Current tip position')
        
        # Plot desired position if provided (ONLY ADDITION)
        if desired_position is not None:
            ax.scatter(desired_position[0], desired_position[1], desired_position[2], 
                      c='blue', s=100, marker='*', label='Desired position', 
                      edgecolors='black', linewidths=2)
        
        # Ground plane
        L = self.L
        ground_x = [-L*0.5, L*0.5, L*0.5, -L*0.5, -L*0.5]
        ground_y = [-L*0.5, -L*0.5, L*0.5, L*0.5, -L*0.5]
        ground_z = [0, 0, 0, 0, 0]
        ax.plot(ground_x, ground_y, ground_z, 'k-', alpha=0.5)
        
        # Set axis properties
        ax.set_xlim([-L*2.5, L*2.5])
        ax.set_ylim([-L*2.5, L*2.5])
        ax.set_zlim([0, 2.5*L])
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(title)
        ax.legend()
        
        # Invert Z axis to match MATLAB convention
        ax.invert_zaxis()
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def inv_hat(self, M: np.ndarray) -> np.ndarray:
        """Inverse of the hat (skew-symmetric) operator.
           [  0  -z   y ]          [x]
           [  z   0  -x ]  →  v =  [y]
           [ -y   x   0 ]          [z]
        """
        return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)
