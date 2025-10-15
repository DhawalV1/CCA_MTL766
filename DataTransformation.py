import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Re-using and slightly adapting functions from previous step ---

def fourier_basis_function(t, k, T, is_cosine=True):
    """
    Fourier basis function:
    phi_0(t) = 1 (for k=0, if is_cosine is True)
    phi_{2k-1}(t) = sqrt(2/T) * sin(2*pi*k*t / T)
    phi_{2k}(t) = sqrt(2/T) * cos(2*pi*k*t / T)
    """
    if k == 0:
        return 1.0
    if is_cosine:
        return np.sqrt(2 / T) * np.cos(2 * np.pi * k * t / T)
    else:
        return np.sqrt(2 / T) * np.sin(2 * np.pi * k * t / T)

def legendre_polynomial(x, k):
    """
    Legendre polynomial P_k(x) over [-1, 1].
    P_0(x) = 1
    P_1(x) = x
    P_{k+1}(x) = ( (2k+1)x P_k(x) - k P_{k-1}(x) ) / (k+1)
    """
    if k == 0:
        return 1.0
    elif k == 1:
        return x
    else:
        P_k_minus_1 = legendre_polynomial(x, k - 1)
        P_k_minus_2 = legendre_polynomial(x, k - 2)
        return ((2 * (k - 1) + 1) * x * P_k_minus_1 - (k - 1) * P_k_minus_2) / k

def legendre_basis_function(t, k, T_min, T_max):
    """
    Orthonormalized Legendre basis function over [T_min, T_max].
    First, transform t from [T_min, T_max] to x in [-1, 1].
    Then apply the orthonormalization factor.
    """
    x = (2 * t - (T_max + T_min)) / (T_max - T_min)
    return np.sqrt((2 * k + 1) / 2) * legendre_polynomial(x, k)

def build_phi_matrix(time_points, K, T_period, basis_type, T_min, T_max):
    """
    Helper function to build the Phi matrix given parameters.
    Handles both Fourier and Legendre basis types.
    """
    if basis_type == 'fourier':
        num_coeffs = 1 + 2 * K
        Phi = np.zeros((len(time_points), num_coeffs))
        for j, t_j in enumerate(time_points):
            Phi[j, 0] = 1.0
            for k_val in range(1, K + 1):
                Phi[j, 2 * k_val - 1] = fourier_basis_function(t_j, k_val, T_period, is_cosine=False)
                Phi[j, 2 * k_val] = fourier_basis_function(t_j, k_val, T_period, is_cosine=True)
    elif basis_type == 'legendre':
        num_coeffs = K + 1
        Phi = np.zeros((len(time_points), num_coeffs))
        for j, t_j in enumerate(time_points):
            for k_val in range(K + 1):
                Phi[j, k_val] = legendre_basis_function(t_j, k_val, T_min, T_max)
    else:
        raise ValueError("basis_type must be 'fourier' or 'legendre'")
    return Phi, num_coeffs

def get_functional_representation_func(coeffs, K, T_period, basis_type, T_min, T_max):
    """
    Returns a callable function for the continuous functional representation.
    """
    if basis_type == 'fourier':
        def functional_representation(t_eval):
            Phi_eval, _ = build_phi_matrix(np.atleast_1d(t_eval), K, T_period, basis_type, T_min, T_max)
            return Phi_eval @ coeffs
    elif basis_type == 'legendre':
        def functional_representation(t_eval):
            Phi_eval, _ = build_phi_matrix(np.atleast_1d(t_eval), K, T_period, basis_type, T_min, T_max)
            return Phi_eval @ coeffs
    else:
        raise ValueError("basis_type must be 'fourier' or 'legendre'")
    return functional_representation

def fit_basis_functions_fixed_K(time_points, discrete_data, K, T_period, basis_type='fourier', T_min=None, T_max=None):
    """
    Fits a set of K+1 orthonormal basis functions to discrete data using least squares
    with a *fixed* K, assuming K has already been optimally selected.

    Args:
        time_points (np.array): Array of time points (t_ij).
        discrete_data (np.array): Array of discrete observations (y_ij or x_ij).
        K (int): Number of basis functions (highest index for Legendre, highest frequency for Fourier).
        T_period (float): The total period for Fourier basis functions.
        basis_type (str): 'fourier' or 'legendre'.
        T_min, T_max (float): Min/Max time points for Legendre basis (inferred if None).

    Returns:
        np.array: Array of estimated coefficients (alpha_k or beta_l).
        function: A callable function representing the fitted continuous function.
        int: Number of coefficients used.
    """
    if T_min is None and basis_type == 'legendre':
        T_min = np.min(time_points)
    if T_max is None and basis_type == 'legendre':
        T_max = np.max(time_points)

    Phi, num_coeffs = build_phi_matrix(time_points, K, T_period, basis_type, T_min, T_max)
        
    try:
        coefficients = np.linalg.solve(Phi.T @ Phi, Phi.T @ discrete_data)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(Phi) @ discrete_data

    functional_representation_func = get_functional_representation_func(
        coefficients, K, T_period, basis_type, T_min, T_max
    )
    return coefficients, functional_representation_func, num_coeffs

def calculate_bic(discrete_data, fitted_data, num_coefficients, num_observations):
    """
    Calculates the Bayesian Information Criterion (BIC) for a given fit.
    Using standard BIC: n * log(MSE) + k * log(n)
    """
    residuals = discrete_data - fitted_data
    sum_sq_residuals = np.sum(residuals**2)
    
    if num_observations <= num_coefficients:
        return np.inf
    
    mse = sum_sq_residuals / num_observations
    if mse <= 0:
        return np.inf

    bic_val = num_observations * np.log(mse) + num_coefficients * np.log(num_observations)
    return bic_val

def select_optimal_K(time_points, discrete_data, max_K, T_period, basis_type='fourier', T_min=None, T_max=None):
    """
    Selects the optimal number of basis functions K using BIC.
    """
    best_bic = np.inf
    optimal_K = 0
    optimal_coeffs = None
    optimal_func = None
    optimal_num_coeffs = 0

    if T_min is None and basis_type == 'legendre':
        T_min = np.min(time_points)
    if T_max is None and basis_type == 'legendre':
        T_max = np.max(time_points)

    for K_current in range(max_K + 1):
        coeffs, func, num_coeffs = fit_basis_functions_fixed_K(
            time_points, discrete_data, K_current, T_period, basis_type, T_min, T_max
        )
        
        if num_coeffs > len(time_points):
            break

        fitted_values = func(time_points)
        current_bic = calculate_bic(discrete_data, fitted_values, num_coeffs, len(time_points))

        if current_bic < best_bic:
            best_bic = current_bic
            optimal_K = K_current
            optimal_coeffs = coeffs
            optimal_func = func
            optimal_num_coeffs = num_coeffs
            
    return optimal_K, optimal_coeffs, optimal_func, best_bic, optimal_num_coeffs

# --- New Code for CCA Setup ---

if __name__ == "__main__":
    np.random.seed(42) # for reproducibility

    # --- Global parameters (assuming they are consistent across all N series) ---
    N_regions = 8 # Number of independent realizations (countries/regions)
    num_time_points_per_series = 19 # J* or J** in the paper (1993-2011 inclusive = 19 points)
    T_min_global = 1993
    T_max_global = 2011
    T_period_global = T_max_global - T_min_global # 18 years

    # Define the basis type and max K for optimal selection
    # Using Fourier for demonstration, as per common practice in time series
    basis_type = 'fourier'
    max_K_components = 5 # Maximum number of sine/cosine pairs to consider for optimal K selection

    print(f"Starting CCA setup for N={N_regions} regions using {basis_type} basis...")

    # --- 1. Simulate N realizations for Y(t) and X(t) ---
    # In a real scenario, you would load your N time series from files/database.
    # Here, we simulate them to get varied data for each 'region'.

    all_time_points_y = []
    all_discrete_data_y = []
    all_time_points_x = []
    all_discrete_data_x = []

    for i in range(N_regions):
        # Simulate time points (assuming same for all for simplicity, but could be unique)
        time_points = np.sort(np.linspace(T_min_global, T_max_global, num_time_points_per_series) + np.random.normal(0, 0.1, num_time_points_per_series))
        
        # Simulate Y(t) - GDP Growth
        # Adding some region-specific variation to the underlying 'true' function
        true_y_i = (2 + 0.5*i) * np.sin(2 * np.pi * (time_points - T_min_global) / T_period_global) + \
                   (1.5 - 0.1*i) * np.cos(4 * np.pi * (time_points - T_min_global) / T_period_global) + \
                   (3 + 0.2*i)
        discrete_data_y_i = true_y_i + np.random.normal(0, 0.5, num_time_points_per_series) # Add noise

        # Simulate X(t) - Rate of growth in direct foreign investment
        true_x_i = (3 - 0.3*i) * np.cos(2 * np.pi * (time_points - T_min_global) / T_period_global) - \
                   (1 + 0.2*i) * np.sin(6 * np.pi * (time_points - T_min_global) / T_period_global) + \
                   (5 - 0.1*i)
        discrete_data_x_i = true_x_i + np.random.normal(0, 0.7, num_time_points_per_series) # Add noise

        all_time_points_y.append(time_points)
        all_discrete_data_y.append(discrete_data_y_i)
        all_time_points_x.append(time_points)
        all_discrete_data_x.append(discrete_data_x_i)

    # --- 2. Determine Optimal K for Y and X (averaged across regions or chosen from a representative one) ---
    # For robustness, you might select an optimal K for each series and then
    # take a modal value, as suggested by the paper (page 3, "then from the values of K1
    # corresponding to all functions a modal value is selected, as the common value for all yi(t)").
    # For simplicity, we'll determine optimal K for the *first* region's data.

    # Optimal K for Y
    optimal_K_y, _, _, _, num_coeffs_y_opt = select_optimal_K(
        all_time_points_y[0], all_discrete_data_y[0], max_K_components,
        T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
    )
    print(f"\nOptimal K selected for Y (from first region data): {optimal_K_y} "
          f"({num_coeffs_y_opt} coefficients)")

    # Optimal K for X
    optimal_K_x, _, _, _, num_coeffs_x_opt = select_optimal_K(
        all_time_points_x[0], all_discrete_data_x[0], max_K_components,
        T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
    )
    print(f"Optimal K selected for X (from first region data): {optimal_K_x} "
          f"({num_coeffs_x_opt} coefficients)")

    # --- 3. Transform each realization to coefficients and construct A and B matrices ---
    # A will hold the alpha_i vectors (N rows, num_coeffs_y_opt columns)
    # B will hold the beta_i vectors (N rows, num_coeffs_x_opt columns)
    
    A = np.zeros((N_regions, num_coeffs_y_opt))
    B = np.zeros((N_regions, num_coeffs_x_opt))

    print("\nFitting basis functions and collecting coefficients for all regions...")
    for i in range(N_regions):
        # Fit Y_i(t) to get alpha_i
        coeffs_y_i, _, _ = fit_basis_functions_fixed_K(
            all_time_points_y[i], all_discrete_data_y[i], optimal_K_y,
            T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
        )
        A[i, :] = coeffs_y_i

        # Fit X_i(t) to get beta_i
        coeffs_x_i, _, _ = fit_basis_functions_fixed_K(
            all_time_points_x[i], all_discrete_data_x[i], optimal_K_x,
            T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
        )
        B[i, :] = coeffs_x_i
    
    print("\nMatrix A (alpha coefficients for Y processes, first 3 rows):")
    print(A[:3, :])
    print(f"Shape of A: {A.shape}")

    print("\nMatrix B (beta coefficients for X processes, first 3 rows):")
    print(B[:3, :])
    print(f"Shape of B: {B.shape}")

    # --- 4. Estimate Covariance Matrices Sigma_11, Sigma_22, Sigma_12 ---
    # From Section 4 of the paper:
    # Sigma_11_hat = (1/N) * A' * A
    # Sigma_22_hat = (1/N) * B' * B
    # Sigma_12_hat = (1/N) * A' * B

    # Ensure N > K1+1 and N > K2+1 for positive definiteness (as per paper)
    if N_regions <= num_coeffs_y_opt:
        print(f"\nWarning: N ({N_regions}) is not greater than K1+1 ({num_coeffs_y_opt}). "
              "Sigma_11_hat might not be positive definite.")
    if N_regions <= num_coeffs_x_opt:
        print(f"\nWarning: N ({N_regions}) is not greater than K2+1 ({num_coeffs_x_opt}). "
              "Sigma_22_hat might not be positive definite.")

    Sigma_11_hat = (1 / N_regions) * (A.T @ A)
    Sigma_22_hat = (1 / N_regions) * (B.T @ B)
    Sigma_12_hat = (1 / N_regions) * (A.T @ B)
    Sigma_21_hat = Sigma_12_hat.T # Transpose for Sigma_21

    print("\nEstimated Sigma_11_hat (covariance matrix of alpha coefficients):")
    print(Sigma_11_hat)
    print(f"Shape of Sigma_11_hat: {Sigma_11_hat.shape}")

    print("\nEstimated Sigma_22_hat (covariance matrix of beta coefficients):")
    print(Sigma_22_hat)
    print(f"Shape of Sigma_22_hat: {Sigma_22_hat.shape}")

    print("\nEstimated Sigma_12_hat (cross-covariance matrix of alpha and beta coefficients):")
    print(Sigma_12_hat)
    print(f"Shape of Sigma_12_hat: {Sigma_12_hat.shape}")

    # Now you have the estimated covariance matrices. The next step is to
    # perform the actual eigenvalue problem to find canonical correlations and weights.
    # This involves calculating C = Sigma_11_hat^-1 @ Sigma_12_hat
    # and D = Sigma_22_hat^-1 @ Sigma_21_hat, then solving for eigenvalues/eigenvectors of CD and DC.

    print("\nNext, you would perform the eigenvalue decomposition on CD and DC "
          "to find the canonical correlations and weight vectors.")

    # --- Optional: Visualize one of the functional data transformations ---
    time_grid = np.linspace(T_min_global, T_max_global, 200)

    # Transform the first region's data again to get the function for plotting
    _, func_y_region1, _ = fit_basis_functions_fixed_K(
        all_time_points_y[0], all_discrete_data_y[0], optimal_K_y,
        T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
    )
    _, func_x_region1, _ = fit_basis_functions_fixed_K(
        all_time_points_x[0], all_discrete_data_x[0], optimal_K_x,
        T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
    )

    plt.figure(figsize=(10, 5))
    plt.scatter(all_time_points_y[0], all_discrete_data_y[0], s=20, label='Region 1 Discrete Y(t)', alpha=0.7)
    plt.plot(time_grid, func_y_region1(time_grid), color='red', label=f'Region 1 Functional Y(t) (K={optimal_K_y})')
    plt.scatter(all_time_points_x[0], all_discrete_data_x[0], s=20, label='Region 1 Discrete X(t)', alpha=0.7, color='purple')
    plt.plot(time_grid, func_x_region1(time_grid), color='orange', label=f'Region 1 Functional X(t) (K={optimal_K_x})')
    plt.title(f'Functional Data Transformation for one Region ({basis_type.capitalize()} Basis)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("functional_data_transformation.png")