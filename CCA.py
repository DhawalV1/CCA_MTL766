import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig # For generalized eigenvalue problem

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

# --- Main execution block (combining previous steps and adding new CCA logic) ---

if __name__ == "__main__":
    np.random.seed(42) # for reproducibility

    # --- Global parameters (assuming they are consistent across all N series) ---
    N_regions = 8 # Number of independent realizations (countries/regions)
    num_time_points_per_series = 19 # J* or J** in the paper (1993-2011 inclusive = 19 points)
    T_min_global = 1993
    T_max_global = 2011
    T_period_global = T_max_global - T_min_global # 18 years

    basis_type = 'fourier'
    max_K_components = 5 

    print(f"Starting Functional CCA for N={N_regions} regions using {basis_type} basis.")

    # --- Load Real GDP and FDI Data ---


    # Load the merged dataset
    data = pd.read_csv("merged_gdp_fdi_data.csv")

    # Define region codes (same order as N_regions)
    region_labels = ["EUU", "LMY", "WLD", "BGR", "POL", "RUS", "UGA", "USA"]

    # Define the range of years of interest
    years = np.arange(1993, 2012)  # 1993–2011 inclusive

    # --- Prepare lists of discrete observations ---
    all_time_points_y = []  # For GDP
    all_discrete_data_y = []
    all_time_points_x = []  # For FDI
    all_discrete_data_x = []

    for code in region_labels:
        # Filter data for this country
        row = data[data["Country Code"] == code]

        if row.empty:
            print(f"Warning: No data for {code}")
            continue

        # Extract GDP and FDI columns for selected years
        gdp_cols = [f"GDP_{year}" for year in years]
        fdi_cols = [f"FDI_{year}" for year in years]

        gdp_values = row[gdp_cols].values.flatten()
        fdi_values = row[fdi_cols].values.flatten()

        # Handle missing data by interpolation and filling
        gdp_values = pd.Series(gdp_values).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill").values
        fdi_values = pd.Series(fdi_values).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill").values

        # Append to the lists
        all_time_points_y.append(years)
        all_discrete_data_y.append(gdp_values)
        all_time_points_x.append(years)
        all_discrete_data_x.append(fdi_values)

    # Quick sanity check
    print(f"Loaded data for {len(all_discrete_data_y)} regions.")
    print(f"Example GDP data for {region_labels[0]}:", all_discrete_data_y[0][:5])
    print(f"Example FDI data for {region_labels[0]}:", all_discrete_data_x[0][:5])
    




    # --- Determine Optimal K for Y and X ---
    optimal_K_y, _, _, _, num_coeffs_y_opt = select_optimal_K(
        all_time_points_y[0], all_discrete_data_y[0], max_K_components,
        T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
    )
    print(f"\nOptimal K for Y (from first region data): {optimal_K_y} "
          f"({num_coeffs_y_opt} coefficients)")

    optimal_K_x, _, _, _, num_coeffs_x_opt = select_optimal_K(
        all_time_points_x[0], all_discrete_data_x[0], max_K_components,
        T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
    )
    print(f"Optimal K for X (from first region data): {optimal_K_x} "
          f"({num_coeffs_x_opt} coefficients)")

    # --- Transform each realization to coefficients and construct A and B matrices ---
    A = np.zeros((N_regions, num_coeffs_y_opt))
    B = np.zeros((N_regions, num_coeffs_x_opt))

    print("\nFitting basis functions and collecting coefficients for all regions...")
    for i in range(N_regions):
        coeffs_y_i, _, _ = fit_basis_functions_fixed_K(
            all_time_points_y[i], all_discrete_data_y[i], optimal_K_y,
            T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
        )
        A[i, :] = coeffs_y_i

        coeffs_x_i, _, _ = fit_basis_functions_fixed_K(
            all_time_points_x[i], all_discrete_data_x[i], optimal_K_x,
            T_period_global, basis_type=basis_type, T_min=T_min_global, T_max=T_max_global
        )
        B[i, :] = coeffs_x_i
    
    # --- Estimate Covariance Matrices Sigma_11, Sigma_22, Sigma_12 ---
    Sigma_11_hat = (1 / N_regions) * (A.T @ A)
    Sigma_22_hat = (1 / N_regions) * (B.T @ B)
    Sigma_12_hat = (1 / N_regions) * (A.T @ B)
    Sigma_21_hat = Sigma_12_hat.T

    print("\nEstimated Sigma_11_hat (covariance matrix of alpha coefficients):")
    print(Sigma_11_hat)
    print("\nEstimated Sigma_22_hat (covariance matrix of beta coefficients):")
    print(Sigma_22_hat)
    print("\nEstimated Sigma_12_hat (cross-covariance matrix of alpha and beta coefficients):")
    print(Sigma_12_hat)

    # --- NEW STEP: Perform Canonical Correlation Analysis ---
    print("\n--- Performing Canonical Correlation Analysis ---")

    # Check for invertibility
    if np.linalg.det(Sigma_11_hat) == 0:
        print("Error: Sigma_11_hat is singular. Cannot invert.")
        exit()
    if np.linalg.det(Sigma_22_hat) == 0:
        print("Error: Sigma_22_hat is singular. Cannot invert.")
        exit()

    # 1. Compute C and D matrices
    try:
        inv_Sigma_11_hat = np.linalg.inv(Sigma_11_hat)
        inv_Sigma_22_hat = np.linalg.inv(Sigma_22_hat)
    except np.linalg.LinAlgError:
        print("Error: Covariance matrices are not invertible. This can happen if N is small "
              "relative to the number of coefficients, or if there's perfect multicollinearity.")
        # Fallback to pseudo-inverse if strict inverse fails (though generally not ideal for CCA)
        inv_Sigma_11_hat = np.linalg.pinv(Sigma_11_hat)
        inv_Sigma_22_hat = np.linalg.pinv(Sigma_22_hat)
        print("Attempting with pseudo-inverse. Results might be less reliable.")


    C_matrix = inv_Sigma_11_hat @ Sigma_12_hat
    D_matrix = inv_Sigma_22_hat @ Sigma_21_hat

    # 2. Solve the Eigenvalue Problems
    # For u_k (weight vectors for alpha): eigenvalues and eigenvectors of CD
    # For v_k (weight vectors for beta): eigenvalues and eigenvectors of DC

    # Eigenvalues and eigenvectors for CD (gives u_k)
    eigenvalues_CD, eigenvectors_CD = eig(C_matrix @ D_matrix)
    
    # Eigenvalues and eigenvectors for DC (gives v_k)
    eigenvalues_DC, eigenvectors_DC = eig(D_matrix @ C_matrix)

    # Sort the canonical correlations (eigenvalues) in descending order
    # The eigenvalues should be real and non-negative (squared correlations)
    # We take the real part and ensure they are non-negative, accounting for potential numerical precision issues
    rho_squared = np.real(eigenvalues_CD)
    rho_squared[rho_squared < 0] = 0 # Ensure non-negativity
    
    # Get sorted indices
    sort_indices = np.argsort(rho_squared)[::-1] # Descending order

    canonical_correlations = np.sqrt(rho_squared[sort_indices])
    canonical_weights_alpha = eigenvectors_CD[:, sort_indices]
    canonical_weights_beta = eigenvectors_DC[:, sort_indices]

    # Normalize weight vectors to ensure Var(<u,Y>)=1, Var(<v,X>)=1
    # As per paper's equation (4) and "subject to the restriction u'Sigma_11_u=1, v'Sigma_22_v=1"
    # This involves scaling the eigenvectors
    print("\nNormalizing canonical weight vectors...")
    for k in range(len(canonical_correlations)):
        uk = canonical_weights_alpha[:, k]
        vk = canonical_weights_beta[:, k]

        # u'Sigma_11_u = 1
        norm_factor_u = np.sqrt(uk.T @ Sigma_11_hat @ uk)
        if norm_factor_u > 1e-9: # Avoid division by zero
            canonical_weights_alpha[:, k] = uk / norm_factor_u
        
        # v'Sigma_22_v = 1
        norm_factor_v = np.sqrt(vk.T @ Sigma_22_hat @ vk)
        if norm_factor_v > 1e-9:
            canonical_weights_beta[:, k] = vk / norm_factor_v

    print("\nCanonical Correlations (rho_k):")
    print(canonical_correlations)
    
    print("\nFirst Canonical Weight Vector for alpha (u_1):")
    print(canonical_weights_alpha[:, 0])
    
    print("\nFirst Canonical Weight Vector for beta (v_1):")
    print(canonical_weights_beta[:, 0])

    # Ensure the minimum of (K1+1, K2+1) corresponds to the number of canonical pairs
    num_canonical_pairs = min(num_coeffs_y_opt, num_coeffs_x_opt)
    print(f"\nNumber of meaningful canonical pairs: {num_canonical_pairs}")

    # --- 3. Reconstruct Functional Canonical Weight Functions (u_k(t), v_k(t)) ---
    # These are u_k(t) = u_k' * phi(t) and v_k(t) = v_k' * psi(t)
    print("\n--- Reconstructing Functional Canonical Weight Functions ---")

    time_grid_dense = np.linspace(T_min_global, T_max_global, 200)

    # Get the basis function matrices for evaluation
    Phi_basis_y, _ = build_phi_matrix(time_grid_dense, optimal_K_y, T_period_global, basis_type, T_min_global, T_max_global)
    Psi_basis_x, _ = build_phi_matrix(time_grid_dense, optimal_K_x, T_period_global, basis_type, T_min_global, T_max_global)

    # Functional canonical weight functions (u_k(t) and v_k(t))
    functional_uk = [] # List of functions
    functional_vk = [] # List of functions

    for k in range(num_canonical_pairs):
        uk_vec = canonical_weights_alpha[:, k]
        vk_vec = canonical_weights_beta[:, k]

        # functional_uk_t = Phi_basis_y @ uk_vec
        # functional_vk_t = Psi_basis_x @ vk_vec
        
        # Store as callable functions for convenience
        functional_uk.append(get_functional_representation_func(
            uk_vec, optimal_K_y, T_period_global, basis_type, T_min_global, T_max_global
        ))
        functional_vk.append(get_functional_representation_func(
            vk_vec, optimal_K_x, T_period_global, basis_type, T_min_global, T_max_global
        ))

    # --- Plotting the First Pair of Weight Functions (u_1(t), v_1(t)) ---
    # This directly relates to Figure 3 in your paper.
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid_dense, functional_uk[0](time_grid_dense), label='u_1(t) (Weight Function for Y)', color='blue')
    plt.plot(time_grid_dense, functional_vk[0](time_grid_dense), label='v_1(t) (Weight Function for X)', color='red', linestyle='--')
    plt.title(f'First Pair of Functional Canonical Weight Functions (rho_1 = {canonical_correlations[0]:.2f})')
    plt.xlabel('Time')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("functional_cca_plot.png")
    print("Plotting the first pair of functional canonical weight functions.")

    canonical_variables_U = np.zeros((N_regions, num_canonical_pairs))
    canonical_variables_V = np.zeros((N_regions, num_canonical_pairs))

    for i in range(N_regions):
        alpha_i = A[i, :] # Coefficients for Y_i(t)
        beta_i = B[i, :]  # Coefficients for X_i(t)

        for k in range(num_canonical_pairs):
            uk_vec = canonical_weights_alpha[:, k]
            vk_vec = canonical_weights_beta[:, k]

            canonical_variables_U[i, k] = uk_vec.T @ alpha_i
            canonical_variables_V[i, k] = vk_vec.T @ beta_i

    print("\nFirst Canonical Variables U_1 for each region:")
    print(canonical_variables_U[:, 0])

    print("\nFirst Canonical Variables V_1 for each region:")
    print(canonical_variables_V[:, 0])

    # --- Plotting the Projection of Regions on the (U_1, V_1) Plane ---
    # This directly relates to Figure 4 in your paper.
    plt.figure(figsize=(8, 8))
    for i in range(N_regions):
        # Assign a label for each region (e.g., EUU, LMY, WLD, BGR, POL, RUS, UGA, USA as in paper)
        region_labels = ["EUU", "LMY", "WLD", "BGR", "POL", "RUS", "UGA", "USA"]
        plt.scatter(canonical_variables_U[i, 0], canonical_variables_V[i, 0], label=region_labels[i], s=100)
        plt.text(canonical_variables_U[i, 0] + 0.02, canonical_variables_V[i, 0] + 0.02, region_labels[i])
        
    plt.title(f'Projection of Regions on the (U_1, V_1) Plane (rho_1 = {canonical_correlations[0]:.2f})')
    plt.xlabel('U_1 (First Canonical Variable for Y)')
    plt.ylabel('V_1 (First Canonical Variable for X)')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("cca_projection_plot.png")
    print("Plotting the projection of regions on the (U_1, V_1) plane.")