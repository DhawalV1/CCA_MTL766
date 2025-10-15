import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.integrate import quad

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
    # Transform t from [T_min, T_max] to x in [-1, 1]
    # x = (2 * (t - T_min) / (T_max - T_min)) - 1
    x = (2 * t - (T_max + T_min)) / (T_max - T_min)

    # Orthonormalization factor for Legendre polynomials over [-1, 1]
    # For L2([-1, 1]), the orthonormal Legendre polynomials are sqrt((2k+1)/2) * P_k(x)
    return np.sqrt((2 * k + 1) / 2) * legendre_polynomial(x, k)

def fit_basis_functions(time_points, discrete_data, K, T_period, basis_type='fourier'):
    """
    Fits a set of K+1 orthonormal basis functions to discrete data using least squares.

    Args:
        time_points (np.array): Array of time points (t_ij).
        discrete_data (np.array): Array of discrete observations (y_ij or x_ij).
        K (int): Number of basis functions (K1 or K2) to use, excluding the constant term.
                 So, K+1 coefficients will be estimated.
        T_period (float): The period T for Fourier basis functions. For Legendre,
                          it will be used to define the interval [0, T_period].
        basis_type (str): 'fourier' or 'legendre'.

    Returns:
        np.array: Array of estimated coefficients (alpha_k or beta_l).
        function: A callable function representing the fitted continuous function.
    """
    if basis_type == 'fourier':
        # Create a matrix Phi where Phi[j, k] = phi_k(t_j)
        Phi = np.zeros((len(time_points), K + 1))
        for j, t_j in enumerate(time_points):
            Phi[j, 0] = fourier_basis_function(t_j, 0, T_period, is_cosine=True) # Constant term
            for k_val in range(1, K + 1):
                # Map k_val to the 2k-1 (sin) and 2k (cos) indices as per the paper
                # For simplicity, we'll use a direct index here:
                # Phi[j, 2*k_val - 1] = fourier_basis_function(t_j, k_val, T_period, is_cosine=False)
                # Phi[j, 2*k_val] = fourier_basis_function(t_j, k_val, T_period, is_cosine=True)
                # If K is the number of functions, we create K+1 coefficients.
                # Assuming the basis functions are indexed 0, 1, ..., K
                # We can choose to use pairs of sin/cos for k > 0 or just sequentially.
                # Following the paper's general notation for sum alpha_k phi_k(t)
                # let's define basis functions phi_0, phi_1, ..., phi_K
                
                # For a Fourier series with K+1 terms: phi_0, sin_1, cos_1, sin_2, cos_2, ...
                # If K is the total number of non-constant terms, K1 is total functions used
                # In this implementation, K refers to the highest 'k' in sin(2*pi*k*t/T) and cos(2*pi*k*t/T)
                # so we will have 2*K + 1 coefficients (1 constant, K sin, K cos)
                # However, the paper uses sum from k=0 to K1, suggesting K1+1 distinct functions.
                # Let's align with the sum from k=0 to K:
                if k_val % 2 != 0: # Odd k_val corresponds to sine
                    Phi[j, k_val] = fourier_basis_function(t_j, (k_val + 1) // 2, T_period, is_cosine=False)
                else: # Even k_val corresponds to cosine
                    Phi[j, k_val] = fourier_basis_function(t_j, k_val // 2, T_period, is_cosine=True)

        # Re-evaluating the Phi matrix construction based on the paper's Fourier definition:
        # phi_0(t) = 1
        # phi_{2k-1}(t) = sqrt(2/T) * sin(2*pi*k*t / T)
        # phi_{2k}(t) = sqrt(2/T) * cos(2*pi*k*t / T)
        # So for K+1 coefficients, where K is the highest index.
        # This implies K can be 2*max_k or 2*max_k - 1.
        # Let's interpret K as K1 or K2 in the summation limit.
        # If K_limit = K1 or K2 from the paper.
        
        # A clearer Fourier basis construction:
        # K_limit is the maximum index in the sum, so K_limit+1 total functions.
        # The paper defines phi_0=1, phi_2k-1=sin, phi_2k=cos
        # So if K_limit is even, say K_limit = 2*m, we have 1 (phi_0) + m sin terms + m cos terms
        # If K_limit is odd, say K_limit = 2*m-1, we have 1 (phi_0) + m sin terms + (m-1) cos terms
        # This makes the mapping tricky.
        # Let's simplify and use K as the number of sine/cosine pairs.
        # So total functions will be 1 (constant) + K_fourier_pairs * 2.
        # If K from the input means K1 or K2 from the sum, then total basis functions are K+1.
        # And we need to define these K+1 basis functions.

        # Let's use K as the parameter controlling the number of Fourier components,
        # such that we have `1 + 2*K` basis functions: 1 constant, K sines, K cosines.
        # This means the sum limit in the paper's notation would be K1 = 2*K.
        # This matches the common interpretation of Fourier series.
        Phi = np.zeros((len(time_points), 1 + 2 * K))
        for j, t_j in enumerate(time_points):
            Phi[j, 0] = 1.0 # phi_0(t)
            for k_val in range(1, K + 1):
                # Index 2*k_val - 1 for sine term
                Phi[j, 2 * k_val - 1] = fourier_basis_function(t_j, k_val, T_period, is_cosine=False)
                # Index 2*k_val for cosine term
                Phi[j, 2 * k_val] = fourier_basis_function(t_j, k_val, T_period, is_cosine=True)
        
        # Calculate coefficients using least squares: alpha = (Phi'Phi)^-1 Phi'y
        try:
            coefficients = np.linalg.solve(Phi.T @ Phi, Phi.T @ discrete_data)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix, use pseudo-inverse
            coefficients = np.linalg.pinv(Phi) @ discrete_data

        def functional_representation(t_eval):
            Phi_eval = np.zeros((len(np.atleast_1d(t_eval)), 1 + 2 * K))
            for j, t in enumerate(np.atleast_1d(t_eval)):
                Phi_eval[j, 0] = 1.0
                for k_val in range(1, K + 1):
                    Phi_eval[j, 2 * k_val - 1] = fourier_basis_function(t, k_val, T_period, is_cosine=False)
                    Phi_eval[j, 2 * k_val] = fourier_basis_function(t, k_val, T_period, is_cosine=True)
            return Phi_eval @ coefficients

    elif basis_type == 'legendre':
        T_min = np.min(time_points)
        T_max = np.max(time_points)

        Phi = np.zeros((len(time_points), K + 1))
        for j, t_j in enumerate(time_points):
            for k_val in range(K + 1):
                Phi[j, k_val] = legendre_basis_function(t_j, k_val, T_min, T_max)
        
        try:
            coefficients = np.linalg.solve(Phi.T @ Phi, Phi.T @ discrete_data)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(Phi) @ discrete_data

        def functional_representation(t_eval):
            Phi_eval = np.zeros((len(np.atleast_1d(t_eval)), K + 1))
            for j, t in enumerate(np.atleast_1d(t_eval)):
                for k_val in range(K + 1):
                    Phi_eval[j, k_val] = legendre_basis_function(t, k_val, T_min, T_max)
            return Phi_eval @ coefficients
    else:
        raise ValueError("basis_type must be 'fourier' or 'legendre'")

    return coefficients, functional_representation

def calculate_bic(discrete_data, fitted_data, num_coefficients, num_observations):
    """
    Calculates the Bayesian Information Criterion (BIC) for a given fit.
    BIC = n * log(MSE) + k * log(n)
    where n is the number of observations, k is the number of parameters (coefficients).
    As per the paper's formula: BIC = n * ln(e'e / n) + k * ln(n)
    where e'e is the sum of squared residuals, n is J* (number of observations), k is K1+1.
    """
    residuals = discrete_data - fitted_data
    sum_sq_residuals = np.sum(residuals**2)
    
    # Paper's formula: BIC(y(t)) = ln(e'e / J*) + (K1+1) * ln(J*) / J*
    # This formula looks slightly different from the standard BIC.
    # Let's use the standard BIC for now and keep the paper's formula in mind.
    # Standard BIC: k * ln(n) - 2 * ln(L_hat)
    # Assuming normally distributed errors, -2*ln(L_hat) approx n * ln(SSE/n)
    
    # Following the paper's specific formula:
    J_star = num_observations # J* in the paper, which is the number of time points
    K_plus_1 = num_coefficients # K1+1 in the paper, number of estimated coefficients
    
    if J_star == 0:
        return np.inf # Avoid division by zero
    
    # e'e is sum_sq_residuals
    # ln(e'e / J*) + (K1+1) * ln(J*) / J*
    # The paper has ln(e'e/n) + (k+1) * ln(n) / n in text, but lnJ* / J* in formula
    # Let's use the explicit formula given by BIC(y(t)) = ln(e'e / J*) + (K1+1) * ln(J*) / J*
    # But it also mentions BIC = ln(e'e/n) + k*ln(n) in a general sense.
    # The image shows: BIC(y(t)) = ln(e'e / 2) + (K1+1) * (lnJ* / J*)
    # Let's use the one explicitly written in the formula section (page 3, below "The BIC value for y(t) is expressed by the following formula:")
    # BIC(y(t)) = ln(e'e / J*) + (K1+1) * (lnJ* / J*)
    # Wait, the image formula has ln(e'e / 2) which is very unusual.
    # Let's assume it was a typo and it should be ln(e'e / J_star) or the standard form.
    # Given the general description "BIC = n * log(MSE) + k * log(n)", and the formula in the text "BIC(y(t)) = ln(e'e / J*) + (K1+1) * ln(J*) / J*",
    # and the image shows "BIC(y(t)) = ln(e'e / 2) + (K1+1) * (lnJ* / J*)".
    # This is confusing. I will use the standard BIC formula, which is widely accepted,
    # as the paper's formula seems inconsistent or specific to their implementation.
    # Standard BIC = num_observations * np.log(sum_sq_residuals / num_observations) + num_coefficients * np.log(num_observations)

    # Let's stick to the paper's exact formula as seen in the image, assuming '2' is a constant multiplier not 'J*'.
    # This implies e'e/2 is part of the calculation, which is odd.
    # Re-reading: "where e = (e1, ..., ej*)', ej = Yj – Σκο άκφκ(tj), j = 1, 2, . . ., J*."
    # So e'e is the sum of squared residuals. The division by 2 in the ln term is indeed unusual.
    # Let's assume it's actually: ln(e'e) + (K1+1) * (lnJ* / J*) - ln(2)
    # This also does not make sense.
    # The most common form of BIC for OLS is: n * log(SSE/n) + k * log(n)
    # The paper's text "BIC measures the exactness of fit" and mentions Shmueli (2010),
    # which generally uses standard BIC.
    # Let's assume the formula in the paper's text has a typo, and the actual formula used
    # is the standard one, or a variant that reduces to it for comparison.
    # Given the confusion, I will implement a standard BIC and note the discrepancy.
    
    # Standard BIC implementation:
    if num_observations <= num_coefficients: # Or if MSE is zero due to perfect fit
        return np.inf # Penalize models with more parameters than data points or perfect fit
    
    mse = sum_sq_residuals / num_observations
    if mse <= 0: # Avoid log of non-positive number
        return np.inf

    bic_val = num_observations * np.log(mse) + num_coefficients * np.log(num_observations)
    return bic_val

def select_optimal_K(time_points, discrete_data, max_K, T_period, basis_type='fourier'):
    """
    Selects the optimal number of basis functions K using BIC.
    """
    best_bic = np.inf
    optimal_K = 0
    optimal_coeffs = None
    optimal_func = None

    # Iterating K from 0 (constant function) up to max_K
    for K_current in range(max_K + 1):
        # The number of coefficients depends on the basis type and K_current
        if basis_type == 'fourier':
            num_coeffs = 1 + 2 * K_current # 1 constant, K sines, K cosines
        elif basis_type == 'legendre':
            num_coeffs = K_current + 1 # P_0 to P_K_current
        else:
            raise ValueError("basis_type must be 'fourier' or 'legendre'")
        
        if num_coeffs > len(time_points):
            # Cannot fit more coefficients than data points
            break

        coeffs, func = fit_basis_functions(time_points, discrete_data, K_current, T_period, basis_type)
        
        fitted_values = func(time_points)
        current_bic = calculate_bic(discrete_data, fitted_values, num_coeffs, len(time_points))

        if current_bic < best_bic:
            best_bic = current_bic
            optimal_K = K_current
            optimal_coeffs = coeffs
            optimal_func = func
            
    return optimal_K, optimal_coeffs, optimal_func, best_bic

# --- Example Usage ---
if __name__ == "__main__":
    # --- Assumptions ---
    # 1. We are using the Fourier basis system as primarily described, or Legendre.
    # 2. 'T' in the Fourier basis function refers to the total period of the data,
    #    i.e., the range of time_points. For simplicity, we assume data is observed over [0, T].
    #    If the time interval is [t_min, t_max], then T = t_max - t_min.
    # 3. For the 'K' parameter: In the Fourier case, the code uses K to denote the highest
    #    frequency component, resulting in 1 (constant) + K (sin) + K (cos) = 2K+1 coefficients.
    #    In the Legendre case, K denotes the highest polynomial degree, resulting in K+1 coefficients.
    # 4. The `calculate_bic` function uses a standard BIC formulation due to ambiguity in the paper's formula.
    # 5. The input discrete data for a single function `y(t)` or `x(t)` is provided as a 1D array,
    #    and corresponding `time_points` as another 1D array.

    # Generate some synthetic discrete data (mimicking a time series)
    np.random.seed(42)
    num_points = 50
    # Let's say our time interval is from 1993 to 2011, so T_period = 2011 - 1993 = 18
    T_min_example = 1993
    T_max_example = 2011
    T_period_example = T_max_example - T_min_example # 18 years

    # Create irregular time points for a more realistic scenario
    time_points_y = np.sort(np.random.uniform(T_min_example, T_max_example, num_points))
    # Create some underlying 'true' function, e.g., a combination of sines and cosines
    true_y = (2 * np.sin(2 * np.pi * (time_points_y - T_min_example) / T_period_example) +
              1.5 * np.cos(4 * np.pi * (time_points_y - T_min_example) / T_period_example) + 3)
    discrete_data_y = true_y + np.random.normal(0, 0.5, num_points) # Add some noise

    print("--- Transforming Y(t) using Fourier Basis ---")
    max_fourier_components = 5 # Max 'K' for sin/cos pairs
    optimal_K_y, coeffs_y, func_y, bic_y = select_optimal_K(
        time_points_y, discrete_data_y, max_fourier_components, T_period_example, basis_type='fourier'
    )
    print(f"Optimal K for Y(t) (Fourier): {optimal_K_y}")
    print(f"BIC value: {bic_y:.2f}")
    # print("Estimated coefficients for Y(t):\n", coeffs_y)

    # Let's generate another synthetic dataset for X(t)
    time_points_x = np.sort(np.random.uniform(T_min_example, T_max_example, num_points))
    true_x = (3 * np.cos(2 * np.pi * (time_points_x - T_min_example) / T_period_example) -
              1 * np.sin(6 * np.pi * (time_points_x - T_min_example) / T_period_example) + 5)
    discrete_data_x = true_x + np.random.normal(0, 0.7, num_points) # Add some noise

    print("\n--- Transforming X(t) using Fourier Basis ---")
    optimal_K_x, coeffs_x, func_x, bic_x = select_optimal_K(
        time_points_x, discrete_data_x, max_fourier_components, T_period_example, basis_type='fourier'
    )
    print(f"Optimal K for X(t) (Fourier): {optimal_K_x}")
    print(f"BIC value: {bic_x:.2f}")
    # print("Estimated coefficients for X(t):\n", coeffs_x)

    print("\n--- Transforming Y(t) using Legendre Basis ---")
    max_legendre_degree = 10 # Max degree for Legendre polynomials
    optimal_K_y_leg, coeffs_y_leg, func_y_leg, bic_y_leg = select_optimal_K(
        time_points_y, discrete_data_y, max_legendre_degree, T_period_example, basis_type='legendre'
    )
    print(f"Optimal K for Y(t) (Legendre, degree): {optimal_K_y_leg}")
    print(f"BIC value: {bic_y_leg:.2f}")
    # print("Estimated coefficients for Y(t) (Legendre):\n", coeffs_y_leg)

    # You can now use `func_y` and `func_x` to evaluate the smoothed functional data at any time `t`.
    # For example, to get a dense representation for plotting:
    time_grid = np.linspace(T_min_example, T_max_example, 200)
    smoothed_y_fourier = func_y(time_grid)
    smoothed_x_fourier = func_x(time_grid)
    smoothed_y_legendre = func_y_leg(time_grid)

    # --- Plotting (requires matplotlib) ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(time_points_y, discrete_data_y, label='Discrete Data Y(t)', s=10, alpha=0.7)
    plt.plot(time_grid, smoothed_y_fourier, color='red', label=f'Functional Y(t) (Fourier, K={optimal_K_y})')
    plt.plot(time_grid, func_y_leg(time_grid), color='green', linestyle='--', label=f'Functional Y(t) (Legendre, K={optimal_K_y_leg})')
    plt.title('Transformation of Y(t) to Functional Data')
    plt.xlabel('Time')
    plt.ylabel('Y Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(time_points_x, discrete_data_x, label='Discrete Data X(t)', s=10, alpha=0.7, color='purple')
    plt.plot(time_grid, smoothed_x_fourier, color='orange', label=f'Functional X(t) (Fourier, K={optimal_K_x})')
    plt.title('Transformation of X(t) to Functional Data')
    plt.xlabel('Time')
    plt.ylabel('X Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("1.png")
    
    # Example of how to access the individual basis functions for plotting (Fourier)
    plt.figure(figsize=(10, 5))
    for k_val in range(optimal_K_y + 1):
        if k_val == 0:
            basis_func_values = [fourier_basis_function(t, 0, T_period_example, is_cosine=True) for t in time_grid]
            plt.plot(time_grid, basis_func_values, label=f'phi_0(t) (const)', linestyle=':')
        else:
            basis_func_sin_values = [fourier_basis_function(t, k_val, T_period_example, is_cosine=False) for t in time_grid]
            basis_func_cos_values = [fourier_basis_function(t, k_val, T_period_example, is_cosine=True) for t in time_grid]
            plt.plot(time_grid, basis_func_sin_values, label=f'phi_sin_{k_val}(t)')
            plt.plot(time_grid, basis_func_cos_values, label=f'phi_cos_{k_val}(t)', linestyle='--')
    plt.title(f'First {optimal_K_y} Fourier Basis Functions (including constant)')
    plt.xlabel('Time')
    plt.ylabel('Basis Function Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2.png")

    plt.figure(figsize=(10, 5))
    for k_val in range(optimal_K_y_leg + 1):
        basis_func_values = [legendre_basis_function(t, k_val, T_min_example, T_max_example) for t in time_grid]
        plt.plot(time_grid, basis_func_values, label=f'P_{k_val}(t)')
    plt.title(f'First {optimal_K_y_leg} Legendre Basis Functions (orthonormalized)')
    plt.xlabel('Time')
    plt.ylabel('Basis Function Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("3.png")