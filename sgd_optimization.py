import numpy as np
from scipy.optimize import minimize

def get_increment_poses(df):
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert 'theta' in df.columns
    df['delta_x'] = df['x'].diff()
    df['delta_y'] = df['y'].diff()
    df['delta_theta'] = df['theta'].diff()
    
#constraint1: distance between poses 
def f1(delta_x,delta_y):
    return np.sqrt(delta_x**2 + delta_y**2 )
#constraint2: curvature
def f2(delta_x,delta_y):
    return 2 * delta_y / (delta_x**2 + delta_y**2)
#constraint3: heading angle
def f3(delta_theta):
    return delta_theta

def compute_jacobian(delta_x, delta_y):
    # Calculate the terms used in the Jacobian matrix
    term1 = delta_x / np.sqrt(delta_x**2 + delta_y**2)
    term2 = delta_y / np.sqrt(delta_x**2 + delta_y**2)
    term3 = -4 * delta_x * delta_y / (delta_x**2 + delta_y**2)**2
    term4 = 2 * (delta_x**2 - delta_y**2) / (delta_x**2 + delta_y**2)**2

    # Construct the Jacobian matrix
    jacobian = np.array([[term1, term2, 0],
                         [term3, term4, 0],
                         [0,     0,     1]])
    
    return jacobian

def observation_model(delta_z):
    # Extract the incremental poses
    delta_x, delta_y, delta_theta = delta_z

    # Calculate the values of the observation model
    f1 = np.sqrt(delta_x**2 + delta_y**2)  # Equation 7a
    f2 = 2 * delta_y / (delta_x**2 + delta_y**2)  # Equation 7b
    f3 = delta_theta  # Equation 7c

    return np.array([f1, f2, f3])

def cost_function(delta_z, obs, sigma):
    # Calculate the observation model and its residuals
    obs_model = observation_model(delta_z)
    residuals = obs - obs_model

    # Compute the cost as the weighted sum of squared residuals
    cost = np.sum(residuals**2 / sigma)

    return cost

def cost_gradient(delta_z, obs, sigma):
    # Extract the incremental poses
    delta_x, delta_y, delta_theta = delta_z

    # Calculate the Jacobian matrix
    jacobian = compute_jacobian(delta_x, delta_y)

    # Calculate the observation model and its residuals
    obs_model = observation_model(delta_z)
    residuals = obs - obs_model

    # Compute the gradient of the cost
    gradient = -2 * jacobian.T @ (residuals / sigma)

    return gradient

def stochastic_gradient_descent(data, sigma, learning_rate=0.01, max_iter=1000, tol=1e-6):
    # Initialize the pose estimates
    delta_z = np.zeros(3)

    # Initialize the cost
    cost = np.inf

    # Iterate until convergence or maximum number of iterations
    for i in range(max_iter):
        # Randomly select a data point
        index = np.random.randint(len(data))
        delta_z_sample = data[['delta_x', 'delta_y', 'delta_theta']].iloc[index].values
        obs_sample = data[['f1', 'f2', 'f3']].iloc[index].values

        # Calculate the cost and gradient for the selected data point
        cost_new = cost_function(delta_z_sample, obs_sample, sigma)
        gradient = cost_gradient(delta_z_sample, obs_sample, sigma)

        # Update the pose estimates
        delta_z -= learning_rate * gradient

        # Check for convergence
        if np.abs(cost_new - cost) < tol:
            break

        # Update the cost
        cost = cost_new

    return delta_z
