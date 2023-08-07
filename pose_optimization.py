import numpy as np
import pandas as pd

#constraint1: distance between poses 
def f1(delta_x,delta_y):
    return np.sqrt(delta_x**2 + delta_y**2 )
#constraint2: curvature
def f2(delta_x,delta_y):
    return 2 * delta_y / (delta_x**2 + delta_y**2)
#constraint3: heading angle
def f3(delta_theta):
    return delta_theta

def get_observation_model(delta_z):
    delta_x, delta_y, delta_theta = delta_z
    return np.array([f1(delta_x,delta_y),f2(delta_x,delta_y),f3(delta_theta)])

def compute_jacobian(delta_z):
    delta_x, delta_y, delta_theta = delta_z
    eps = 1e-6
    # Calculate the terms used in the Jacobian matrix
    term1 = delta_x / np.sqrt(delta_x**2 + delta_y**2+eps)
    term2 = delta_y / np.sqrt(delta_x**2 + delta_y**2+eps)
    term3 = -4 * delta_x * delta_y / (delta_x**2 + delta_y**2+eps)**2
    term4 = 2 * (delta_x**2 - delta_y**2) / (delta_x**2 + delta_y**2+eps)**2+1e-6

    # Construct the Jacobian matrix
    jacobian = np.array([[term1, term2, 0],
                         [term3, term4, 0],
                         [0,     0,     1]])
    
    return jacobian
    
def compute_residual(delta_z, obs):
    obs_model = get_observation_model(delta_z)
    residual = obs - obs_model
    return residual

def cost_function(search_direction, Jacobian, residuals, sigma):
    d = search_direction
    J = Jacobian
    risiduals = np.array(residuals)
    S = sigma
    cost =0
    for r in risiduals:
        cost += d.T @ J.T @ np.linalg.inv(S) @ J @ d \
            - 2 * d.T @ J.T @ np.linalg.inv(S) @ r \
            + r.T @ np.linalg.inv(S) @ r
            
    return cost
    
def get_search_direction(risidual,sigma,J,alpha=0.01):
    r = risidual
    S = sigma
    #update search direction
    M = J.T @ np.linalg.inv(sigma) @ J

    search_direction = 2*alpha * np.linalg.inv(M) @ r.T @ sigma.T @ J
    
    return search_direction

    
def gradient_descent(observations, initial_pose,initial_z,sigma,loop_closure_indices, alpha=0.001, epochs=100, tolerance=1e-3):
    # Initialize the pose estimates (delta_x, delta_y, delta_theta)
    delta_z = np.array(initial_z)

    # Initialize the absolute pose with the initial pose
    absolute_pose = np.array(initial_pose)

    # For each epoch
    for epoch in range(epochs):
        # Shuffle the dataset
        #np.random.shuffle(observations)
        # Initialize a list to store the absolute poses
        absolute_poses = [absolute_pose]
        residuals = []
        # Initialize the total cost to zero
        total_cost = 0
        # For each observation in the dataset
        for i,obs in enumerate(observations):
            print('*'*50)
            print('iteration ',i+1,' of ',len(observations))
            if i in np.array(loop_closure_indices)+1:
                print('loop closure at index',i+1)
                if i == loop_closure_indices[0]+1:
                    first_loop_pose = absolute_pose
                else:
                    print('first_loop_pose:',first_loop_pose,'current_pose:',absolute_pose,'delta_z:',delta_z)
                    delta_z = first_loop_pose - absolute_pose
            
            print('obs:',obs)
            
            
            # Compute the Jacobian
            Jacobian = compute_jacobian(delta_z)
            print('Jacobian:\n',Jacobian)

            # Compute the residual
            residual = compute_residual(delta_z, obs)
            print('residual:',residual)
            residuals.append(residual)

            # Compute the search direction
            search_direction = get_search_direction(residual ,sigma, Jacobian, alpha)

            if np.any(np.isnan(search_direction)):
                print('search_direction is nan')
                break
            # Update the pose estimates
            delta_z = -search_direction
            #print('delta_z:',delta_z)

            # Update the absolute pose by adding the incremental pose
            print('last pose {}\n + delta_z {} ='.format(absolute_pose, delta_z))
            absolute_pose += delta_z
            print('pose:',absolute_pose)

            # Add the new absolute pose to the list
            absolute_poses.append(absolute_pose)

            # Compute the cost
            cost = cost_function(search_direction, Jacobian, residuals, sigma)
            #total_cost += cost
         
        print(f"Epoch {epoch}, Cost {cost}")

        # If the cost is below the tolerance, stop the algorithm
        if total_cost < tolerance:
            return np.array(absolute_poses)

    return np.array(absolute_poses)/100
    

def add_loop_closure_to_observations(poses,observations, loop_closure_indices):
    #add new observation with [0,0,delta_theta] to observations
    first_loop_pose = poses[loop_closure_indices[0]]
    for i in loop_closure_indices[1:]:
        delta_theta = poses[i][2] - first_loop_pose[2]
        #add new observation with [0,0,delta_theta] to observations at position i+1
        observations = np.insert(observations,i+1,[0,0,delta_theta],axis=0)
        print('adding loop closure at index',i+1)
    return observations
        
    
#data: x,y,theta,k,yaw_rate
data =pd.read_csv('../Aufnahmen/data/sgdVars.csv')
data['x'] = data['x'] * 100
data['y'] = data['y'] * 100
data['x_diff'] = data['x'].diff()
data['y_diff'] = data['y'].diff()
data['theta_diff']= data['theta'].diff()
data = data.fillna(0)
distance = np.sqrt(data['x_diff']**2 + data['y_diff']**2)
curvature = data['curvature']

theta_diff = data['theta_diff']
observations = np.array([distance,curvature,theta_diff]).T
#add loop closure to observations
loop_closure_indices = [152,743,1331,1916,2483]#wrong
#need to calculate right loop closure indices
observations = add_loop_closure_to_observations(data[['x','y','theta']].values,observations,loop_closure_indices)
observations = np.nan_to_num(observations,0)
print(data.head())
print(data.tail())


sigma = np.diag([0.1, 0.1, 0.1])
initial_pose = data[['x','y','theta']].iloc[0].values
initial_z = data[['x_diff','y_diff','theta_diff']].iloc[1].values
corrected_poses = gradient_descent(observations, initial_pose,initial_z, sigma, loop_closure_indices)