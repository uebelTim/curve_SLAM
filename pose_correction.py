import numpy as np
from sklearn.neighbors import NearestNeighbors

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 2D points in two datasets.
    '''
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = np.dot(A_centered.T, B_centered)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
       Vt[-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    t = centroid_B - np.dot(centroid_A, R)

    return R, t

def icp(A, B, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: aligns two 2D point clouds
    '''
    src = np.ones((3,A.shape[0]))
    dst = np.ones((3,B.shape[0]))
    src[:2,:] = np.copy(A.T)
    dst[:2,:] = np.copy(B.T)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        neighbors = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(src[:2,:].T)
        distances, indices = neighbors.kneighbors(dst[:2,:].T)

        # compute the transformation between the current source and nearest destination points
        R, t = best_fit_transform(src[:2,:].T, dst[:2,indices].T.reshape(-1, 2))

        # update the current source
        src[:2,:] = np.dot(R, src[:2,:]) + t.reshape(-1,1)

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    R, t = best_fit_transform(A, src[:2,:].T)

    return R, t



from simpleicp import PointCloud, SimpleICP
import numpy as np
from .curvature_estimation import CurvatureEstimator
from PIL import Image 
import os

curve_estimator = CurvatureEstimator.CurvatureEstimator()

frames_dir = '../Aufnahmen/data/raw/'
testframe1 = Image.open(os.path.join(frames_dir, 'frame_0001.png'))

# Read point clouds from xyz files into n-by-3 numpy arrays
X_fix = np.genfromtxt("bunny_part1.xyz")
X_mov = np.genfromtxt("bunny_part2.xyz")

# Create point cloud objects
pc_fix = PointCloud(X_fix, columns=["x", "y", "z"])
pc_mov = PointCloud(X_mov, columns=["x", "y", "z"])

# Create simpleICP object, add point clouds, and run algorithm!
icp = SimpleICP()
icp.add_point_clouds(pc_fix, pc_mov)
H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run(max_overlap_distance=1)
