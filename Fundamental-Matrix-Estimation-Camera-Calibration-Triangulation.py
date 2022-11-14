#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import *


# In[36]:


def D2ToD3HomoCoords(vector_2d):
    return [vector_2d[0], vector_2d[1], 1]


def D3HomoCoordsToD2(vector_3d):
    return np.array([vector_3d[0] / vector_3d[2], vector_3d[1] / vector_3d[2]])


def D4HomoCoordsToD3(vector_4d):
    return np.array([vector_4d[0] / vector_4d[3], vector_4d[1] / vector_4d[3], vector_4d[2] / vector_4d[3]])


# In[3]:


"""
Access image arrays with im_dict['image_name']    
Access match array with match_dict['match_name']
"""

# load all images, normalize them
im_dir = os.path.join(os.getcwd(), "MP4_part2_data")
im_names = ["gaudi1", "gaudi2", "house1", "house2", "lab1", "lab2", "library1", "library2"]
im_paths = [os.path.join(im_dir, im_name + ".jpg") for im_name in im_names]
im_array = [Image.open(im_path) for im_path in im_paths]

pair_dict = dict()
pair_dict["gaudi"] = (im_array[0], im_array[1])
pair_dict["house"] = (im_array[2], im_array[3])
pair_dict["lab"] = (im_array[4], im_array[5])
pair_dict["library"] = (im_array[6], im_array[7])

# load all match files
match_names = ["lab", "library"]
match_paths = [os.path.join(im_dir, match_name + "_matches.txt") for match_name in match_names]
match_arrays = [np.loadtxt(match_path) for match_path in match_paths]
match_dict = dict()
for i in range(len(match_names)):
    match_dict[match_names[i]] = match_arrays[i]

# load other files
lab3d = np.loadtxt(os.path.join(im_dir, "lab_3d.txt"))
lib1_cam = np.loadtxt(os.path.join(im_dir, "library1_camera.txt"))
lib2_cam = np.loadtxt(os.path.join(im_dir, "library2_camera.txt"))
text_dict = dict()
text_dict["lab"] = lab3d
text_dict["library"] = (lib1_cam, lib2_cam)


# In[4]:


# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image:
# matches(i,0:2) is a point in the first image
# matches(i,2:4) is a corresponding point in the second image
def fit_fundamental(matches, normalize=True, debug=False):
    n_rows, n_cols = matches.shape
    U = np.empty((n_rows, 9))

    matches_copy = np.copy(matches)
    centered_matches = np.empty((matches.shape))
    norm_matches = np.empty((matches.shape))

    # if normalizing, center and normalize both sets of points
    if (normalize):
        mean_x_im_1 = np.mean(matches[:, 0])
        mean_y_im_1 = np.mean(matches[:, 1])
        mean_x_im_2 = np.mean(matches[:, 2])
        mean_y_im_2 = np.mean(matches[:, 3])
        centered_matches[:, 0] = matches[:, 0] - mean_x_im_1
        centered_matches[:, 1] = matches[:, 1] - mean_y_im_1
        centered_matches[:, 2] = matches[:, 2] - mean_x_im_2
        centered_matches[:, 3] = matches[:, 3] - mean_y_im_2
        # calculate distances from each point to origin
        norms_im_1 = np.linalg.norm(centered_matches[:, 0:2], axis=1)
        norms_im_2 = np.linalg.norm(centered_matches[:, 2:4], axis=1)
        # calculate mean squared distance between points in both images to origin
        ms_distance_im_1 = np.mean(norms_im_1)
        ms_distance_im_2 = np.mean(norms_im_2)
        # create scaling matrices T and T' (shape 2x2)
        scaling_factor = 2 ** 0.5
        s_fact_1 = scaling_factor / ms_distance_im_1
        s_fact_2 = scaling_factor / ms_distance_im_2
        T2 = np.diag([s_fact_1, s_fact_1])
        Tp2 = np.diag([s_fact_2, s_fact_2])
        # scale points in both images
        matches_1 = T2 @ centered_matches[:, 0:2].T
        matches_2 = Tp2 @ centered_matches[:, 2:4].T
        matches = np.vstack((matches_1, matches_2)).T

        if (debug):
            # for sanity, check that our matches array is now scaled as we expected
            norms_im_1 = np.linalg.norm(matches[:, 0:2], axis=1)
            norms_im_2 = np.linalg.norm(matches[:, 2:4], axis=1)
            # calculate mean squared distance between points in both images to origin
            ms_distance_im_1 = np.mean(norms_im_1)
            ms_distance_im_2 = np.mean(norms_im_2)
            print("Mean distance after normalization: " + str((ms_distance_im_1, ms_distance_im_2)))

    # construct U
    for i in range(n_rows):
        row = matches[i]
        x = row[0]
        y = row[1]
        xp = row[2]
        yp = row[3]
        U[i] = [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1]

    # do svd on U.T * U
    utu = np.matmul(U.T, U)
    u, s, v = np.linalg.svd(utu)

    # take v[-1] to get eigenvector with smallest value
    F_init = v[-1].reshape((3, 3))

    # do svd again and throw out smallest eigenvalue
    u, s, v = np.linalg.svd(F_init)
    s[2] = 0  # throw out smallest value
    s = np.diag(s)
    # reconstruct F
    F = u @ s @ v

    # if normalizing, transform F back into original coordinates
    if (normalize):
        T3 = np.array([[s_fact_1, 0, -s_fact_1 * mean_x_im_1],
                       [0, s_fact_1, -s_fact_1 * mean_y_im_1],
                       [0, 0, 1]])
        Tp3 = np.array([[s_fact_2, 0, -s_fact_2 * mean_x_im_2],
                        [0, s_fact_2, -s_fact_2 * mean_y_im_2],
                        [0, 0, 1]])
        F = Tp3.T @ F @ T3

    return F


# ## Fundamental matrix estimation from ground truth matches.
# Load the lab and library image pairs and matching points file using the starter code. Add your own code to fit a fundamental matrix to the matching points and use the sample code to visualize the results. You need to implement and compare the normalized and the unnormalized algorithms (see this lecture for the methods). For each algorithm and each image pair, report your residual, or the mean squared distance in pixels between points in both images and the corresponding epipolar lines.

# In[5]:


plot_xcoords = dict()
plot_xcoords["lab"] = [0, 4000]
plot_xcoords["library"] = [0, 3000]

NORMALIZATION = True

# In[6]:


for loc in ["lab", "library"]:
    I1, I2 = pair_dict[loc]
    matches = match_dict[loc]

    # this is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image:
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image

    N = len(matches)

    ##
    ## display two images side-by-side with matches
    ## this code is to help you visualize the matches, you don't need
    ## to use it to produce the results for the assignment
    ##

    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
    I3[:, :I1.size[0], :] = I1;
    I3[:, I1.size[0]:, :] = I2;

    im1 = np.array(I1).astype('float') / 255
    im2 = np.array(I2).astype('float') / 255
    im3 = np.array(I3).astype('float') / 255

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(im3)
    ax.plot(matches[:, 0], matches[:, 1], '+r')
    ax.plot(matches[:, 2] + I1.size[0], matches[:, 3], '+r')
    ax.plot([matches[:, 0], matches[:, 2] + I1.size[0]], [matches[:, 1], matches[:, 3]], 'r')
    plt.show()

    ##
    ## display second image with epipolar lines reprojected
    ## from the first image
    ##

    # first, fit fundamental matrix to the matches
    F = fit_fundamental(matches, NORMALIZATION);

    M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,2:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # calculate residuals
    residuals = np.mean(np.abs(pt_line_dist))
    print("Residuals for " + loc + ": " + str(residuals) + " with normalization = " + str(NORMALIZATION))

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(im2)
    ax.plot(matches[:, 2], matches[:, 3], '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')

    # display segments
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')

    # save fig
    plt.savefig(loc + "_" + ("norm" if NORMALIZATION else "no_norm") + ".png")

    #     # display whole lines
    #     NUM_LINES = max(8, matches.shape[0] // 32)
    #     # generate random indices
    #     r_indices = np.random.choice(L1.shape[0], NUM_LINES, replace=False)
    #     A = L[:,0][r_indices]
    #     B = L[:,1][r_indices]
    #     C = L[:,2][r_indices]
    #     slope = -A / B
    #     intercept = -C / B
    #     x1s = np.full(NUM_LINES, plot_xcoords[loc][0])
    #     x2s = np.full(NUM_LINES, plot_xcoords[loc][1])
    #     y1s = slope * x1s + intercept
    #     y2s = slope * x2s + intercept
    #     ax.plot([x1s, x2s], [y1s, y2s], 'g')

    plt.show()


# ## Camera calibration.
# For the lab pair, calculate the camera projection matrices by using 2D matches in both views and 3-D point coordinates given in lab_3d.txt in the data file. Refer to [this lecture](http://slazebni.cs.illinois.edu/fall22/lec15_calibration.pdf) for the calibration method. Once you have computed your projection matrices, you can evaluate them using the evaluate_points function included in the starter code, which will provide you the projected 2-D points and residual error. (Hint: For a quick check to make sure you are on the right track, empirically this residual error should be < 20 and the squared distance of your projected 2-D points from actual 2-D points should be < 4.)

# In[7]:


## Camera Calibration
def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u - points_2d[:, 0], v - points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual


# In[20]:


# calculate camera projection matrices
matches = match_dict['lab']
N = matches.shape[0]
coords_3d = np.hstack((lab3d, np.ones((N, 1))))

A1 = np.empty((2 * N, 12))
A2 = np.empty((2 * N, 12))

for i in range(N):
    A1[2 * i] = np.hstack((np.zeros(4), coords_3d[i], -matches[i][1] * coords_3d[i]))
    A1[2 * i + 1] = np.hstack((coords_3d[i], np.zeros(4), -matches[i][0] * coords_3d[i]))
    A2[2 * i] = np.hstack((np.zeros(4), coords_3d[i], -matches[i][3] * coords_3d[i]))
    A2[2 * i + 1] = np.hstack((coords_3d[i], np.zeros(4), -matches[i][2] * coords_3d[i]))

u1, s1, v1 = np.linalg.svd(A1.T @ A1)
u2, s2, v2 = np.linalg.svd(A2.T @ A2)

P1_lab = v1[-1].reshape((3, 4))
P2_lab = v2[-1].reshape((3, 4))

projected_points_1, residual_1 = evaluate_points(P1_lab, matches[:, 0:2], lab3d)
projected_points_2, residual_2 = evaluate_points(P2_lab, matches[:, 2:4], lab3d)

print("P1: ")
print(P1_lab)
print("P2: ")
print(P2_lab)
print("Residual 1:")
print(residual_1)
print("Residual 2:")
print(residual_2)

# Calculate the camera centers for the lab and library pairs using the estimated or provided projection matrices.

# In[9]:


proj_dict = {
    'lab': (P1_lab, P2_lab),
    'library': (lib1_cam, lib2_cam)
}

# In[45]:


## Camera Centers
camera_centers_dict = dict()

for loc in ['lab', 'library']:
    matches = match_dict[loc]
    N = matches.shape[0]
    P1, P2 = proj_dict[loc]
    u1, s1, v1 = np.linalg.svd(P1)
    u2, s2, v2 = np.linalg.svd(P2)
    n_space1 = v1[-1]
    n_space2 = v2[-1]
    camera_center_3d1 = D4HomoCoordsToD3(n_space1)
    camera_center_3d2 = D4HomoCoordsToD3(n_space2)
    camera_centers_dict[loc] = (camera_center_3d1, camera_center_3d2)
    print("Camera center 1 for location `" + loc + "`: " + str(camera_center_3d1))
    print("Camera center 2 for location `" + loc + "`: " + str(camera_center_3d2))


# ## Triangulation
# For the lab and library pairs, use linear least squares to triangulate the 3D position of each matching pair of 2D points given the two camera projection matrices (see [this lecture](http://slazebni.cs.illinois.edu/fall22/lec15_calibration.pdf) for the method). As a sanity check, your triangulated 3D points for the lab pair should match very closely the originally provided 3D points in lab_3d.txt. For each pair, display the two camera centers and reconstructed points in 3D. Also report the residuals between the observed 2D points and the projected 3D points in the two images.

# In[11]:


def toCrossMatrix(vector_3d):
    a1 = vector_3d[0]
    a2 = vector_3d[1]
    a3 = vector_3d[2]
    return [[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]]


# In[56]:


## Triangulation
true_pts = lab3d

for loc in ['lab', 'library']:
    matches = match_dict[loc]
    N = matches.shape[0]
    P1, P2 = proj_dict[loc]
    P1_inv, P2_inv = np.linalg.pinv(P1), np.linalg.pinv(P2)
    triangulated_coords = np.empty((N, 3))  # desired coordinates
    residuals_1 = 0
    residuals_2 = 0

    for i in range(N):
        p_1 = D2ToD3HomoCoords(matches[i][0:2])
        p_2 = D2ToD3HomoCoords(matches[i][2:4])
        x1_x = toCrossMatrix(p_1)
        x2_x = toCrossMatrix(p_2)
        prod_1 = x1_x @ P1
        prod_2 = x2_x @ P2
        U = prod_1 - prod_2
        u, s, v = np.linalg.svd(U.T @ U)
        soln = v[-1]
        triangulated_coords[i] = D4HomoCoordsToD3(soln)
        # calculate 2d coordinates of triangulated point in each image
        proj_pt_1_homo = P1 @ soln
        proj_pt_2_homo = P2 @ soln
        # convert to standard 2d coordinates
        proj_pt_1 = D3HomoCoordsToD2(proj_pt_1_homo)
        proj_pt_2 = D3HomoCoordsToD2(proj_pt_2_homo)
        # calculate distance between 2d projection of triangulated 3d pt and observed points
        residuals_1 += np.linalg.norm(proj_pt_1 - matches[i][0:2])
        residuals_2 += np.linalg.norm(proj_pt_2 - matches[i][2:4])

    avg_residuals_1 = residuals_1 / N
    avg_residuals_2 = residuals_2 / N
    print("Residuals for camera 1 for location " + loc + ": " + str(avg_residuals_1))
    print("Residuals for camera 2 for location " + loc + ": " + str(avg_residuals_2))

    if loc == 'lab':
        total_distance = 0
        for i in range(N):
            true_pt = true_pts[i]
            triangulated_pt = triangulated_coords[i]
            distance = np.linalg.norm(true_pt - triangulated_pt)
            total_distance += distance
        avg_distance = total_distance / N
        print("For sanity, avg distance between triangulated 3d coords and known 3d coords: " + str(avg_distance))

    # plotting
    xs = triangulated_coords[:, 0]
    ys = triangulated_coords[:, 1]
    zs = triangulated_coords[:, 2]
    camera_center_1, camera_center_2 = camera_centers_dict[loc]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs, ys, zs, marker='o')
    ax.scatter(camera_center_1[0], camera_center_1[1], camera_center_1[2], 'r')
    ax.scatter(camera_center_2[0], camera_center_2[1], camera_center_2[2], 'r')
    ax.text(camera_center_1[0], camera_center_1[1], camera_center_1[2], 'Camera 1')
    ax.text(camera_center_2[0], camera_center_2[1], camera_center_2[2], 'Camera 2')
    plt.show()

# In[ ]:


# In[ ]:




