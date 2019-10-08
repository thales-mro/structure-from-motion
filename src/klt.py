import cv2
import numpy as np
from scipy import ndimage, interpolate


class KLT:
    """
    A class used to calculate the optical flow using KLT method

    Methods
    -------
    calc(size, max_iteration, min_error)
        Find the optical flow
    """

    def __init__(self, size=(15, 15), max_iteration=10, min_error=0.01):
        self.size = size
        self.max_iteration = max_iteration
        self.min_error = min_error

    def calc(self, previous_image, current_image, keypoints):

        output_points = []
        
        # Normalize the images
        previous_image = previous_image/255
        current_image = current_image/255

        # Reshape the keypoints
        keypoints = np.int0(keypoints.reshape(-1, 2))
        
        for x, y in keypoints:
            
            # Find the optical flow
            u, v = self._calc_optical_flow(previous_image, current_image, x, y)
        
            output_points.append([x+v, y+u])
    
            
        return np.float32(output_points)

    def _calc_optical_flow(self, previous_image, current_image, x, y):

        v = np.array([0., 0.])
            
        # Create the region around the keypoint
        w = self.size[0]//2
        h = self.size[1]//2
        
        # Select the neighborhood around the keypoint on the gradient image
        neighborhood = previous_image[y-h:y+h+1, x-w:x+w+1]
        
        x_diff, y_diff = np.gradient(neighborhood)
    
        # Compute the gradient
        gradients = np.array([x_diff.flatten(), y_diff.flatten()])
        
        # Compute A*A.T
        z = np.dot(gradients, gradients.T)
        
        # Find the pseudo inverse
        z_inverse = np.linalg.pinv(z)

        iteration = 0

        while iteration < self.max_iteration:
            
            iteration += 1
            
            # Get the current window
            t_win = current_image[y-w:y+w+1, x-h:x+h+1]
                        
            # Compute the difference
            i_k = (neighborhood - t_win).flatten()

            b = -1 * np.dot(gradients, i_k)
            
            n = np.dot(z_inverse, b)
            
            v += n
                
            if np.sqrt(n[0]**2 + n[1]**2) <= self.min_error:
                break
 
        
        return v
