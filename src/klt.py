import cv2
import numpy as np
from scipy import ndimage, interpolate

class KLT:
    """
    A class used to calculate the optical flow using KLT method

    Methods
    -------
    calc(input_path, output_path, operation)
        Execeute the 3D reconstruction operation on a video
    """


    def __init__(self, size=(15, 15)):
        self.size = size

    def calc(self, previous_image, current_image, keypoints):

        output_points = []
                
        # Normalize images
        previous_image = previous_image/np.amax(previous_image)
        current_image = current_image/np.amax(current_image)
        
        # Reshape the keypoints
        keypoints = np.int0(keypoints.reshape(-1, 2))
        
        for x, y in keypoints:
            
            try:
            
                # Find the optical flow
                u, v = self._calc_optical_flow(previous_image, current_image, x, y)
            
                output_points.append([x+u, y+v])
            except:
                continue
            
        return np.float32(output_points)

    def _calc_optical_flow(self, previous_image, current_image, x, y):
    
        d = np.array([0., 0.])

        # Create the region around the keypoint
        w = self.size[0]//2
        h = self.size[1]//2
        
        # Select the neighborhood around the keypoint
        neighborhood = previous_image[x-(w+1):x+w, y-h:y+(h+1)]

        # Define the kernels
        kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        kernel_y = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])

        # Compute the convolutions
        x_diff = ndimage.convolve(neighborhood, kernel_y).flatten()
        y_diff = ndimage.convolve(neighborhood, kernel_x).flatten()

        # Compute the gradient
        gradients = np.array([x_diff, y_diff])
        
        # Compute A*A.T
        z = np.dot(gradients, gradients.T)
        
        if np.linalg.det(z) == 0:
            return d
        
        # Find the inverse
        z_inverse = np.linalg.inv(z)

        # Create the interpolation sizes
        current_frame_x = np.arange(0, current_image.shape[0])
        current_frame_y = np.arange(0, current_image.shape[1])
        
        # Interpolate the current frame
        current_image_interpolated = interpolate.interp2d(current_frame_y, current_frame_x, current_image, kind='cubic')

        # The window to evaluate
        win_x = np.arange(x-(w+1), x+w, dtype=np.float32)
        win_y = np.arange(y-h, y+(h+1), dtype=np.float32)

        iteration = 0
        
        while iteration < 10:
            
            iteration += 1

            # Get the current window
            t_win = current_image_interpolated(win_y + d[1], win_x + d[0])
                        
            # Compute the difference
            b = (neighborhood - t_win).flatten()

            e = -1 * np.dot(gradients, b)
            
            d_ = np.dot(z_inverse, e)
            
            d += d_
            
            if np.hypot(d_[0], d_[1]) <= 0.03:
                break

        return d
