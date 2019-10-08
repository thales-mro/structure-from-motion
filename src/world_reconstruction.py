import cv2
import numpy as np
from klt import KLT
from math import sqrt


class WorldReconstruction:
    """
    A class used to execute the World Reconstruction operations

    Methods
    -------
    execute(input_path, output_path, operation)
        Execeute the 3D reconstruction operation on a video
    """

    def __init__(self):

        # Create KLT algorithm
        self.klt = KLT(size=(15, 15))
    

    def execute(self, input_path, output_path, operation=2, max_frames=-1, print_frames=False):
        """
        It executes the reconstrution for a video file

        Keyword arguments:
        input_path -- the input video path
        output_path -- the output video path
        operation -- operation to apply on the frame
        max_frames -- maximum number of frames to process
        print_frames -- flag to print the current frame
        """

        # Parameters for KLT
        #lkt_params = dict(winSize=(15,15), maxLevel=0, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1, 0.01))
        
        index = 1
        
        # Previous frame gray
        previous_frame_gray = None

        # Previous keypoints
        previous_keypoints = None
        
        # Open the video
        video_capture = cv2.VideoCapture(input_path)

        # Read the first frame
        success, current_frame = video_capture.read()
        
        # For each frame
        while success:
    
            print(f"Processing Frame {index}")

            # Convert the frame to gray
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Get the current frame keypoints
            current_keypoints = cv2.goodFeaturesToTrack(current_frame_gray, maxCorners=30, qualityLevel=0.2, minDistance=0.5, useHarrisDetector=True)
            
            # If the previous frame is not set
            if previous_frame_gray is not None:

                # Calculate the optical flow with opencv
                #keypoints_opencv, st, _ = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, previous_keypoints, None, **lkt_params)
                #keypoints_opencv = keypoints_opencv[st==1]
                
                # Calculate the optical flow with KLT
                keypoints = self.klt.calc(previous_frame_gray, current_frame_gray, previous_keypoints)
            
                if operation == 0:
                    
                    for i in np.int0(current_keypoints):
                        
                        # Get the point
                        x, y = i.ravel()
                        
                        # Draw the circles
                        output_frame = cv2.circle(current_frame, (x, y), 5, 255, -1)

                elif operation == 1:
                    
                    # Draw our flow
                    output_frame = self.plot_flow(current_frame, previous_keypoints, keypoints, (0, 0, 255))
                    
                    # Draw opencv flow
                    #output_frame = self.plot_flow(current_frame, previous_keypoints, keypoints_opencv, (255, 255, 255), output_frame)
                
                elif operation == 2:
                    pass
                
                
                # Save the current frame as an image
                if print_frames:
                    cv2.imwrite(f"output/frame-{index}.jpg", output_frame)
                            
            index += 1

            if max_frames > 0 and index > max_frames:
                break

            # Set the previous values
            previous_frame_gray = current_frame_gray
            
            # Update the keypoints
            previous_keypoints = current_keypoints

            # Read the next frame
            success, current_frame = video_capture.read()


    def plot_flow(self, current_frame, previous_keypoints, keypoints, color=(255, 255, 255), output_frame=None):
                    
        for previous, current in zip(previous_keypoints, keypoints):
            
            # Get the line params
            a, b = previous.ravel()
            c, d = current.ravel()
            
            if a - c > 0:
                c = int(c*1.2)
            else:
                c = int(c//1.2)
                
            if b - d > 0:
                d = int(d*1.2)
            else:
                d = int(d//1.2)
                
            if c < 0 or c > current_frame.shape[1] or d < 0 or d > current_frame.shape[1]:
                continue
            
            # Draw the line
            output_frame = cv2.arrowedLine(current_frame,  (a,b), (c,d), color, 2, tipLength=0.5)
                
            
        return output_frame
