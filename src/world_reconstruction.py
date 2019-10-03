import cv2
import numpy as np


class WorldReconstruction:
    """
       A class used to execute the World Reconstruction operations

       Methods
       -------
        execute_video(input_path, output_path, operation)
            Execeute the ar operation on a video
       """

    def __init__(self):

        # Create SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
    

    def execute(self, input_path, output_path, operation=2, start_frame=-1, max_frames=-1, print_frames=False):
        """
        It executes the reconstrution for a video file

        Keyword arguments:
        input_path -- the input video path
        output_path -- the output video path
        operation -- operation to apply on the frame
        start_frame -- starting video frame
        max_frames -- maximum number of frames to process
        print_frames -- flag to print the current frame
        """

        # Parameters for KLT
        lkt_params = dict(winSize=(15,15), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Open the video
        video_capture = cv2.VideoCapture(input_path)

        # Read the first frame
        success, current_frame = video_capture.read()
        
        index_2 = 0
        while success and index_2 < start_frame:
            success, current_frame = video_capture.read()
            index_2 += 1 
        
        index = 0
        
        # Convert the frame to gray
        previous_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Get the best corners
        previous_corners = cv2.goodFeaturesToTrack(previous_frame_gray, maxCorners=100, qualityLevel=0.2, minDistance=10, useHarrisDetector=True)
        
        # For each frame
        while success:
    
            print(f"Processing Frame {index}")
            
            # Read the second frame
            success, current_frame = video_capture.read()
        
            # Convert the frame to gray
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Recalculate the best corners every 5 frames
            current_corners = cv2.goodFeaturesToTrack(current_frame_gray, maxCorners=100, qualityLevel=0.2, minDistance=7, useHarrisDetector=True)

            # Calculate the optical flow with KLT
            corners, st, _ = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, previous_corners, current_corners, **lkt_params)
            
            # Select the good points
            good_current = corners[st==1]
            good_previous = previous_corners[st==1]
        
        
            if operation == 0:
                
                for i in np.int0(current_corners):
                    
                    # Get the point
                    x, y = i.ravel()
                    
                    # Draw the circles
                    output_frame = cv2.circle(current_frame, (x, y), 5, 255, -1)

            elif operation == 1:
                        
                # Draw the tracking
                for i, (current, previous) in enumerate(zip(good_current, good_previous)):
                    
                    # Get the line params
                    a, b = current.ravel()
                    c, d = previous.ravel()
                    
                    # Draw the line
                    output_frame = cv2.arrowedLine(current_frame, (c, d), (a, b), (255, 0, 0), 2, tipLength=0.3)
            
            elif operation == 2:
                pass
            
                
            # Save the current frame as an image
            if print_frames:
                cv2.imwrite(f"output/frame-{index}.jpg", output_frame)

            index += 1

            if max_frames > 0 and index > max_frames:
                break

            # Set the previous values
            previous_frame = current_frame
            previous_corners = current_corners

            # Read the next frame
            success, current_frame = video_capture.read()
