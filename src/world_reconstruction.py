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
        compare_frame_by_frame -- compare fram by frame
        start_frame -- starting video frame
        max_frames -- maximum number of frames to process
        print_frames -- flag to print the current frame
        min_matches -- minimum number os matches to find the affine matrix
        """

        # Open the video
        video_capture = cv2.VideoCapture(input_path)

        # Read the first frame
        success, current_frame = video_capture.read()
        
        index_2 = 0
        while success and index_2 < start_frame:
            success, current_frame = video_capture.read()
            index_2 += 1 
        
        index = 0
        
        # For each frame
        while success:
    
            print(f"Processing Frame {index}")
        
            # Convert the frame to gray
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Get the best corners
            corners = cv2.goodFeaturesToTrack(current_frame_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
            if operation == 0:
                
                output_frame = current_frame.copy()
                
                for i in np.int0(corners):
                    
                    # Get the point
                    x, y = i.ravel()
                    
                    # Draw the circles
                    cv2.circle(output_frame, (x, y), 5, 255, -1)

            elif operation == 1:
                pass
            elif operation == 2:
                pass
                

                
            # Save the current frame as an image
            if print_frames:
                cv2.imwrite(f"output/frame-{index}.jpg", output_frame)

            index += 1

            if max_frames > 0 and index > max_frames:
                break

            # Set the previous frame
            previous_frame = current_frame

            # Read the next frame
            success, current_frame = video_capture.read()
