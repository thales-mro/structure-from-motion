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
    

    def execute(self, input_path, output_path, operation=2, compare_frame_by_frame=True, start_frame=-1, max_frames=-1, print_frames=False):
        """
        It executes the ar for a video file

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

        # The accumulative affine matrix
        a_all = None

        # Open the video
        video_capture = cv2.VideoCapture(input_path)

        # Read frame 1
        success, current_frame = video_capture.read()

        # Compute keypoints and descriptors of the previous frame
        keypoints_previous_frame, descriptors_previous_frame = self.sift.detectAndCompute(
            previous_frame, None)
        
        index_2 = 0
        while success and index_2 < start_frame:
            success, current_frame = video_capture.read()
            index_2 += 1 
        
        index = 0
        
        # For each frame
        while success:
            
            try:

                print(f"Processing Frame {index}")

                # Compute keypoints and descriptors of the current frame
                keypoints_current_frame, descriptors_current_frame = self.sift.detectAndCompute(
                    current_frame, None)

                # Find the matches between the previous frame and the current frame
                matches = self.matcher.match(descriptors_previous_frame, descriptors_current_frame, k=2)
                
                # Sort the matches based on the distance
                matches.sort(key=lambda x: x.distance)

                # If the matches are greate than the threshold
                if len(matches) > min_matches:

                    # Get the keypoints for each match
                    previous_frame_points = np.float32(
                        [keypoints_previous_frame[m.queryIdx].pt for m in matches])
                    current_frame_points = np.float32(
                        [keypoints_current_frame[m.trainIdx].pt for m in matches])

                    # Find the affine matrix                     
                    a = transform.get_affine_transform_matrix(previous_frame_points, current_frame_points)

                    # Set the accumulative transformations
                    if a_all is None or compare_frame_by_frame == False:
                        a_all = np.vstack((a, [0,0,1]))
                    elif np.sum(a) > 0:
                        
                        # Convert to homogeneous coordinates
                        i = np.vstack((a, [0,0,1]))
                        a_all = np.matmul(a_all, i)

                    if operation == 0:
                        
                        # Convert to opencv match class
                        matches_opencv = [cv2.DMatch(i.queryIdx, i.trainIdx, i.distance) for i in matches]
                        
                        # Draw the 20 matches
                        output_frame = cv2.drawMatchesKnn(
                            previous_frame.copy(), keypoints_previous_frame,
                            current_frame.copy(), keypoints_current_frame,
                            [matches_opencv], None, flags=2)

                    elif operation == 1:
                        img_aux = np.zeros_like(current_frame)
                        cv2.drawKeypoints(current_frame, keypoints_current_frame, img_aux, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        current_frame = img_aux
                    elif operation == 2:
                                            
                        # Warp the source image
                        source = self._warpAffine(self.source, a_all, (current_frame.shape[1], current_frame.shape[0]))
                        
                        # Warp the mask
                        target_mask = self._warpAffine(self.initial_target_mask, a_all, (current_frame.shape[1], current_frame.shape[0]))

                        # Convert it to gray scale
                        target_mask_gray = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)

                        # Define the threshold to separate foreground from background
                        _, mask = cv2.threshold(target_mask_gray, 150, 255, cv2.THRESH_BINARY_INV)

                        # Get the inverted mask
                        mask_inv = cv2.bitwise_not(mask)

                        # Get the background image
                        background = cv2.bitwise_and(current_frame, current_frame, mask=mask)

                        # Get the foreground image
                        foregound = cv2.bitwise_and(source, source, mask=mask_inv)

                        # Add both images
                        output_frame = cv2.add(background, foregound)

                    
                    # Write each frame to a new video
                    video_out.write(output_frame)
                    
                    # Save the current frame as an image
                    if print_frames:
                        cv2.imwrite(f"output/frame-{index}.jpg", output_frame)

                    index += 1

                    if max_frames > 0 and index > max_frames:
                        break

                # Set the previous frame
                previous_frame = current_frame

                if compare_frame_by_frame:
                    # Set the previous keypoints and descriptors
                    keypoints_previous_frame = keypoints_current_frame
                    descriptors_previous_frame = descriptors_current_frame

                # Read the next frame
                success, current_frame = video_capture.read()

                #transform.set_inliers_rate(0.75)
                
            except:
                continue      
