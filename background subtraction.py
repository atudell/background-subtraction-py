import numpy as np
import cv2

class Video:
    
    # The object only needs to initialize on the path
    def __init__(self, path):
        self.path = path
        
    # Define the method for background subtraction, using the standard OpenCV methods
    def removeBackgroundCV(self, method):

        if method == "MOG2":
            subtraction = cv2.createBackgroundSubtractorMOG2()
        elif method == "KNN":
            subtraction = cv2.createBackgroundSubtractorKNN()
        else:
            print("Invalid input. Input must be either 'MOG2' or 'KNN', depending on prefered method")
            return None

        video = cv2.VideoCapture(self.path)
        mask_color = (0.0,0.0,0.0)

        while True:

            ret, frame = video.read()

            if ret:

                # Apply background subtraction to create a mask
                mask = subtraction.apply(frame)

                # Create 3-channel alpha mask
                mask_stack = np.dstack([mask]*3)

                # Ensures data types match up
                mask_stack = mask_stack.astype('float32') / 255.0           
                frame = frame.astype('float32') / 255.0                 

                # Blend the image and the mask
                masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
                masked = (masked * 255).astype('uint8') 

                cv2.imshow("OpenCV Method", masked)

                # Use the q button to quit the operation
                if cv2.waitKey(60) & 0xff == ord('q'):
                    break
            else:
                break

        cv2.destroyAllWindows()
        video.release()
        
    def removeBackground(self, canny_low, canny_high, min_area, max_area, mask_dilate_iter = 10, mask_erode_iter = 10, blur = 21):
        
        # initialize video from the webcam
        video = cv2.VideoCapture(self.path)
        # Set mask color to black
        mask_color = (0.0,0.0,0.0)

        while True:

            # Read from video frame
            ret, frame = video.read()

            # Test if the camera properly captured a frame. If not, break the loop
            if ret == True:

                # Read the image and covert to grayscale
                image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                # Get the area of the image as a comparison
                image_area = frame.shape[0] * frame.shape[1]
                # calculate max and min areas in terms of pixels
                max_area = max_area * image_area
                min_area = min_area * image_area

                # Use canny to make initial detection of edges. Use dilate and erode to make them more clear
                edges = cv2.Canny(image_gray, canny_low, canny_high)
                edges = cv2.dilate(edges, None)
                edges = cv2.erode(edges, None)

                # get the contours
                contour_info = [(c,cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]

                # Set up mask with a matrix of 0's
                mask = np.zeros(edges.shape, dtype = np.uint8)

                # Go through and find relevant contours and apply to mask
                for contour in contour_info:

                    # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
                    if contour[1] > min_area and contour[1] < max_area:
                        # Add contour to mask
                        mask = cv2.fillConvexPoly(mask, contour[0], (255))

                # use dilate, erode, and blur to smooth out the mask
                mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
                mask = cv2.erode(mask, None, iterations=mask_erode_iter)
                mask = cv2.GaussianBlur(mask, (blur, blur), 0)

                # Create 3-channel alpha mask
                mask_stack = np.dstack([mask]*3)

                # Ensures data types match up
                mask_stack = mask_stack.astype('float32') / 255.0           
                frame = frame.astype('float32') / 255.0                 

                # Blend the image and the mask
                masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
                masked = (masked * 255).astype('uint8') 

                cv2.imshow('img2', masked)

                # Use the q button to quit the operation
                if cv2.waitKey(60) & 0xff == ord('q'):
                    break

            else:
                break

        cv2.destroyAllWindows()
        video.release()