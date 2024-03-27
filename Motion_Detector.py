import cv2

def motion_detection():
    # Create a background subtractor object
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening camera")
        return

    try:
        while True:
            # Read a new frame
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the background subtractor to get the foreground mask
            fg_mask = background_subtractor.apply(frame)

            # Optional: remove shadows (which are gray)
            _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

            # Display the original frame and the foreground mask
            cv2.imshow('Frame', frame)
            cv2.imshow('Foreground', fg_mask)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

motion_detection()
