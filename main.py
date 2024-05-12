import cv2
import numpy as np

# Load the pre-trained Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of users with IDs and associated images
users = [
    {"id": "Jade", "image_path": "image.jpg"}
]

threshold = 100

# Function to compare faces and recognize the user
def recognize_face(face):
    for user in users:
        user_image = cv2.imread(user["image_path"])
        user_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
        # Detect face in the user's image
        user_faces = face_cascade.detectMultiScale(user_gray, scaleFactor=1.3, minNeighbors=5)
        # Assuming there's only one face in each user image
        if len(user_faces) == 1:
            user_face = user_faces[0]
            user_face_roi = user_gray[user_face[1]:user_face[1]+user_face[3], user_face[0]:user_face[0]+user_face[2]]
            # Resize the detected face to match the size of the user's face ROI
            face_resized = cv2.resize(face, (user_face_roi.shape[1], user_face_roi.shape[0]))
            # Compute the L2 distance between the detected face and the user's face
            distance = np.linalg.norm(face_resized - user_face_roi)
            # If the distance is below the threshold, consider it a match
            if distance < threshold:
                return user["id"]
    return None


def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces and display the name
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Recognize the face
        user_id = recognize_face(face_roi)
        
        if user_id is not None:
            # Draw a rectangle around the face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display the user ID
            cv2.putText(image, str(user_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    return image

def main():
    # Option to choose between image or video stream
    choice = input("Choose an option:\n1. Image\n2. Video Stream\nEnter your choice (1/2): ")

    if choice == '1':
        # Read the image
        image_path = input("Enter the path to the image: ")
        image = cv2.imread(image_path)
        
        # Detect faces in the image
        faces_detected = detect_faces(image)
        
        # Display the image with detected faces
        cv2.imshow("Faces Detected", faces_detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif choice == '2':
        # Capture video from the default camera
        cap = cv2.VideoCapture(0)
        
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()
            
            # Detect faces in the frame
            frame_with_faces = detect_faces(frame)
            
            # Display the frame with detected faces
            cv2.imshow("Video Stream", frame_with_faces)
            
            # Press 'q' to exit the video stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        print("Invalid choice. Please choose either 1 or 2.")

if __name__ == "__main__":
    main()
