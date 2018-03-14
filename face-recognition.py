import cv2
import os
import numpy as np

subjects = ["", "Nathan Cashmore"]
last_face = ""

def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path)
    
    faces = []
    labels = []

    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;
            
        label = int(dir_name.replace("s", ""))
        
        subject_dir_path = data_folder_path + "/" + dir_name
        
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))



#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def say(sentence):
	from gtts import gTTS
	import os
	tts = gTTS(text=sentence, lang='en')
	tts.save("sentence.mp3")
	os.system("mpg123 sentence.mp3")

def predict(test_img):	
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if face is None:
		return (img, "")

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]

    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    
    #draw name of predicted person where confidence HIGH
    if confidence < 50.0:
    	draw_text(img, label_text, rect[0], rect[1]-5)
    	
    return (img, label_text)
    
if __name__ == "__main__":

	vc = cv2.VideoCapture(0)

	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	else:
		rval = False

	while rval:
		rval, frame = vc.read()

		displayImage, person_seen = predict(frame)

		if last_face != person_seen and person_seen is not "":
			say("Hello " + person_seen)

		if last_face != person_seen and person_seen is "":
			say("Goodbye.  See you soon")

		last_face = person_seen

		cv2.imshow('img', displayImage)
		key = cv2.waitKey(20)
		if key == 27: # exit on ESC
			break






