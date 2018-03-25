import cv2
import os
import logging
import numpy as np
import time

import os.path
import json
import click
import tempfile

import google.auth.transport.grpc
import google.auth.transport.requests
import google.oauth2.credentials

from gtts import gTTS
from google.assistant.embedded.v1alpha2 import (
    embedded_assistant_pb2,
    embedded_assistant_pb2_grpc
)

from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file

from googlesamples.assistant.grpc.textinput import SampleTextAssistant
from googlesamples.assistant.grpc.pushtotalk import SampleAssistant
from googlesamples.assistant.grpc.audio_helpers import SoundDeviceStream, ConversationStream, WaveSink
from googlesamples.assistant.grpc.device_helpers import DeviceRequestHandler

subjects = ["", "Nathan", "Ona", "Ethan", "Beau", "Amy"]

face_recognizer = None

LANG = "en-GB"
DEVICE_MODEL_ID = "mirror-mirror-on-the-wall-2017-magic-pi-k0rtqx"
PROJECT = "mirror-mirror-on-the-wall-2017"

TALK_TO_MIRROR = "talk to the princess mirror"
MIRROR_START_RESPONSE = "Welcome to the all powerful mirror, what is your name"
GOODBYE = "stop"
RETRY_LIMIT = 3
MATCH_THRESHOLD = 60

ASSISTANT_API_ENDPOINT = 'embeddedassistant.googleapis.com'
DEFAULT_GRPC_DEADLINE = 60 * 3 + 5

DEFAULT_AUDIO_SAMPLE_RATE = 16000
DEFAULT_AUDIO_SAMPLE_WIDTH = 2
DEFAULT_AUDIO_ITER_SIZE = 3200
DEFAULT_AUDIO_DEVICE_BLOCK_SIZE = 6400
DEFAULT_AUDIO_DEVICE_FLUSH_SIZE = 25600

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

def train_recognizer():
    global face_recognizer
    
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
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def say(sentence):
    if sentence != "":
        tts = gTTS(text=sentence, lang='en')
        tts.save("ip-text-to-sound.mp3")
        os.system("mpg123 ip-text-to-sound.mp3")

def predict(test_img):	
    has_found_face = False
    face_is_known = False
    label_text = ""
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if face is None:
		return (img, label_text, has_found_face, face_is_known)
    else:
        has_found_face = True

    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    logging.info("<" + subjects[label] + "> @ " + str(confidence))
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    
    #draw name of predicted person where confidence HIGH
    if confidence < MATCH_THRESHOLD:
        label_text = subjects[label]
        face_is_known = True
        logging.info("<" + subjects[label] + "> @ " + str(confidence) + " --MATCH-- ")
        draw_text(img, label_text, rect[0], rect[1]-5)
    
    return (img, label_text, has_found_face, face_is_known)

def auth_with_assistant():
    # Authenticate with Google Assistant
    creds_file = os.path.join(os.path.expanduser('~/.config'),'google-oauthlib-tool','credentials.json')
    try:
        with open(creds_file, 'r') as f:
            credentials = google.oauth2.credentials.Credentials(token=None,
                                                            **json.load(f))
                                                            
            http_request = google.auth.transport.requests.Request()
            credentials.refresh(http_request)
    except Exception as e:
        logging.error('Error loading credentials: %s', e)
        logging.error('Run google-oauthlib-tool to initialize '
                      'new OAuth 2.0 credentials.')
        return
    
    # Create an authorized gRPC channel.
    grpc_channel = google.auth.transport.grpc.secure_authorized_channel(
        credentials, http_request, ASSISTANT_API_ENDPOINT)
    
    logging.info("Authenticated with assistant and set up comms channel")
    return grpc_channel
    
    
def start_conversation_with_magic_mirror(grpc_channel, person_seen):   
    logging.info('Connecting to %s', ASSISTANT_API_ENDPOINT)

    with SampleTextAssistant(LANG, DEVICE_MODEL_ID, PROJECT,
                             grpc_channel, DEFAULT_GRPC_DEADLINE) as assistant:
        tried = 0
        assistant_response = ""
        carry_on = True
        
        while assistant_response != MIRROR_START_RESPONSE:
            logging.info("Attempt to talk to Mirror attempt no " + str(tried))
            
            assistant_response = assistant.assist(text_query=TALK_TO_MIRROR)
            logging.info("RESP >" + assistant_response)
            
            if(tried == RETRY_LIMIT):
                say("Sorry, I'm not in the mood for questions " + person_seen) 
                end_conversation(grpc_channel)
                carry_on = False
                break
            
            tried = tried + 1
        
        
        if carry_on:
            if person_seen != "":
                assistant_response = assistant.assist(text_query=person_seen)            
                say(assistant_response)
            else:
                say(assistant_response)
            return True
        
        return False

def talk_to_magic_mirror(grpc_channel):
    
    continue_conversation = True;
    conversation_stream = setup_conversation_stream()
        
    logging.info(conversation_stream)
    
    device_handler = DeviceRequestHandler(PROJECT)
    
    with SampleAssistant(LANG, DEVICE_MODEL_ID, PROJECT,
                         conversation_stream,
                         grpc_channel, DEFAULT_GRPC_DEADLINE,
                         device_handler) as assistant:

        while continue_conversation:
            continue_conversation = assistant.assist()
            response = conversation_stream.text_response
            logging.info("Responded with : " + response)
            #say(response)

def end_conversation(grpc_channel):
    with SampleTextAssistant(LANG, DEVICE_MODEL_ID, PROJECT,
                             grpc_channel, DEFAULT_GRPC_DEADLINE) as assistant:
                                 
        assistant.assist(text_query=GOODBYE)
        
def setup_conversation_stream():
    audio_device = SoundDeviceStream(
            sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
            sample_width=DEFAULT_AUDIO_SAMPLE_WIDTH,
            block_size=DEFAULT_AUDIO_DEVICE_BLOCK_SIZE,
            flush_size=DEFAULT_AUDIO_DEVICE_FLUSH_SIZE
        )
        
    audio_sink = WaveSink(
        open("response.wav", 'wb'),
        sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        sample_width=DEFAULT_AUDIO_SAMPLE_WIDTH
        )
        
    # Create conversation stream with the given audio source and sink.
    conversation_stream = ConversationStream(
        source=audio_device,
        sink=audio_device,
        iter_size=DEFAULT_AUDIO_ITER_SIZE,
        sample_width=DEFAULT_AUDIO_SAMPLE_WIDTH,
    )
    
    return conversation_stream
   
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    
    # Setup logging.
    logging.basicConfig(level=logging.DEBUG)
    logging.captureWarnings(True)
   
    say("Just looking up the faces i know about")
    train_recognizer()

    say("Checking in with Google Assistant")
    grpc_channel = auth_with_assistant()

    last_seen = ""
    person_seen_count = 0
    not_seen_count = 0
    
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    say("Ok. I am ready")
    while rval:
        rval, frame = vc.read()

        displayImage, person_seen, has_found_face, face_is_known = predict(frame)

        if has_found_face == False:
            logging.info("No face:" + str(not_seen_count))
            not_seen_count = not_seen_count + 1
            
        if has_found_face:
            person_seen_count = person_seen_count + 1
            not_seen_count = 0
            logging.info("Face seen: " + str(person_seen_count))
        
        if person_seen_count == 50:
            start_conversation_with_magic_mirror(grpc_channel, "")
            talk_to_magic_mirror(grpc_channel)
            last_seen = person_seen
            person_seen_count = 0

        if not_seen_count == 100:
            last_seen = ""
            person_seen_count = 0
            not_seen_count = 0

        if has_found_face and face_is_known:
            if last_seen != person_seen:
                start_conversation_with_magic_mirror(grpc_channel, person_seen)
                talk_to_magic_mirror(grpc_channel)
                last_seen = person_seen
                person_seen_count = 0

        cv2.imshow('img', displayImage)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break








