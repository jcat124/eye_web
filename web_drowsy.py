from flask import Flask,render_template,Response
import cv2
# model from https://github.com/italojs/facial-landmarks-recognition
import winsound
import dlib
import imutils
import pyttsx3
from pyttsx3.engine import Engine
import speech_recognition as sr
from imutils import face_utils
from scipy.spatial import distance
import time

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            def eye_aspect_ratio(eye):
                A = distance.euclidean(eye[1], eye[5])
                B = distance.euclidean(eye[2], eye[4])
                C = distance.euclidean(eye[0], eye[3])
                ear = (A + B) / (2.0 * C)
                return ear


            thresh = 0.25
            frame_check = 20
            detect = dlib.get_frontal_face_detector()
            # Dat file is the crux of the code
            predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
            cap = cv2.VideoCapture(0)
            flag = 0

            while True:
                ret, frame = cap.read()
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                subjects = detect(gray, 0)
                for subject in subjects:
                    shape = predict(gray, subject)
                    shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    if ear < thresh:
                        flag += 1
                        #print (flag)
                        if flag >= frame_check:
                            cv2.putText(frame, "ALERT! ALERT! ALERT! ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "ALERT! ALERT! ALERT! ALERT!", (10, 325),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            time.sleep(3)
                            for x in range(5):
                                winsound.Beep(2000, 500)
                            time.sleep(1)
                            #print ("Drowsy")
                            # Initialize recognizer class (for rectext_audio = r.listen(source)ognizing the speech)
                            engine = pyttsx3.init()
                            r = sr.Recognizer()
                            # Reading Microphone as source
                            # listening the speech and store in audio_text variable

                            with sr.Microphone() as source:
                                engine.say('Do you want questions, your drowsy?')
                                engine.runAndWait()
                                winsound.Beep(2999, 100)
                                audio_text = r.listen(source)
                                try:
                                    text = r.recognize_google(audio_text)
                                except sr.UnknownValueError:
                                    for x in range(10):
                                        winsound.Beep(10000, 700)
                                    audio_text = r.listen(source)
                                    text = r.recognize_google(audio_text)
                                    if 'yes' in text:
                                        engine.say('What is 2 plus 3?')
                                        engine.runAndWait()
                                        winsound.Beep(2999, 100)
                                        audio_text = r.listen(source)
                                        try:
                                            text = r.recognize_google(audio_text)
                                        except sr.UnknownValueError:
                                            if '5' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'five' in text:
                                                engine.say('Right Answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        else:
                                            if '5' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'five' in text:
                                                engine.say('Right Answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        engine.say('How do you spell cat?')
                                        engine.runAndWait()
                                        winsound.Beep(2999, 100)
                                        audio_text = r.listen(source)
                                        try:
                                            text = r.recognize_google(audio_text)
                                        except sr.UnknownValueError:
                                            if 'cat' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'CA EDD' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif "casd" in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        else:
                                            if 'cat' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'CA EDD' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif "casd" in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        engine.say('What is 1 plus 1')
                                        engine.runAndWait()
                                        winsound.Beep(2999, 100)
                                        audio_text = r.listen(source)
                                        try:
                                            text = r.recognize_google(audio_text)
                                        except sr.UnknownValueError:
                                            if "two" in text:
                                                engine.say("Right answer")
                                                engine.runAndWait()
                                            elif 'do' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif '2' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        else:
                                            if "two" in text:
                                                engine.say("Right answer")
                                                engine.runAndWait()
                                            elif 'do' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif '2' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'dude' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                    try:
                                        print("Text: "+r.recognize_google(audio_text))
                                    except:
                                        print("'No questions, drive properly'")

                                else:
                                    if 'yes' in text:
                                        engine.say('What is 2 plus 3?')
                                        engine.runAndWait()
                                        winsound.Beep(2999, 100)
                                        audio_text = r.listen(source)
                                        try:
                                            text = r.recognize_google(audio_text)
                                        except sr.UnknownValueError:
                                            if '5' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'five' in text:
                                                engine.say('Right Answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        else:
                                            if '5' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif '5' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                        engine.say('How do you spell cat?')
                                        engine.runAndWait()
                                        winsound.Beep(2999, 100)
                                        audio_text = r.listen(source)
                                        try:
                                            text = r.recognize_google(audio_text)
                                        except sr.UnknownValueError:
                                            if 'cat' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'CA EDD' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif "casd" in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        else:
                                            if 'cat' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'CA EDD' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif "casd" in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        engine.say('What is 1 plus 1')
                                        engine.runAndWait()
                                        winsound.Beep(2999, 100)
                                        audio_text = r.listen(source)
                                        try:
                                            text = r.recognize_google(audio_text)
                                        except sr.UnknownValueError:
                                            if "two" in text:
                                                engine.say("Right answer")
                                                engine.runAndWait()
                                            elif 'do' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif '2' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                        else:
                                            if "two" in text:
                                                engine.say("Right answer")
                                                engine.runAndWait()
                                            elif 'do' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif '2' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            elif 'dude' in text:
                                                engine.say('Right answer')
                                                engine.runAndWait()
                                            else:
                                                engine.say('That is the wrong anwser')
                                                engine.runAndWait()
                                    try:
                                        print("Text: "+r.recognize_google(audio_text))
                                    except:
                                        print("'No questions, drive properly'")

                    else:
                        flag = 0
                #cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            cv2.destroyAllWindows()
            cap.release()

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
    
@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)