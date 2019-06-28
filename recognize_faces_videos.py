# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np

# constructing the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-r", "--recognizer", required=False,
                help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=False,
                help="path to label encoder")
ap.add_argument("-o", "--output", type=str,
                help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection_method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-u", "--use_embed_only", type=int, default=1,
                help="Only to use distance between embedding for recognition")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# we will be writing processed frames to disk
writer = None
# allow the camera sensor to warm up
time.sleep(2)
# loop over the frames from video file stream
while True:
    # grab a frame from the video stream which is threaded
    frame = vs.read()
    # converting BGR to RGB and downscaling the frames for speed up
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input frame, then compute
    # the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    probs = []
    for encoding in encodings:
        if args['use_embed_only'] == 1:
            match = face_recognition.compare_faces(data['encodings'], encoding)
            name = 'Unknown'
            # check to see if a match is found
            if True in match:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIds = [i for (i, b) in enumerate(match) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIds:
                    name = data['name'][i]
                    counts[name] = counts.get(name, 0) + 1
                    # determine the recognized face with the largest number of
                    # votes (note: in the event of an unlikely tie Python will
                    # select first entry in the dictionary)
                    name = max(counts, key=counts.get)
            # update the list of names
            names.append(name)
        else:
            recognizer = pickle.loads(open(args['recognizer'], 'rb').read())
            le = pickle.loads(open(args['le'], 'rb').read())
            # print(encoding.reshape(1, -1).shape)
            preds = recognizer.predict_proba(encoding.reshape(1, -1))[0]
            j = np.argmax(preds)
            probs.append(preds[j])
            names.append(le.classes_[j])
    # using only l1 distance for recognizing faces
    if args['use_embed_only'] == 1:
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    # using a trained classifier
    elif args['use_embed_only'] == 0:
        for ((top, right, bottom, left), name, prob) in zip(boxes, names, probs):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            text = f"{name}: {prob:.2f}%"
            cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    if writer is None and args['output'] is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args['output'], fourcc, 20, (frame.shape[1], frame.shape[0]), True)
    if writer is not None:
        writer.write(frame)
    # check to see if we are supposed to display the output frame to
    # the screen
    if args['display'] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if q was pressed break from the loop
        if key == ord('q'):
            break

# cleaning up the things :)
cv2.destroyAllWindows()
vs.stop()
if writer is not None:
    writer.release()
