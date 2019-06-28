import face_recognition
import argparse
import pickle
import cv2

# Constructing argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection_method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# loading face embeddings
print("INFO[] loading embeddings")
data = pickle.loads(open(args['encodings'], "rb").read())

# loading and converting images from BGR to RGB
image = cv2.imread(args['image'])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect x and y coordinate for each of the bounding box
# corresponding to each of the faces, then compute
# face embedding for each of the face
boxes = face_recognition.face_locations(rgb, model=args['detection_method'])
encodings = face_recognition.face_encodings(rgb, boxes)

# names of detected faces
names = []
for encoding in encodings:
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

# loop over recognized faces
for ((top, right, bottom, left),name) in zip(boxes, names):
    # draw the predicted face name on the image
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
# show image
cv2.imshow('Image', image)
cv2.waitKey(0)