from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# constructing argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
                help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
                help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="path to output label encoder")
args = vars(ap.parse_args())

# loading embeddings
try:
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(args['embeddings'], 'rb').read())
    # encoding the labels
    le = LabelEncoder()
    labels = le.fit_transform(data['name'])
    # training the model using 128-d facial embeddings
    print('[INFO] training the model...')
    recognizer = SVC(C=1.0, kernel='linear', probability=True)
    recognizer.fit(data['encodings'], labels)
    # writing the trained model to disk
    f = open(args['recognizer'],'wb')
    f.write(pickle.dumps(recognizer))
    f.close()
    # writing the label encoder to the disk
    f = open(args['le'], 'wb')
    f.write(pickle.dumps(le))
    f.close()
    print("[INFO] model trained successfully")
except:
    print("[INFO] model could not be trained..error occurred..")