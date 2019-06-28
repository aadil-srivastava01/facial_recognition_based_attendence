
<h3>Face recognition using Deep Metric Learning</h3>
<p>So, how does deep learning + face recognition work?</p>
<p>The secret is a technique called deep metric learning.

If you have any prior experience with deep learning you know that we typically train a network to:

Accept a single input image
And output a classification/label for that image
However, deep metric learning is different.

Instead, of trying to output a single label (or even the coordinates/bounding box of objects in an image), 
we are instead outputting a real-valued feature vector.

For the dlib facial recognition network, the output feature vector is 128-d (i.e., a list of 128 real-valued numbers) 
that is used to quantify the face. Training the network is done using 
<a href="https://www.pyimagesearch.com/wp-content/uploads/2018/06/face_recognition_opencv_triplet.jpg">Triplets.</a></p>

<p>Our network quantifies the faces, constructing the 128-d embedding (quantification) for each.</p>
<p>Our network architecture for face recognition is based on ResNet-34 from the 
<a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener"><em>Deep Residual Learning for 
Image Recognition</em></a> paper by He et al., but with fewer layers and the number of filters reduced by half.</p>

Install requirement.txt using `pip install -r requirements.txt`

To generate embeddings use: <br>
`python encode_face.py --dataset _path to dataset --encoding __path.pickle`

To recognize faces in an image use: <br>
`python recognize_faces_image.py --encodings _path.pickle_ --image _path to test image_`

To recognize faces in a video stream use: <br>
`python recognize_faces_videos.py --encodings encodings.pickle --output output\webcam_output.avi --display 1`
<br>
Use display value 1 if you want to see the live processed feed
or use 0 if you just want to the processed feed.

If you want a SVC classifier to be trained for using it to perform 
face recognition task use:<br>
`python recognize_faces_videos.py --encodings encodings.pickle --recognizer recognizer.pickle --le le.pickle --output 
output\webcam_output.avi --display 1`

To use a trained SVC classifier use:
`python recognize_faces_videos.py --encodings encodings.pickle --recognizer recognizer.pickle --le le.pickle --output 
output\webcam_output.avi --display 1 --use_embed_only 0` 
