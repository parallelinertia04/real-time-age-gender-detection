#Project description
Lightweight demo that detects faces from a webcam stream and predicts age range and gender using OpenCV's DNN module with pretrained models (Caffe/TensorFlow). The script captures frames, detects faces with a DNN-based face detector, crops each face, runs age and gender networks, and overlays labels on the video.


#How it works

Face detection: uses a DNN face detector (opencv_face_detector.pb / .pbtxt) to produce bounding boxes.
Face crop: each detected box is cropped from the frame (an optional padding value is available).
Age & gender: cropped faces are passed to Caffe-based age and gender networks (age_net.caffemodel, gender_net.caffemodel) to produce predictions.
Output: live window showing bounding boxes and labels; raw detection output is appended to detection_matrix.txt.


#Inputs and outputs

Input: webcam (cv2.VideoCapture(0)) and the pretrained model files in the project root.
Output: an on-screen video window (Age-Gender) with labels and detection logs saved to detection_matrix.txt.


#Requirements

Python 3.7+
OpenCV (pip install opencv-python)


#Quick usage

Place the pretrained model files in the project folder.
Install dependencies: pip install opencv-python
Run: python main.py
Press 'q' to quit.
