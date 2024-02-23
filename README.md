Thesis Project:- Categorizing QR Code Images as Malicious or Benign Using Machine and Deep learning models.

** Overview
This project aims to detect whether a QR code image is malicious or benign using machine learning and deep learning models. QR codes are widely used to encode and transmit information efficiently. 
However, attackers can exploit QR code vulnerabilities to distribute malware, conduct phishing attacks, and steal user data. This project analyzes QR code images to identify potential threats.

** The key objectives are:

- Build machine learning models like XGBoost, SVM, LightGBM to extract features and identify patterns in QR code images.
- Implement deep learning models like CNN, DNN, Binary LSTM to recognize visual characteristics and complex relationships.
- Evaluate model accuracy through rigorous manual testing with real QR code images.
- Develop an easy-to-use web interface for users to test QR code security.
- Analyze model limitations and enhance QR code detection techniques.

** Installation
The project requires Python 3.x and common data science libraries like Pandas, NumPy, Scikit-Learn, Keras, OpenCV etc. The dependencies are listed in requirements.txt.

** To install:

//Copy code
pip install -r requirements.txt
Run streamlit run app.py to launch the web application for QR code detection. Upload a QR code image and view malicious/benign predictions.

** Results
The BinaryLSTM and CNN models demonstrated exceptional performance in distinguishing between malicious and benign QR codes. However, rigorous manual testing revealed limitations in model accuracy compared to initial scores. 
This highlights the need for continuously evaluating model predictions in real-world scenarios.

** Future Work
- Expand the image dataset to include more diverse and complex malicious QR code examples.
- Evaluate advanced computer vision models like YOLO for object detection capabilities.
- Test model robustness against purposefully adversarial QR codes intended to evade detection.
- Optimize interface and architecture for mobile QR code scanning. 
