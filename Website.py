import streamlit as st
import cv2
import numpy as np
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import joblib
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import skew


# Loading Models
CNN_model = load_model('qr_code_CNN_model.h5')
BinaryLSTM_model = load_model('qr_code_BinaryLSTM_model.h5')
DNN_model=load_model('qr_code_dnn_model.h5')
lightGBM_model=lgb.Booster(model_file='mode.txt')
# lightGBM_model=joblib.load('lgb.pkl')
SVM_model = joblib.load('qr_code_svm_model.joblib')
XgBoost_model=xgb.Booster(model_file='qr_code_xgboost_model.json') 

# DNN Model
# Function to preprocess a new QR code image
def preprocess_image_dnn(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = np.array(img) / 255.0
    img = np.reshape(img, (1, 128, 128))  # Add an extra dimension for batch_size
    return img

# Function to predict the class of a new QR code
def predict_qr_code_dnn(img_path):
    dnn_accuracy=0.4725
    img = preprocess_image_dnn(img_path)
    prediction = DNN_model.predict(img)
    # Assuming the model has a sigmoid activation in the output layer for binary classification
    predicted_class = 1 if prediction[0][0] > 0.5 else 0
    if predicted_class == 0:
        return "Non-Malicious"
    else:
        return "Malicious"

# CNN Model
def classify_qr_code_CNN(image_path):
    cnn_accuracy="98%"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    classes = CNN_model.predict(image)
    # Assuming you have two classes: Malicious and Non-Malicious
    if classes[0][0]<=0.5:
        return "Malicious"
    else:
        return "Non-Malicious"
    
# Binary LSTM
def classify_qr_code(qr_code_image_path):
    lstm_accuracy=0.96
    qr_code_image = cv2.imread(qr_code_image_path)
    gray_image = cv2.cvtColor(qr_code_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (128, 128))
    qr_code_data = np.expand_dims(resized_image / 255.0, axis=0)
    prediction = BinaryLSTM_model.predict(qr_code_data)
    label = np.argmax(prediction)
    if label == 0:
        return "Malicious"
    else:
        return "Non-Malicious"
    

# Light GBM
def extract_features_gbm(image):
    features = []
    features.append(np.mean(image))
    features.append(np.std(image))
    features.append(skew(image.flatten()))
    return features
 
def lightGBM(path):
    new_qr_code_path = path
    img = cv2.imread(new_qr_code_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    new_features = extract_features_gbm(img)
    prediction = lightGBM_model.predict(np.array([new_features]))
    predicted_class = 1 if prediction[0] > 0.5 else 0
    # Print the prediction
    if predicted_class == 0:
        return "Non-Malicious"
    else:
        return "Malicious"
    
# XgBoost MODEL
# XgBoost Function to preprocess a new QR code image    
def preprocess_image_xgboost(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img_features = [np.mean(img), np.std(img), skew(img.flatten())]
    return np.array(img_features).reshape(1, -1)

# XgBoost Function to predict the class of a new QR code
def predict_qr_code_xgboost(img_path):
    img_features = preprocess_image_xgboost(img_path)
    dtest = xgb.DMatrix(img_features)
    prediction = XgBoost_model.predict(dtest)
    if prediction[0] != 0:
        return "Non-Malicious"
    else:
        return "Malicious"

# SVM MODEL
# SVM Feature Extract
def extract_features_svm(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    skewness = skew(image.flatten())
    return [mean, std_dev, skewness]
# SVM Model CALL
def svm_call(path):
    img_path=path
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = extract_features_svm(img)
    img = np.reshape(img, (1, -1))  # Reshape to match the input shape of the SVM model
    prediction = SVM_model.predict(img)
    if prediction == 0:
        return "Non-Malicious"
    else:
        return "Malicious"


# TO SAVE IMAGE   
def save_uploadedfile(uploadedfile):
    suff=(uploadedfile.name.split('.')[1])
    with open(os.path.join("QR_DIr",'QR_code.'+suff),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return 'QR_code.'+suff


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    name=save_uploadedfile(uploaded_file)
    PATH=f"QR_DIr\{name}"
    st.image(uploaded_file, caption="Uploaded Image",width=200)
    st.write(PATH)

    lstm_result=classify_qr_code(PATH)
    dnn_result=predict_qr_code_dnn(PATH)
    cnn_rseult=classify_qr_code_CNN(PATH)
    svm_result=svm_call(PATH)
    lightgbm_result=lightGBM(PATH)
    xgboost_result=predict_qr_code_xgboost(PATH)
    
    

    
    # output={
    #     "Binary LSTM":lstm_result,
    #     "DNN":dnn_result,
    #     "CNN":cnn_rseult,
    #     "SVM":svm_result,
    #     "Light GBM":lightgbm_result,
    #     "Xg Boost":xgboost_result
    # }
    
    # st.write("LSTM: "+lstm_result)
    # st.write("DNN: "+dnn_result)
    # st.write("CNN: "+cnn_rseult)
    # st.write("SVM: "+svm_result)
    # st.write("LightGBM: "+lightgbm_result)
    # st.write("XgBoost: "+xgboost_result)
    
    report={'Model':["Binary LSTM","DNN","CNN","SVM","Light GBM","Xg Boost"],
        "Result":[lstm_result,dnn_result,cnn_rseult,svm_result,lightgbm_result,xgboost_result],
        "Accuracy":["96%","47.25%","99.87%","64.62%","66.12%","66.62%"],
        "F1-Score":["0.9571","0.6417","0.9988","0.4863","0.6251","0.5898"],
        }
    report=pd.DataFrame(report)

    def color_output(val):
        color = 'green' if val=="Non-Malicious" else 'red'
        return f'background-color: {color}'

    # st.dataframe(df.style.apply(highlight_survived, axis=1))
    st.dataframe(report.style.applymap(color_output, subset=['Result']),width=700,height=400)
    # df = pd.DataFrame(report)
    # st.table(df)
    
    
    
    

    



    
    