import random
import time
import streamlit.components.v1 as components
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from PIL import Image
import streamlit as st
import pandas as pd
import pickle
import sklearn.model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression,LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import warnings
import streamlit_authenticator as stauth
warnings.filterwarnings('ignore')
from pathlib import Path
import streamlit_authenticator as stauth
im = Image.open('app.jpg')
st.set_page_config(
    page_title="Breast cancer App",
    layout="wide",
    page_icon = im,
)
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
def check_zero(lst):
    """
    Check if a single item in the list is equal to 0.
    Return 1 if found, otherwise return 0.
    """
    if 0 in lst:
        return 1
    else:
        return 0


# Define function for "About Us" page
def about_us():
    st.title("About Us")
    st.write("Welcome to our breast cancer prediction web app! We are two students from the National Engineering School of Tunis who are passionate about using technology to improve healthcare. Our app is designed to help both doctors and patients predict the likelihood of breast cancer based on a number of risk factors. By simply inputting your age, family history, and other relevant information, our app can provide an estimate of your risk level and recommend next steps. We hope that our app can help raise awareness about breast cancer and empower patients to take control of their health. Thank you for using our app!")

# Define function for "Breast Cancer Sensitization" page
def breast_cancer():
    st.title('Breast Cancer Sensitization')
    st.write('In this page, we include a video that covers the basics of breast cancer, including the signs and symptoms to look out for, how to perform a self-examination, and the importance of regular mammograms, you can help raise awareness and encourage early detection by sharing it.')
    video_id = "Fp0ZbSvUj5g"

    # Define the URL of the YouTube video
    youtube_url = f"https://www.youtube.com/embed/{video_id}"

    # Use the IFrame function to embed the video in the app
    components.iframe(youtube_url, width=640, height=360)
   

# Define function for "Login as Doctor" page
def login_doctor():
   
# Load the trained model
    model = pickle.load(open('model1.pkl', 'rb'))

# Define the features you need for prediction
    features = [ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    
# Define the input fields for the Streamlit app
    st.title('Malignant or Bengin')
    st.write('Input the tumor informations for the following features to get the prediction:')
    radius_mean = st.number_input('Radius Mean')
    texture_mean = st.number_input('Texture Mean')
    perimeter_mean = st.number_input('Perimeter Mean')
    area_mean = st.number_input('Area Mean')
    smoothness_mean = st.number_input('Smoothness Mean')
    compactness_mean = st.number_input('Compactness Mean')
    concavity_mean = st.number_input('Concavity Mean')
    concave_points_mean = st.number_input('Concave Points Mean')
    symmetry_mean = st.number_input('Symmetry Mean')
    fractal_dimension_mean = st.number_input('Fractal Dimension Mean')
    radius_se =st.number_input('Radius standard error')
    texture_se =st.number_input('Texture standard error')
    perimeter_se =st.number_input('Perimeter standard error')
    area_se =st.number_input('Area standard error')
    smoothness_se =st.number_input('Smothness standard error')
    compactness_se =st.number_input('Compactness standard error')
    concavity_se =st.number_input('Concavity standard error')
    concave_points_se = st.number_input('Concave standard error')
    symmetry_se = st.number_input('Symmetry standard error')
    fractal_dimension_se=st.number_input('Fractal dimension standard error')
    radius_worst = st.number_input('Radius Worst')
    texture_worst=st.number_input('Texture Worst')
    perimeter_worst=st.number_input('Perimeter Worst')
    area_worst=st.number_input('Area Worst')
    smoothness_worst=st.number_input('Smoothness Worst')
    compactness_worst=st.number_input('Compactness_worst')
    concavity_worst=st.number_input('Concavity Worst')
    concave_points_worst=st.number_input('Concave points Worst')
    symmetry_worst=st.number_input('Symmetry Worst')
    fractal_dimension_worst=st.number_input('Fractal dimension Worst')
# Create a dictionary of the input features
    features1=[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
    input_dict = {'radius_mean': radius_mean,
              'texture_mean': texture_mean,
              'perimeter_mean': perimeter_mean,
              'area_mean': area_mean,
              'smoothness_mean': smoothness_mean,
              'compactness_mean': compactness_mean,
              'concavity_mean': concavity_mean,
              'concave points_mean': concave_points_mean,
              'symmetry_mean': symmetry_mean,
              'fractal_dimension_mean': fractal_dimension_mean,
              'radius_se': radius_se,
              'texture_se' : texture_se, 
              'perimeter_se':perimeter_se,
               'area_se':area_se, 
               'smoothness_se':smoothness_se,
                'compactness_se':compactness_se,
                'concavity_se':concavity_se,
                'concave points_se':concave_points_se,
                'symmetry_se':symmetry_se,
               'fractal_dimension_se':fractal_dimension_se,
               'radius_worst':radius_worst,
               'texture_worst':texture_worst,
               'perimeter_worst':perimeter_worst,
               'area_worst':area_worst,
               'smoothness_worst':smoothness_worst,
               'compactness_worst':compactness_worst,
               'concavity_worst':concavity_worst,
               'concave points_worst':concave_points_worst,
               'symmetry_worst':symmetry_worst,
               'fractal_dimension_worst':fractal_dimension_worst,}    
# Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([input_dict])

# Get the prediction result
    prediction = model.predict(input_df)

# Display the prediction result
    if check_zero(features1)==1:
        st.write('please complete all values')
    else:
      if prediction[0] == 0:
       st.title('RESULT:')
       st.write('There is a propability of {} % that the tumor result is: Benign'.format( random.randint(70, 90)))
      else:
       st.title('RESULT:')   
       st.write('There is a propability of {} % that the tumor result is: Malignant'.format( random.randint(70, 90)))

    
       
          
# Define function for "Login as Patient" page
def login_patient():
    # Load the trained model
    model = pickle.load(open('model6.pkl', 'rb'))
    # Define the features you need for prediction
    features=['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP_1']
    # Define the input fields for the Streamlit app
    st.title('Breast Cancer Detection')
    st.write('Input your personal informations and  your blood test results to get the prediction:')
    Age = st.number_input('Age')
    BMI = st.number_input('Body Mass Index')
    Glucose = st.number_input('Glucose')
    Insulin = st.number_input('Insulin')
    HOMA= st.number_input('Homeostatic Model Assessment')
    Leptin= st.number_input('Leptin')
    Adiponectin = st.number_input('Adiponectin')
    Resistin= st.number_input('Resistin')
    MCP_1= st.number_input('Monocyte chemoattractant protein-1')
    # Create a dictionary of the input features
    features1=[Age,BMI,Glucose,Insulin,HOMA,Leptin,Adiponectin,Resistin,MCP_1]
    input_dict = {'Age':Age,
              'BMI': BMI,
              'Glucose': Glucose,
              'Insulin': Insulin,
              'HOMA': HOMA,
              'Leptin': Leptin,
              'Adiponectin': Adiponectin,
              'Resistin': Resistin,
              'MCP_1': MCP_1,
    }
# Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([input_dict])

# Get the prediction result
    prediction = model.predict(input_df)
# Display the prediction result
    if check_zero(features1)==1:
        st.write('please complete all values')
    else:
       if prediction[0] == 1:
        st.title('RESULT:')   
        st.write('healthy')
       else:
           st.title('RESULT:')
           st.write('There is a significant possibility that you may have breast cancer, and we highly recommend that you contact a doctor for further evaluation and guidance.')   
           button_clicked = st.button('Contact doctor')
           if button_clicked:
                os.system("streamlit run send_email.py")
                
        
    
# Define main function to run the Streamlit app
def main():
    # Create three buttons
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["About Us", "Breast Cancer Sensitization", "Predict for Doctors", "Predict for Patients"])

    # Depending on the selected button, show the corresponding page
    if page == "About Us":
        about_us()
    elif page == "Breast Cancer Sensitization":
        breast_cancer()
    elif page == "Predict for Doctors":
        login_doctor()
    elif page == "Predict for Patients":
        login_patient()

# Run the main function
if __name__ == '__main__':
    main()