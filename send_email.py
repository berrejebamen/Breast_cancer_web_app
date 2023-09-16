import streamlit as st
import time
import smtplib
import email
from PIL import Image
#from email.mime.text import MIMEText
#from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
im = Image.open('app.jpg')
st.set_page_config(
    page_title="CONTACT A DOCTOR",
    layout="wide",
    page_icon = im,
)
st.title('Send Email')
email = st.text_input('Your Email Address')
email1 = st.text_input('The doctor Email Address')
subject = st.text_input('Subject')
body = st.text_area('Body')
if st.button('Send Email'):
    if email1 and email and subject and body:
        with st.spinner('Sending email...'):
            #send_emailo(email, subject, body)
            time.sleep(2)  # simulate email sending delay  
        
            st.success('Email sent successfully!')
    else:
            st.warning('Please fill in all fields.')