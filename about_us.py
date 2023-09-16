import streamlit as st
st.set_page_config(page_title="About us")
def main():
    st.title("About Us")
    st.write("The GestionRH application is a CRUD (Create, Read, Update, Delete) application designed for managing human resources data within an organization. It provides a user-friendly interface built with Streamlit and utilizes a SQLite database for data storage. The application allows users to perform various operations on different tables such as ProcesVerbal, AgendaDept, Departement, Employe, Absent, ActivitesDept, Activites, and Alertes.")
if __name__ == "__main__":
    main()
