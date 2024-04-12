import os
import numpy as np
from pathlib import Path
import joblib
import pandas as pd
import altair as alt
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
from streamlit_extras.colored_header import colored_header

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set page config, add title and description
st.set_page_config(layout="wide", page_title="FD Vs HCM")

# Set the page layout to reduce the padding
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Add Title to the page
st.title("Fabry Disease (FD) Vs Hypertrophic Cardiomyopathy (HCM)")

# Add a description
st.write('This app differentiates between Fabry and HCM based on various cardiac markers.')

# Load model to streamlit
model_path = Path('./models/model.pkl')
model = joblib.load(model_path)

# Set today's date to ensure all reports are on or before this date
today = datetime.today().date()

# Create a container for the overall layout after title
with st.container():
  # Adjust 
  input_cont, pred_cont = st.columns(2)
  
  with input_cont.container():
    # Patient demographic Data
    with st.expander("Patient Data", expanded=False):
      dem_cols1, dem_cols2, ecg_date_col, echo_date_col, holter_date_col = st.columns(5)
      age = dem_cols1.number_input('Age',min_value=18, max_value=120, step=1, value=40, key='age', help='Enter the patient\'s age.') 
      gender = dem_cols2.selectbox('Gender', options=['Male', 'Female'], key='gender')
      ecg_date = ecg_date_col.date_input('ECG Date', format="DD/MM/YYYY", max_value=today, value="today", key='ecg_date')
      echo_date = echo_date_col.date_input('Echo Date', format="DD/MM/YYYY", max_value=today, value="today", key='echo_date')
      holter_date = holter_date_col.date_input('Holter Date', format="DD/MM/YYYY", max_value=today, value="today", key='holter_date')

    # ECG Report Variables
    with st.expander("ECG Report Data", expanded=False):
        ecg_col1, ecg_col2, ecg_col3, ecg_col4, ecg_col5, ecg_col6, ecg_col7, ecg_col8 = st.columns(8)
        
        with ecg_col1:
            vent_rate = st.number_input('Vent. rate', min_value=30, max_value=200, value=70, step=1, key='vent_rate')
        with ecg_col2:
            qrs_duration = st.number_input('QRS dur.', min_value=70, max_value=200, value=90, step=1, key='qrs_duration')
        with ecg_col3:
            p_axis = st.number_input('P-axis', min_value=-180, max_value=180, value=60, step=1, key='p_axis')
        with ecg_col4:
            r_axis = st.number_input('R-axis', min_value=-180, max_value=180, value=60, step=1, key='r_axis')
        with ecg_col5:
            t_axis = st.number_input('T-axis', min_value=-180, max_value=180, value=60, step=1, key='t_axis')
        with ecg_col6:
            qt = st.number_input('QT', min_value=300, max_value=600, value=350, step=10, key='qt')
        with ecg_col7:
            qtc = st.number_input('QTc', min_value=300, max_value=500, value=420, step=10, key='qtc')
        with ecg_col8:
            bsa = st.number_input('BSA', min_value=1.2, max_value=2.5, value=1.7, step=0.1, key='bsa')

    # Echocardiogram Variables
    with st.expander("Echocardiogram Data", expanded=False):
      echo_col1, echo_col2, echo_col3, echo_col4 = st.columns(4)
      with echo_col1:
        ivsd = st.number_input('IVSd (cm)', min_value=0.6, max_value=1.1, value=0.9, step=0.01)
        lvot_diam = st.number_input('LVOT Diam (cm)', min_value=1.0, max_value=3.5, value=2.0, step=0.01)
        lvids = st.number_input('LVIDs (cm)', min_value=2.0, max_value=4.0, value=3.0, step=0.01)
        la_dimension = st.number_input('LA Dimension (cm)', min_value=2.0, max_value=4.0, value=3.5, step=0.01)
        lvidd = st.number_input('LVIDd (cm)', min_value=3.5, max_value=6.0, value=5.0, step=0.01)
        lvpwd = st.number_input('LVPWd (cm)', min_value=0.6, max_value=1.1, value=0.9, step=0.01)
        ivs = st.number_input('IVS (cm)', min_value=0.6, max_value=1.1, value=1.0, step=0.01)
        ao_root_diam = st.number_input('Ao Root Diam (cm)', min_value=2.0, max_value=3.7, value=3.0, step=0.01)
        fs = st.number_input('FS (%)', min_value=25.0, max_value=40.0, value=30.0, step=0.1)
        edv_teich = st.number_input('EDV (Teichholz) (ml)', min_value=70, max_value=150, value=120, step=1)
        lvld_ap4 = st.number_input('LVLd ap4 (cm)', min_value=3.0, max_value=6.0, value=5.0, step=0.01)
        lvld_ap2 = st.number_input('LVLd ap2 (cm)', min_value=3.0, max_value=6.0, value=5.0, step=0.01)
        edv_mod_sp4 = st.number_input('EDV (Mod Simpson"s) sp4 (ml)', min_value=70, max_value=150, value=120, step=1)
        edv_mod_sp2 = st.number_input('EDV (Mod Simpson"s) sp2 (ml)', min_value=70, max_value=150, value=120, step=1)
        edv_sp4_el = st.number_input('EDV(sp4-el) (ml)', min_value=70, max_value=300, value=120, step=1)
        edv_sp2_el = st.number_input('EDV(sp2-el) (ml)', min_value=70, max_value=300, value=120, step=1)
      with echo_col2:
        lvas_ap4 = st.number_input('LVAs ap4 (cm)', min_value=2.0, max_value=4.0, value=3.0, step=0.01)
        lvas_ap2 = st.number_input('LVAs ap2 (cm)', min_value=2.0, max_value=4.0, value=3.0, step=0.01)
        lvl_ap4 = st.number_input('LVLs ap4 (cm)', min_value=3.5, max_value=7.5, value=5.5, step=0.01)
        lvl_ap2 = st.number_input('LVLs ap2 (cm)', min_value=3.5, max_value=7.5, value=5.5, step=0.01)
        esv_mod_sp4 = st.number_input('ESV (Mod Simpson"s) sp4 (ml)', min_value=20, max_value=70, value=50, step=1)
        esv_mod_sp2 = st.number_input('ESV (Mod Simpson"s) sp2 (ml)', min_value=20, max_value=70, value=50, step=1)
        esv_sp4_el = st.number_input('ESV sp4-el (ml)', min_value=20, max_value=120, value=50, step=1)
        esv_sp2_el = st.number_input('ESV sp2-el (ml)', min_value=20, max_value=120, value=50, step=1)
        ef_mod_sp4 = st.number_input('EF (Mod Simpson"s) sp4 (%)', min_value=53.0, max_value=73.0, value=55.0, step=0.1)
        ef_sp4_el = st.number_input('EF sp4-el (%)', min_value=50.0, max_value=70.0, value=60.0, step=0.1)
        sv_mod_sp4 = st.number_input('SV (Mod Simpson"s) sp4 (ml)', min_value=55, max_value=100, value=70, step=1)
        sv_sp4_el = st.number_input('SV sp4-el (ml)', min_value=50, max_value=130, value=70, step=1)
        ao_root_area = st.number_input('Ao Root Area (cm²)', min_value=3.0, max_value=7.0, value=5.0, step=0.01)
        laa = st.number_input('LAA (cm²)', min_value=3.0, max_value=7.0, value=5.0, step=0.01)
        raa = st.number_input('RAA (cm²)', min_value=3.0, max_value=7.0, value=5.0, step=0.01)
        mapse = st.number_input('MAPSE (cm)', min_value=1.0, max_value=1.5, value=1.2, step=0.01)
      with echo_col3:
        tapse = st.number_input('TAPSE (cm)', min_value=1.5, max_value=2.3, value=1.9, step=0.01)
        mv_e_max_vel = st.number_input('MV E Max Vel (m/s)', min_value=0.6, max_value=1.5, value=1.0, step=0.01)
        mv_a_max_vel = st.number_input('MV A Max Vel (m/s)', min_value=0.6, max_value=1.5, value=1.0, step=0.01)
        mv_e_a = st.number_input('MV E/A Ratio', min_value=0.8, max_value=1.5, value=1.0, step=0.01)
        mv_dec_time = st.number_input('MV dec time (ms)', min_value=150, max_value=300, value=220, step=1)
        lat_peak_e_vel = st.number_input('Lat Peak E" Vel (m/s)', min_value=0.5, max_value=1.0, value=0.7, step=0.01)
        med_peak_e_vel = st.number_input('Med Peak E" Vel (m/s)', min_value=0.5, max_value=1.0, value=0.7, step=0.01)
        ao_v2_max = st.number_input('Ao V2 Max (m/s)', min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        ao_max_pg = st.number_input('Ao Max PG (mmHg)', min_value=5, max_value=20, value=10, step=1)
        lv_v1_max_pg = st.number_input('LV V1 max PG (mmHg)', min_value=5, max_value=25, value=14, step=1)
        lv_v1_max = st.number_input('LV V1 max (m/s)', min_value=0.7, max_value=1.5, value=1.0, step=0.01)
        pa_v2_max = st.number_input('PA V2 Max (m/s)', min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        pa_max_pg = st.number_input('PA Max PG (mmHg)', min_value=5, max_value=20, value=10, step=1)
        tr_max_vel = st.number_input('TR Max Vel (m/s)', min_value=0.5, max_value=1.5, value=1.0, step=0.01)
        tr_max_pg = st.number_input('TR Max PG (mmHg)', min_value=5, max_value=20, value=10, step=1)
        pi_end_d_vel = st.number_input('PI end-d vel (m/s)', min_value=0.6, max_value=1.2, value=0.9, step=0.01)
      with echo_col4:
        e_e_lat = st.number_input('E/E" Lat', min_value=6.0, max_value=15.0, value=8.0, step=0.1)
        e_e_med = st.number_input('E/E" Med', min_value=6.0, max_value=15.0, value=8.0, step=0.1)
        desc_ao_max_vel = st.number_input('Desc Ao Max Vel (m/s)', min_value=0.8, max_value=1.5, value=1.2, step=0.01)
        desc_ao_max_pg = st.number_input('Desc Ao Max PG (mmHg)', min_value=5, max_value=20, value=10, step=1)
        ao_sinus_diam = st.number_input('Ao Sinus Diam (cm)', min_value=2.5, max_value=4.0, value=3.5, step=0.01)
        mv_sax_meas_a = st.number_input('MV SAX Measurements A (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        mv_sax_meas_b = st.number_input('MV SAX Measurements B (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        mv_sax_meas_c = st.number_input('MV SAX Measurements C (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        mv_sax_meas_d = st.number_input('MV SAX Measurements D (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        pm_sax_meas_a = st.number_input('PM SAX Measurements A (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        pm_sax_meas_b = st.number_input('PM SAX Measurements B (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        pm_sax_meas_c = st.number_input('PM SAX Measurements C (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)
        pm_sax_meas_d = st.number_input('PM SAX Measurements D (cm)', min_value=1.5, max_value=2.5, value=2.0, step=0.01)

    # Holter Monitor Data Configuration
    with st.expander("Holter Monitor Data", expanded=False):
      hol_col1, hol_col2, hol_col3 = st.columns(3)
      with hol_col1:
        artefacts = st.number_input('Artefacts', min_value=0, max_value=1, value=0, step=1)
        normal_count = st.number_input('Normal Count', min_value=0, max_value=1000, value=500, step=10)
        normal_percent = st.number_input('Normal Percent', min_value=0, max_value=100, value=99, step=1)
        normal_max_hour = st.number_input('Normal Max/Hour', min_value=0, max_value=100, value=50, step=1)
        ve_beats_count = st.number_input('VE Beats Count', min_value=0, max_value=1000, value=5, step=10)
        ve_beats_percent = st.number_input('VE Beats Percent', min_value=0, max_value=100, value=1, step=1)
        ve_beats_max_hour = st.number_input('VE Beats Max/Hour', min_value=0, max_value=100, value=5, step=1)
        sve_beats_count = st.number_input('SVE Beats Count', min_value=0, max_value=1000, value=5, step=10)
        sve_beats_percent = st.number_input('SVE Beats Percent', min_value=0, max_value=100, value=1, step=1)
        sve_beats_max_hour = st.number_input('SVE Beats Max/Hour', min_value=0, max_value=100, value=5, step=1)
        paced_beats_count = st.number_input('Paced Beats Count', min_value=0, max_value=1000, value=0, step=10)
        paced_beats_percent = st.number_input('Paced Beats Percent', min_value=0, max_value=100, value=0, step=1)
        paced_beats_max_hour = st.number_input('Paced Beats Max/Hour', min_value=0, max_value=100, value=0, step=1)
        heart_rates_max_hr = st.number_input('Heart Rates (1 min avg) Max HR', min_value=0, max_value=300, value=120, step=1)
      with hol_col2:
        heart_rates_mean_hr = st.number_input('Heart Rates (1 min avg) Mean HR', min_value=0, max_value=300, value=80, step=1)
        heart_rates_min_hr = st.number_input('Heart Rates (1 min avg) Min HR', min_value=0, max_value=300, value=60, step=1)
        bradycardia = st.number_input('Bradycardia', min_value=0, max_value=10000, value=0, step=10)
        bradycardia_event_longest = st.number_input('Bradycardia Event Longest', min_value=0, max_value=1000, value=0, step=1)
        bradycardia_event_min_rate = st.number_input('Bradycardia Event Min Rate', min_value=0, max_value=300, value=40, step=1)
        pause = st.number_input('Pause', min_value=0, max_value=1000, value=0, step=1)
        broad_complex_tachycardia = st.number_input('Broad Complex Tachycardia', min_value=0, max_value=1000, value=0, step=10)
        broad_complex_tachycardia_longest = st.number_input('Broad Complex Tachycardia Longest', min_value=0, max_value=1000, value=0, step=1)
        broad_complex_tachycardia_max_rate = st.number_input('Broad Complex Tachycardia Max Rate', min_value=0, max_value=300, value=150, step=1)
        v_run_aivr = st.number_input('V-Run/AIVR', min_value=0, max_value=1000, value=0, step=10)
        v_run_aivr_longest = st.number_input('V-Run/AIVR Longest', min_value=0, max_value=1000, value=0, step=1)
        v_run_aivr_max_rate = st.number_input('V-Run/AIVR Max Rate', min_value=0, max_value=300, value=150, step=1)
        couplet = st.number_input('Couplet', min_value=0, max_value=1000, value=0, step=10)
        triplet = st.number_input('Triplet', min_value=0, max_value=1000, value=0, step=10)
      with hol_col3:
        single_ve_events = st.number_input('Single VE Events', min_value=0, max_value=1000, value=0, step=10)
        svt = st.number_input('SVT', min_value=0, max_value=1000, value=0, step=10)
        svt_longest = st.number_input('SVT Longest', min_value=0, max_value=1000, value=0, step=1)
        svt_max_rate = st.number_input('SVT Max Rate', min_value=0, max_value=300, value=150, step=1)
        sve = st.number_input('SVE', min_value=0, max_value=1000, value=0, step=10)
        sve_max_per_minute = st.number_input('SVE Max per Minute', min_value=0, max_value=100, value=5, step=1)
        sve_max_per_hour = st.number_input('SVE Max per Hour', min_value=0, max_value=100, value=30, step=1)
        sve_mean_per_hour = st.number_input('SVE Mean per Hour', min_value=0, max_value=100, value=10, step=1)
        sve_run = st.number_input('SVE Run', min_value=0, max_value=1000, value=0, step=10)
        sve_run_longest = st.number_input('SVE Run Longest', min_value=0, max_value=1000, value=0, step=1)
        sve_run_max_rate = st.number_input('SVE Run Max Rate', min_value=0, max_value=300, value=150, step=1)

# Make a prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [1 if gender == 'Male' else 2],
    'Vent. rate': [vent_rate], 
    'QRS duration': [qrs_duration],
    'P-axis': [p_axis],
    'R-axis': [r_axis],
    'T-axis': [t_axis],
    'QT': [qt],
    'QTc': [qtc],
    'BSA': [bsa],
    'IVSd': [ivsd],
    'LVOT diam': [lvot_diam],
    'LVIDs': [lvids],
    'LA dimension': [la_dimension],
    'LVIDd': [lvidd],
    'LVPWd': [lvpwd],
    'IVS': [ivs],
    'Ao root diam': [ao_root_diam],
    'FS': [fs], 
    'EDV(Teich)': [edv_teich],
    'LVLd ap4': [lvld_ap4], 
    'LVLd ap2': [lvld_ap2], 
    'EDV(MOD-sp4)': [edv_mod_sp4], 
    'EDV(MOD-sp2)': [edv_mod_sp2], 
    'EDV(sp4-el)': [edv_sp4_el], 
    'EDV(sp2-el)': [edv_sp2_el], 
    'LVAs ap4': [lvas_ap4], 
    'LVAs ap2': [lvas_ap2], 
    'LVLs ap4': [lvl_ap4], 
    'LVLs ap2': [lvl_ap2], 
    'ESV(MOD-sp4)': [esv_mod_sp4], 
    'ESV(MOD-sp2)': [esv_mod_sp2], 
    'ESV(sp4-el)': [esv_sp4_el], 
    'ESV(sp2-el)': [esv_sp2_el], 
    'EF(MOD-sp4)': [ef_mod_sp4], 
    'EF(sp4-el)': [ef_sp4_el], 
    'SV(MOD-sp4)': [sv_mod_sp4], 
    'SV(sp4-el)': [sv_sp4_el], 
    'Ao root area': [ao_root_area], 
    'LAA': [laa], 
    'RAA': [raa], 
    'MAPSE': [mapse], 
    'TAPSE': [tapse], 
    'MV E max vel': [mv_e_max_vel], 
    'MV A max vel': [mv_a_max_vel], 
    'MV E/A': [mv_e_a], 
    'MV dec time': [mv_dec_time], 
    'Lat Peak E" Vel': [lat_peak_e_vel], 
    'Med Peak E" Vel': [med_peak_e_vel], 
    'Ao V2 max': [ao_v2_max], 
    'Ao max PG': [ao_max_pg], 
    'LV V1 max PG': [lv_v1_max_pg], 
    'LV V1 max': [lv_v1_max], 
    'PA V2 max': [pa_v2_max], 
    'PA max PG': [pa_max_pg],
    'TR max vel': [tr_max_vel],
    'TR max PG': [tr_max_pg],
    'PI end-d vel': [pi_end_d_vel],
    'E/E" Lat': [e_e_lat],
    'E/E" Med': [e_e_med],
    'desc Ao max vel': [desc_ao_max_vel], 
    'desc Ao max PG': [desc_ao_max_pg], 
    'Ao sinus diam': [ao_sinus_diam], 
    'MV SAX Measurements A': [mv_sax_meas_a], 
    'MV SAX Measurements B': [mv_sax_meas_b], 
    'MV SAX Measurements C': [mv_sax_meas_c], 
    'MV SAX Measurements D': [mv_sax_meas_d], 
    'PM SAX Measurements A': [pm_sax_meas_a], 
    'PM SAX Measurements B': [pm_sax_meas_b], 
    'PM SAX Measurements C': [pm_sax_meas_c], 
    'PM SAX Measurements D': [pm_sax_meas_d], 
    'Artefacts': [artefacts],
    'Normal Count': [normal_count],
    'Normal Percent': [normal_percent],
    'Normal Max/Hour': [normal_max_hour],
    'VE Beats Count': [ve_beats_count],
    'VE Beats Percent': [ve_beats_percent],
    'VE Beats Max/Hour': [ve_beats_max_hour],
    'SVE Beats Count': [sve_beats_count],
    'SVE Beats Percent': [sve_beats_percent],
    'SVE Beats Max/Hour': [sve_beats_max_hour],
    'Paced Beats Count': [paced_beats_count],
    'Paced Beats Percent': [paced_beats_percent],
    'Paced Beats Max/Hour': [paced_beats_max_hour],
    'Heart Rates (1 min avg) Max HR': [heart_rates_max_hr],
    'Heart Rates (1 min avg) Mean HR': [heart_rates_mean_hr],
    'Heart Rates (1 min avg) Min HR': [heart_rates_min_hr],
    'Bradycardia': [bradycardia],
    'Bradycardia Event Longest': [bradycardia_event_longest], 
    'Bradycardia Event Min Rate': [bradycardia_event_min_rate], 
    'Pause': [pause], 
    'Broad Complex Tachycardia': [broad_complex_tachycardia], 
    'Broad Complex Tachycardia Longest': [broad_complex_tachycardia_longest], 
    'Broad Complex Tachycardia Max Rate': [broad_complex_tachycardia_max_rate], 
    'V-Run/AIVR': [v_run_aivr], 
    'V-Run/AIVR Longest': [v_run_aivr_longest], 
    'V-Run/AIVR Max Rate': [v_run_aivr_max_rate], 
    'Couplet': [couplet], 
    'Triplet': [triplet], 
    'Single VE Events': [single_ve_events], 
    'SVT': [svt], 
    'SVT Longest': [svt_longest], 
    'SVT Max Rate': [svt_max_rate], 
    'SVE': [sve], 
    'SVE Max per Minute': [sve_max_per_minute], 
    'SVE Max per Hour': [sve_max_per_hour], 
    'SVE Mean per Hour': [sve_mean_per_hour], 
    'SVE Run': [sve_run], 
    'SVE Run Longest': [sve_run_longest], 
    'SVE Run Max Rate': [sve_run_max_rate],
    'Holter_date_diff': [(holter_date - ecg_date).days],
    'Echo_date_diff': [(echo_date - ecg_date).days],
    })

# Create a container for the prediction and additional interpretability
with pred_cont.container():

  with st.container(border=False, height=60):
    # Make a prediction
    prediction = model.predict_proba(input_data).flatten()
    
    # Create a DataFrame for the chart
    data = pd.DataFrame({
        'Condition': ['HCM', 'FD'],
        'Probability': prediction,
        'Cond_Position': [0.05, 0.95],  # Custom positions for labels
        'Pred_Position': [prediction[0] - 0.03, prediction[0] + 0.03],  # Positions adjusted based on the prediction
        'Sort': [0, 1]  # Ensure the order of bars
    })

    # Base chart for the single bar
    base = alt.Chart(data).mark_bar().encode(
        x=alt.X('sum(Probability):Q', stack='zero', axis=None),  # No axis for a cleaner look
        color=alt.Color('Condition:N', legend=None, scale=alt.Scale(domain=['HCM', 'FD'], range=['#ae2514', '#0467a5'])),
        order='Sort'  # Sorting order for conditions
    ).properties(
        height=60, # Fixed height to reduce vertical space
    )
    
    # Text annotations for condition names
    text_desc = alt.Chart(data).mark_text(
        align='center',
        baseline='middle',
        color='white',
        fontSize=15
    ).encode(
        x='Cond_Position:Q',
        text='Condition:N'
    )

    # Text annotations for probabilities
    text_probs = alt.Chart(data).mark_text(
        align='center',
        baseline='middle',
        color='white',
        fontSize=15
    ).encode(
        x='Pred_Position:Q',
        text=alt.Text('Probability:N', format='.2f')
    )

    # Combine the charts
    chart = alt.layer(base, text_desc, text_probs).configure_view(
        strokeWidth=0  # Remove border around the chart
    )
    
    # Display the chart
    st.altair_chart(chart, use_container_width=True)

  with st.expander("Additional Interpretability", expanded=False):
    # Create a SHAP Explainer object
    shap_values = model.get_booster().predict(xgb.DMatrix(input_data), pred_contribs=True)[:,:-1]
    
    # Create a DataFrame for the SHAP values
    shap_values = pd.DataFrame(shap_values, columns=input_data.columns)
    
    # Create a bar chart for top 10 features by mean absolute SHAP value using altair
    shap_values_mean = shap_values.abs().mean().sort_values(ascending=False).head(10)
    shap_values_mean = shap_values_mean.reset_index()
    shap_values_mean.columns = ['Feature', 'Mean Absolute SHAP Value']
    st.altair_chart(alt.Chart(shap_values_mean).mark_bar().encode(
        x='Mean Absolute SHAP Value:Q',
        y=alt.Y('Feature:N', sort='-x')
    ), use_container_width=True)
    
    # Display the SHAP values as an altair bar chart removing all the features with 0 SHAP values
    shap_values_sum = shap_values.sum().sort_values()
    shap_values_sum = shap_values_sum[shap_values_sum != 0]
    shap_values_sum = shap_values_sum.reset_index()
    shap_values_sum.columns = ['Feature', 'SHAP Value']
    st.altair_chart(alt.Chart(shap_values_sum).mark_bar().encode(
        x='SHAP Value:Q',
        y=alt.Y('Feature:N', sort='-x')
    ), use_container_width=True)

# Create a line between the containers and the chatbot
colored_header(label='', description='', color_name='blue-70')

# Chatbot
st.title('🤗💬 Ask Away!')

# Initialize or get the current session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Initialize the LLM model
llm = HuggingFaceEndpoint(
        repo_id=st.secrets["HUGGINGFACE_REPO_ID"],
        task="text-generation",
        max_new_tokens=1024,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        )

# Initialize the chatbot by asking the user's name
st.chat_message("assistant").markdown("Hello! I'm here to help you find the right prognosis. \
                                      Please press the 'Analyse Data' after entering the patient's cardiac data.")

# Load sample patients
if st.button("Sample Fabry Patient"):
  with st.spinner("Loading sample data..."):
    sample_data = pd.read_csv('data/sample_data.csv')
    
    # Change all input fields to the sample patient's data
    for col in sample_data.columns:
        st.session_state[col] = sample_data[col].values[0]
    
    st.experimental_rerun()

if st.button("Sample HCM Patient"):
  with st.spinner("Loading sample data..."):
    # Load sample data
    sample_data = pd.read_csv('data/sample_data.csv')
    sample_data

# Load model and prepare data after the user clicks the button
if st.button("Analyse Data"):
  with st.spinner("Analysing data..."):
    # Clear previous messages
    st.session_state['messages'] = []
    
    # Make a prediction
    predicted_condition = {0: 'Hypertrophic Cardiomyopathy', 1: 'Fabry Disease'}[prediction.argmax()]
    feature_values = input_data.iloc[0].to_dict()
    features_info = ', '.join([f"{feature}: {feature_values[feature]} ({shap_value:.2f})" 
                                for feature, shap_value in zip(shap_values_sum['Feature'], shap_values_sum['SHAP Value'])])
    initial_prompt = f"Based on the input data, the model predicts a higher likelihood of {predicted_condition}. " \
                      f"The key factors influencing this prediction include: {features_info}."
    
    # Set the template for the model instructions
    template = """
    In your role as a cardiologist in a secondary care setting, evaluate the provided comprehensive dataset for a patient referred with potential Hypertrophic Cardiomyopathy (HCM) or Fabry disease. The dataset includes demographic details, ECG, echocardiography (echo), and Holter monitor report values. Guide your analysis with the following considerations:

    1. Assess the integration of the patient's demographic information with findings from ECG, echo, and Holter reports, specifically looking for indicators or patterns that may suggest HCM or Fabry disease.
    2. Given the referral to secondary care, recognize that the patient has undergone extensive initial testing. Look for both confirmatory and contradictory evidence of HCM or Fabry disease in comparison to earlier assessments.
    3. Interpret 'NaN' values as unmeasured parameters or non-occurring events, and note that gender is coded as '1' for male and '2' for female, which may have implications for the diagnosis due to the X-linked inheritance pattern of Fabry disease.
    4. Employ evidence-based guidelines and differential diagnosis strategies to distinguish between HCM and Fabry disease, focusing on the distinguishing features of each condition as revealed by the diagnostic tests.
    5. Based on the XGBoost model's prediction and the SHAP values for each feature, identify key data points that support the diagnosis of either HCM or Fabry disease.
    6. Formulate recommendations for any further diagnostic tests that may be necessary, drawing on your analysis of the patient's specific condition.

    Your goal is to utilize both traditional diagnostic methods and modern data analysis techniques to differentiate between HCM and Fabry disease, providing a detailed and informed diagnostic perspective for this patient.

    Patient history: {patient_history}
    """

    # Initialize the LLMChain with the model and template
    model_instructions = PromptTemplate.from_template(template)
    llm_chain = LLMChain(llm=llm, prompt=model_instructions)  
                          
    response = llm_chain.invoke(initial_prompt)
    
    # Setup initial chat messages
    st.session_state.messages.append(ChatMessage(role="user", content=response.content))
  
# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg.role).markdown(msg.content)

# Chat input
if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").markdown(prompt)
    response = llm(st.session_state.messages)
    st.chat_message("assistant").markdown(response["text"])
    st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
    
# Allow users to clear chat history
if st.button("Clear chat history"):
    st.session_state['messages'] = []
    st.experimental_rerun()
