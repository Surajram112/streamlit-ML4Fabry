from pathlib import Path
import joblib
import pandas as pd
import altair
import streamlit as st
import xgboost
import matplotlib.pyplot as plt

# Set page config to wide
st.set_page_config(layout="wide")

# Set the app title
st.title('Fabry Disease Vs Hypertrophic Cardiomyopathy Prediction')

# Load model to col1reamlit
model_path = Path('./models/model.pkl')
model = joblib.load(model_path)

# Add a description
st.write('This app predicts the differentiates between Fabry and HCM based on various cardiac markers.')

cont1 = st.container()
with cont1:
  cont1_col1, cont1_col2, cont1_col3 = st.columns([1,2,1])  # Adjusted column widths for a better layout of first set of expanders

  # Patient demographic Data
  with cont1_col1.expander("Patient Demographic Data", expanded=True):
      dem_col1, dem_col2 = st.columns(2)
      age = dem_col1.number_input('Age', min_value=18, max_value=120, value=25, key='age')
      gender = dem_col2.selectbox('Gender', options=['Male', 'Female'], key='gender')

  # ECG Report Variables
  with cont1_col2.expander("ECG Report Data", expanded=True):
      ecg_col1, ecg_col2, ecg_col3, ecg_col4, ecg_col5, ecg_col6, ecg_col7, ecg_col8 = st.columns(8)
      with ecg_col1:
        vent_rate = st.number_input('Vent. rate', min_value=0, max_value=300, key='vent_rate')
      with ecg_col2:
        qrs_duration = st.number_input('QRS dur.', min_value=50, max_value=200, key='qrs_duration')
      with ecg_col3:
        p_axis = st.number_input('P-axis', min_value=-180, max_value=180, key='p_axis')
      with ecg_col4:
        r_axis = st.number_input('R-axis', min_value=-180, max_value=180, key='r_axis')
      with ecg_col5:
        t_axis = st.number_input('T-axis', min_value=-180, max_value=180, key='t_axis')
      with ecg_col6:
        qt = st.number_input('QT', min_value=200, max_value=600, key='qt')
      with ecg_col7:
        qtc = st.number_input('QTc', min_value=200, max_value=600, key='qtc')
      with ecg_col8:
        bsa = st.number_input('BSA', min_value=0.0, max_value=3.0, step=0.01, key='bsa')

cont2 = st.container()

with cont2:
  cont2_col1, cont2_col2 = st.columns([3,1])  # Adjusted column widths for a better layout of first set of expanders

  # Echocardiogram Variables
  with cont2_col1.expander("Echocardiogram Data", expanded=True):
      echo_col1, echo_col2, echo_col3, echo_col4, echo_col5, echo_col6, echo_col7, echo_col8 = st.columns(8)
      with echo_col1:
        # structural Measurements
        ivsd = st.number_input('IVSd (cm)', min_value=0.0, max_value=2.0, step=0.01)
        lvot_diam = st.number_input('LVOT diam (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvids = st.number_input('LVIDs (cm)', min_value=0.0, max_value=10.0, step=0.01)
        la_dimension = st.number_input('LA dimension (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvidd = st.number_input('LVIDd (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvpwd = st.number_input('LVPWd (cm)', min_value=0.0, max_value=2.0, step=0.01)
        ivs = st.number_input('IVS (cm)', min_value=0.0, max_value=2.0, step=0.01)
        ao_root_diam = st.number_input('Ao root diam (cm)', min_value=0.0, max_value=10.0, step=0.01)
      with echo_col2:
        # Function Measurements
        fs = st.number_input('FS (%)', min_value=0.0, max_value=100.0, step=0.1)
        edv_teich = st.number_input('EDV(Teich) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        lvld_ap4 = st.number_input('LVLd ap4 (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvld_ap2 = st.number_input('LVLd ap2 (cm)', min_value=0.0, max_value=10.0, step=0.01)
        edv_mod_sp4 = st.number_input('EDV(MOD-sp4) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        edv_mod_sp2 = st.number_input('EDV(MOD-sp2) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        edv_sp4_el = st.number_input('EDV(sp4-el) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        edv_sp2_el = st.number_input('EDV(sp2-el) (ml)', min_value=0.0, max_value=500.0, step=1.0)
      with echo_col3:
        lvas_ap4 = st.number_input('LVAs ap4 (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvas_ap2 = st.number_input('LVAs ap2 (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvl_ap4 = st.number_input('LVLs ap4 (cm)', min_value=0.0, max_value=10.0, step=0.01)
        lvl_ap2 = st.number_input('LVLs ap2 (cm)', min_value=0.0, max_value=10.0, step=0.01)
        esv_mod_sp4 = st.number_input('ESV(MOD-sp4) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        esv_mod_sp2 = st.number_input('ESV(MOD-sp2) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        esv_sp4_el = st.number_input('ESV(sp4-el) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        esv_sp2_el = st.number_input('ESV(sp2-el) (ml)', min_value=0.0, max_value=500.0, step=1.0)
      with echo_col4:
        ef_mod_sp4 = st.number_input('EF(MOD-sp4) (%)', min_value=0.0, max_value=100.0, step=0.1)
        ef_sp4_el = st.number_input('EF(sp4-el) (%)', min_value=0.0, max_value=100.0, step=0.1)
        sv_mod_sp4 = st.number_input('SV(MOD-sp4) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        sv_sp4_el = st.number_input('SV(sp4-el) (ml)', min_value=0.0, max_value=500.0, step=1.0)
        ao_root_area = st.number_input('Ao root area (cm2)', min_value=0.0, max_value=10.0, step=0.01)
        laa = st.number_input('LAA (cm2)', min_value=0.0, max_value=10.0, step=0.01)
        raa = st.number_input('RAA (cm2)', min_value=0.0, max_value=10.0, step=0.01)
        mapse = st.number_input('MAPSE (cm)', min_value=0.0, max_value=10.0, step=0.01)   
      with echo_col5:
        tapse = st.number_input('TAPSE (cm)', min_value=0.0, max_value=10.0, step=0.01)
        mv_e_max_vel = st.number_input('MV E max vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        mv_a_max_vel = st.number_input('MV A max vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        mv_e_a = st.number_input('MV E/A', min_value=0.0, max_value=10.0, step=0.01)
        mv_dec_time = st.number_input('MV dec time (ms)', min_value=0.0, max_value=1000.0, step=1.0)
        lat_peak_e_vel = st.number_input('Lat Peak E" Vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        med_peak_e_vel = st.number_input('Med Peak E" Vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        ao_v2_max = st.number_input('Ao V2 max (m/s)', min_value=0.0, max_value=10.0, step=0.01)
      with echo_col6:
        ao_max_pg = st.number_input('Ao max PG (mmHg)', min_value=0.0, max_value=100.0, step=1.0)
        lv_v1_max_pg = st.number_input('LV V1 max PG (mmHg)', min_value=0.0, max_value=100.0, step=1.0)
        lv_v1_max = st.number_input('LV V1 max (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        pa_v2_max = st.number_input('PA V2 max (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        pa_max_pg = st.number_input('PA max PG (mmHg)', min_value=0.0, max_value=100.0, step=1.0)
        tr_max_vel = st.number_input('TR max vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        tr_max_pg = st.number_input('TR max PG (mmHg)', min_value=0.0, max_value=100.0, step=1.0)
        pi_end_d_vel = st.number_input('PI end-d vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
      with echo_col7:
        e_e_lat = st.number_input('E/E" Lat', min_value=0.0, max_value=10.0, step=0.01)
        e_e_med = st.number_input('E/E" Med', min_value=0.0, max_value=10.0, step=0.01)
        desc_ao_max_vel = st.number_input('Desc Ao max vel (m/s)', min_value=0.0, max_value=10.0, step=0.01)
        desc_ao_max_pg = st.number_input('Desc Ao max PG (mmHg)', min_value=0.0, max_value=100.0, step=1.0)
        ao_sinus_diam = st.number_input('Ao sinus diam (cm)', min_value=0.0, max_value=10.0, step=0.01)
        mv_sax_meas_a = st.number_input('MV SAX Measurements A', min_value=0.0, max_value=10.0, step=0.01)
        mv_sax_meas_b = st.number_input('MV SAX Measurements B', min_value=0.0, max_value=10.0, step=0.01)
        mv_sax_meas_c = st.number_input('MV SAX Measurements C', min_value=0.0, max_value=10.0, step=0.01)
      with echo_col8:
        mv_sax_meas_d = st.number_input('MV SAX Measurements D', min_value=0.0, max_value=10.0, step=0.01)
        pm_sax_meas_a = st.number_input('PM SAX Measurements A', min_value=0.0, max_value=10.0, step=0.01)
        pm_sax_meas_b = st.number_input('PM SAX Measurements B', min_value=0.0, max_value=10.0, step=0.01)
        pm_sax_meas_c = st.number_input('PM SAX Measurements C', min_value=0.0, max_value=10.0, step=0.01)
        pm_sax_meas_d = st.number_input('PM SAX Measurements D', min_value=0.0, max_value=10.0, step=0.01)

# Holter Monitor Variables
st.write('## Enter Holter Monitor Data')

artefacts = st.number_input('Artefacts', min_value=0, max_value=1)
normal_count = st.number_input('Normal Count', min_value=0, max_value=1000)
normal_percent = st.number_input('Normal Percent', min_value=0, max_value=100)
normal_max_hour = st.number_input('Normal Max/Hour', min_value=0, max_value=100)
ve_beats_count = st.number_input('VE Beats Count', min_value=0, max_value=1000)
ve_beats_percent = st.number_input('VE Beats Percent', min_value=0, max_value=100)
ve_beats_max_hour = st.number_input('VE Beats Max/Hour', min_value=0, max_value=100)
sve_beats_count = st.number_input('SVE Beats Count', min_value=0, max_value=1000)
sve_beats_percent = st.number_input('SVE Beats Percent', min_value=0, max_value=100)
sve_beats_max_hour = st.number_input('SVE Beats Max/Hour', min_value=0, max_value=100)
paced_beats_count = st.number_input('Paced Beats Count', min_value=0, max_value=1000)
paced_beats_percent = st.number_input('Paced Beats Percent', min_value=0, max_value=100)
paced_beats_max_hour = st.number_input('Paced Beats Max/Hour', min_value=0, max_value=100)
heart_rates_max_hr = st.number_input('Heart Rates (1 min avg) Max HR', min_value=0, max_value=300)
heart_rates_mean_hr = st.number_input('Heart Rates (1 min avg) Mean HR', min_value=0, max_value=300)
heart_rates_min_hr = st.number_input('Heart Rates (1 min avg) Min HR', min_value=0, max_value=300)
bradycardia = st.number_input('Bradycardia', min_value=0, max_value=10000)
bradycardia_event_longest = st.number_input('Bradycardia Event Longest', min_value=0, max_value=1000)
bradycardia_event_min_rate = st.number_input('Bradycardia Event Min Rate', min_value=0, max_value=300)
pause = st.number_input('Pause', min_value=0, max_value=1000)
broad_complex_tachycardia = st.number_input('Broad Complex Tachycardia', min_value=0, max_value=1000)
broad_complex_tachycardia_longest = st.number_input('Broad Complex Tachycardia Longest', min_value=0, max_value=1000)
broad_complex_tachycardia_max_rate = st.number_input('Broad Complex Tachycardia Max Rate', min_value=0, max_value=300)
v_run_aivr = st.number_input('V-Run/AIVR', min_value=0, max_value=1000)
v_run_aivr_longest = st.number_input('V-Run/AIVR Longest', min_value=0, max_value=1000)
v_run_aivr_max_rate = st.number_input('V-Run/AIVR Max Rate', min_value=0, max_value=300)
couplet = st.number_input('Couplet', min_value=0, max_value=1000)
triplet = st.number_input('Triplet', min_value=0, max_value=1000)
single_ve_events = st.number_input('Single VE Events', min_value=0, max_value=1000)
svt = st.number_input('SVT', min_value=0, max_value=1000)
svt_longest = st.number_input('SVT Longest', min_value=0, max_value=1000)
svt_max_rate = st.number_input('SVT Max Rate', min_value=0, max_value=300)
sve = st.number_input('SVE', min_value=0, max_value=1000)
sve_max_per_minute = st.number_input('SVE Max per Minute', min_value=0, max_value=100)
sve_max_per_hour = st.number_input('SVE Max per Hour', min_value=0, max_value=100)
sve_mean_per_hour = st.number_input('SVE Mean per Hour', min_value=0, max_value=100)
sve_run = st.number_input('SVE Run', min_value=0, max_value=1000)
sve_run_longest = st.number_input('SVE Run Longest', min_value=0, max_value=1000)
sve_run_max_rate = st.number_input('SVE Run Max Rate', min_value=0, max_value=300)
holter_date_diff = st.number_input('Holter_date_diff', min_value=0, max_value=1000)
echo_date_diff = st.number_input('Echo_date_diff', min_value=0, max_value=1000)


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
    'Holter_date_diff': [holter_date_diff],
    'Echo_date_diff': [echo_date_diff]
    })

cont4 = st.container()
with cont4:
  cont4_col1, cont4_col2 = st.columns([3,1])
  with cont4_col2:
    st.button('Predict')
    prediction = model.predict_proba(input_data).flatten()

    # Display the prediction
    st.write('## Prediction Probabilities')
    st.altair_chart(altair.Chart(pd.DataFrame({'Condition': ['Hypertrophic Cardiomyopathy', 'Fabry Disease'], 
                                              'Probability': prediction})).mark_bar(orient='horizontal').encode(x='Condition', y='Probability'))
