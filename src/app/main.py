import streamlit as st

st.set_page_config(page_title='Industrial: Models that learn how processes unfold', layout='wide')
st.title('Industrial: Models that learn how processes unfold')
st.write('Train a sequence model that learns industrial process trajectories and predicts the next states, bottlenecks, or anomalies.')

st.info('Hackathon scaffold: connect data/model/evaluation modules here after case reveal.')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Scenario')
    st.text_area('Input scenario', 'Paste or select a case after track reveal.')
    st.button('Run model / agent')
with col2:
    st.subheader('Output')
    st.write('Prediction / recommendation will appear here.')
    st.write('Confidence, uncertainty, or explanation will appear here.')

st.subheader('Evaluation evidence')
st.write('Baseline vs improved metric table will appear here.')
