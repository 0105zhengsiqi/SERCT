import streamlit as st
from utils import get_wavs, build_model, infer
import os

model = build_model()
output = None

with st.container():
    with st.columns([0.02, 0.96, 0.02])[1]:
        st.image('src/banner_small.jpg', caption='System workflow diagram')

    select_col, upload_col = st.columns(2)
    with select_col:
        wavs = get_wavs()
        option = st.selectbox(
            "What kind of emotional voice do you like?",
            wavs
        )
        st.write("You selected:", option)
        wav_path = os.path.join('emotions', option)
        st.audio(wav_path)
        if option is not None:
            if st.button("Detect", use_container_width=True, key='select'):
                output = infer(wav_path, model)
        else:
            st.warning('Choose one audio file before detection', icon="⚠️")

    with upload_col:
        uploaded_file = st.file_uploader("Choose a wav file", type='wav')
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            st.audio(bytes_data)

        if st.button("Detect", use_container_width=True, key='upload'):
            if uploaded_file is not None:
                output = infer(uploaded_file, model)
            else:
                st.warning('Choose one audio file before detection', icon="⚠️")

    with st.columns(3)[1]:
        if output is not None:
            st.markdown("""Model output: **{output}**! :balloon:""".format(output=output))
