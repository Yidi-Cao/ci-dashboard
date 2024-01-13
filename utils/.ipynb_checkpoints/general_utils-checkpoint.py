import streamlit as st
import base64
import time
from PIL import Image


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    max_width = 1000
    padding_top = 1
    padding_right = 1
    padding_left = 1
    padding_bottom = 1
    # COLOR = 'black'
    # BACKGROUND_COLOR = 'white'
    bin_str = get_base64_of_bin_file(png_file)
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
        }}
    </style>
    """ % bin_str,
        unsafe_allow_html=True,
    )
    return None


def set_png_as_page_bg_tmp(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return None


def my_progress_bar():
    # progress bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)


def display_props():
    # header
    # st.markdown("## 聚类分析 ")
    # feature image
    image = Image.open('static/gammalogo.png')
    st.sidebar.image(image, use_column_width=True)
    return
