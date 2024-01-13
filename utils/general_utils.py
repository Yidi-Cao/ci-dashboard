import streamlit as st
import base64
import time
from PIL import Image
import random
import math


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


def my_progress_bar(pers_step=3, num_steps=10):
    # progress bar
    my_bar = st.progress(0, text=f'0/{num_steps}')
    for complete_steps in range(num_steps):
        time.sleep(pers_step * random.uniform(0.8,1.25))
        my_bar.progress(math.floor((complete_steps+1)*100/num_steps), text=f'{complete_steps + 1}/{num_steps}')


def display_props():
    # feature image
    image = Image.open('images/gammalogo.png')
    st.sidebar.image(image, width=150)
    return


def display_pdf(pdf_file, w=700, h=1000):
    # Opening file from file path
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{w}" height="{h}" type="application/pdf"></iframe>'

    # Displaying File

    st.markdown(pdf_display, unsafe_allow_html=True)

def display_image(image_file, w=700):
    image = Image.open(image_file)
    st.image(image, width=w)