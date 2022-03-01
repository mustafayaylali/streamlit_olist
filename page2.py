import time

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image



orders = pd.read_csv("datasets/olist_orders_dataset.csv")
order_items = pd.read_csv("datasets/olist_order_items_dataset.csv")
geolocation = pd.read_csv("datasets/olist_geolocation_dataset.csv")

def app():
    image = Image.open('img.png')
    st.image(image, caption='Brazillian E-Commerce Olist')

    st.title('OLIST')
    st.write('ESTIMATED DELIVERY TIME RECOMMENDER')

    st.text('This is some text.')

    st.write(orders.shape)

    row1 = orders.sample(n=1)
    row2 = geolocation.sample(n=1)

    st.dataframe(row1)
    st.table(row2)

    #DROPDOWN
    option = st.selectbox(
                '1. Seçenek',
                   # ('Email', 'Home phone', 'Mobile phone'))
                    (orders["order_status"].unique()))

    st.write('You selected:', option)

    t_zipCode = st.text_input("Enter ZIP CODE (Ör:51350)")
    st.write('Girilen Zip Code:', t_zipCode)

    if st.button('GET DATA'):
        if t_zipCode == "":
            st.warning('Değer giriniz')
        else:

            with st.spinner('Wait for it...'):
                time.sleep(3)

            st.success('Done!')

            st.balloons()

            abc = geolocation[geolocation["geolocation_zip_code_prefix"] == int(t_zipCode)]

            st.dataframe(abc)
    else:
        st.write('No Data')
