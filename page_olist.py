import datetime

import pandas as pd
import streamlit as st
from PIL import Image
import time
import pickle


#from olist_asil_son_model import ml_pipeline,pipeline_hazirlik,data_pipeline,read_data
#import olist_asil_son_model as model
import Scripts.ml_pipeline as ml

st.set_page_config(page_title="OLIST", page_icon=":truck:", layout="wide")
def app():

    st.audio("mvk.mp3")

    image = Image.open('img.png')
    st.image(image, caption='Brazillian E-Commerce Olist')


    st.title('ESTIMATED DELIVERY TIME')

    #------------------------------------------
    df_products = pd.read_csv("datasets/olist_products_dataset.csv")
    df_sellers = pd.read_csv("datasets/olist_sellers_dataset.csv")
    df_seller_mean_data = pd.read_csv("Helper Datasets/seller_id_mean_data.csv")
    df_geolocation = pd.read_csv("datasets/olist_geolocation_dataset.csv")
    df_customers = pd.read_csv("datasets/olist_customers_dataset.csv")

    categories = df_products["product_category_name"].unique()

    # 0. DROPDOWN - TARİH
    option_date = st.date_input(
        "PURCHASE DATE",
        datetime.date(2019, 5, 6))

    option_date = str(option_date) +" 00:00:00"
    option_date = pd.to_datetime(option_date)


    # 1. DROPDOWN - KATEGORİ
    option_category = st.selectbox(
        'PRODUCT CATEGORY',
        # ('Email', 'Home phone', 'Mobile phone'))
        (categories))


    # 2. DROPDOWN - KATEGORİ
    if option_category != "":
        df_cat = df_products[df_products["product_category_name"] == option_category]

        opt_products = df_cat["product_id"].head(20)


        option_productId = st.selectbox(
            'PRODUCT',
            (opt_products))

    # 3. DROPDOWN - ZIP CODE
    if option_productId != "":
        df_zip_code = df_customers["customer_zip_code_prefix"].unique()

        #row1 = df_seller_id.sample(n=5)

        option_zipCode = st.selectbox(
            'DELIVERY ADDRESS ZIP CODE',
            (df_zip_code))

    option_zipCode = int(option_zipCode)

    # 4. DROPDOWN - SELLER ID
    if option_zipCode != "":


        df_seller_id =pd.Series(df_seller_mean_data["seller_id"].unique())

        row1 = df_seller_id.sample(n=5, random_state=option_zipCode)

        option_sellerId = st.selectbox(
            'SELLER',
            (row1))

    # 5 MAP

    seller_zipCode = df_sellers[df_sellers["seller_id"] == option_sellerId]["seller_zip_code_prefix"]

    # Teslimat adresi
    df_geolocation = df_geolocation[(df_geolocation["geolocation_zip_code_prefix"] == option_zipCode) |
                                    (df_geolocation["geolocation_zip_code_prefix"] == seller_zipCode.values[0])]

    df_geolocation = df_geolocation.groupby("geolocation_zip_code_prefix").agg(
        {"geolocation_lat": "mean", "geolocation_lng": "mean"})
    df_geolocation.reset_index(inplace=True)
    df = df_geolocation[["geolocation_lat","geolocation_lng"]].rename(columns={"geolocation_lat":'lat', "geolocation_lng":'lon'})

    # Satıcı adresi
    st.write("ORDER DELIVERY MAP")
    st.map(df,use_container_width=False)


    if st.button('SHOW DELIVERY TIME'):
        if option_sellerId == "" or option_category =="" or option_zipCode=="" or option_productId=="":
            st.warning('Değer giriniz')
        else:

            with st.spinner('Wait for it...'):
                time.sleep(2)

            result = ml.ml_pipeline(option_productId,option_sellerId,option_zipCode,option_date)

            st.success(f"🇬🇧 Estimated product delivery time  :  {result} Days")
            st.success(f"🇧🇷 Tempo estimado de entrega do produto  :  {result} Dias")

            # st.balloons()
            # st.snow()


    else:
        st.write('No Data')


# TODO  EDA kısmını STREAMLITe EKLE


#----------- 6-7 GÜN 2018-10-18
#----------- 1-3 GÜN 2019-05-06

# seguros_e_servicos
# 8db75af9aed3315374db44d7860e25da
# 5704
# 4d9fea3499bdc22aa4da4e339365f215

#----------- 8-11 GÜN ÖRNEK

# cool_stuff
# b5cfb1d3c5e435a7a52227e08f220ee7
# 35182
# c9a06ece156bb057372c68718ec8909b
