import streamlit as st
from PIL import Image
import time
import pickle

# from olist_asil_son_model import ml_pipeline,pipeline_hazirlik,data_pipeline,read_data
#
# from olist_asil_son_model import read_data,pipeline_hazirlik,data_pipeline,ml_pipeline

import olist_asil_son_model as model


def app():
    image = Image.open('img.png')
    st.image(image, caption='Brazillian E-Commerce Olist')

    st.title('OLIST')
    st.write('ESTIMATED DELIVERY TIME RECOMMENDER')


    t_zipCode = st.text_input("Enter ZIP CODE (Ör:51350)")
    st.write('Girilen Zip Code:', t_zipCode)

    #TODO user friendly alabiliriz
    t_productId = st.text_input("Enter Product ID (Ör:1e9e8ef04dbcff4541ed26657ea517e5)")
    st.write('Girilen Ürün Kodu:', t_productId)

    t_sellerId = st.text_input("Enter Seller ID (Ör:0015a82c2db000af6aaaf3ae2ecb0532)")
    st.write('Girilen Satıcı ID:', t_sellerId)


    ###############


    # Load the pickled model
    final_model = pickle.load(open("finalized_model.sav", "rb"))

    # Use the loaded pickled model to make predictions



    if st.button('GET DATA'):
        if t_zipCode == "":
            st.warning('Değer giriniz')
        else:

            with st.spinner('Wait for it...'):
                time.sleep(3)

            # PREDICT
            test = model.ml_pipeline(t_productId, t_zipCode, t_sellerId)
            #final_model.predict(test)
            st.write(f"Ürünün Tahmini Teslimat Süresi (Tempo estimado de entrega do produto) :  {final_model.predict(test)[0]}")

            st.success('Done!' + str(final_model.predict(test)[0]))

            st.balloons()
            st.balloons()
            st.balloons()
            st.balloons()
            st.balloons()
            st.balloons()
            st.balloons()

            #abc = geolocation[geolocation["geolocation_zip_code_prefix"] == int(t_zipCode)]
            # st.dataframe(abc)
    else:
        st.write('No Data')


# TODO
#       -- Streamlit
# 1) Free hosting-domain
# 2) Her seferinde IMPORT ile kodların çalıştırılması
# 3) Form geliştirilecek

#       -- Model
# 4) DOCSTRING EKle, her fonksiyona
# 5) EDA kısmını STREAMLITe EKLE