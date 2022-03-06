import page1
import page2
import page_olist
import streamlit as st

# RUN: streamlit run main.py

# DOCS : https://docs.streamlit.io/library/api-reference/charts

# PAGES = {
#     "Olist Model": page_olist,
#     "EDA": page2,
#     "BMI Calculator": page1,
# }
# st.sidebar.title('ERİŞMEK İÇİN QR KODUNU OKUTABİLİRSİNİZ...')
# st.sidebar.image("olist_qr_code.png")

# selection = st.sidebar.radio("Go to", list(PAGES.keys()))
# page = PAGES[selection]
# page.app()

page_olist.app()