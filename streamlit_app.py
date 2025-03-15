import streamlit as st

main_page = st.Page("main.py", title = "Machine Learning", icon = "")
page_2 = st.Page("page_2.py", title = "Detail", icon = "")
page_3 = st.Page("page_3.py", title = "Neural Network", icon = "")
page_4 = st.Page("page_4.py", title = "Detail", icon = "")

pg = st.navigation([main_page, page_2, page_3, page_4])

pg.run()