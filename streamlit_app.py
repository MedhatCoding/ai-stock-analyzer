# -*- coding: utf-8 -*-
"""
عقار — محلل البورصة المصرية بالذكاء الاصطناعي
غلاف Streamlit يعرض التطبيق الكامل (index.html) داخل الصفحة.
يعمل على Streamlit Cloud (share.streamlit.io) وعلى جهازك المحلي.
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="عقار — محلل البورصة المصرية بالذكاء الاصطناعي",
    page_icon="🕌",
    layout="wide",
)

# إخفاء القوائم والفوتر الافتراضية لمظهر أنظف
st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding: 0 !important; max-width: 100% !important;}
      .stApp {background: #0a0f1c;}
    </style>
    """,
    unsafe_allow_html=True,
)

html = Path("index.html").read_text(encoding="utf-8")
components.html(html, height=1200, scrolling=True)
