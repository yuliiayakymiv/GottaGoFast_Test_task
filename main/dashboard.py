import streamlit as st

# import google.generativeai as genai

st.set_page_config(
    page_title="Аналізатор польотів",
    page_icon="🚁",
    layout="wide"
)

st.title("Аналізатор польотів дронів")
st.markdown("Завантаж файл `.bin` і отримай аналіз + AI-звіт")



with st.sidebar:
    st.header("Налаштування")
    api_key = st.text_input(
        "Google Gemini API ключ",
        type="password",
        help="Отримай безкоштовно на aistudio.google.com"
    )
    st.markdown("---")
    st.caption("Введи API ключ і завантаж файл")



# Завантаження файлу
uploaded_file = st.file_uploader(
    "Виберіть BIN файл польоту",
    type=['bin']
)

if uploaded_file is not None:
    st.success(f"Файл {uploaded_file.name} завантажено")

    # Тут будуть метрики
    st.subheader("Метрики польоту")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Тривалість", "—")
    with col2:
        st.metric("Дистанція", "—")
    with col3:
        st.metric("Макс швидкість", "—")
    with col4:
        st.metric("Макс висота", "—")

    # Тут буде AI-звіт
    st.markdown("---")
    st.subheader("AI-аналіз аномалій польоту")

    if st.button("Згенерувати AI звіт", type="primary"):
        if not api_key:
            st.error("Введи API ключ у бічній панелі")
        else:
            st.info("Тут буде звіт від AI")

else:
    st.info("Завантаж BIN файл, щоб почати")
