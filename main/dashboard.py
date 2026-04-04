"""
інтерактивний дашборд, AI-асистент
"""

import streamlit as st
import os
import tempfile
from groq import Groq
import pandas as pd

from parser import TelemetryParser
from analytics import get_metrics

st.set_page_config(
    page_title="Аналізатор польотів",
    page_icon="🚁",
    layout="wide"
)

st.title("Аналізатор польотів дронів")
st.markdown("Завантаж файл `.bin` і отримай аналіз + AI-звіт")


with st.sidebar:
    st.header("Налаштування")
    try:
        groq_api_key  = st.secrets["GROQ_API_KEY"]
    except:
        groq_api_key  = None

    st.markdown("---")
    st.caption("Після завантаження файлу аналіз запуститься автоматично")


def generate_ai_report(metrics: dict, api_key: str) -> str:
    """Генерує текстовий звіт про аномалії польоту на основі метрик"""
    if not api_key:
        return "API ключ не введено"

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
        Ти експерт з аналізу польотів дронів на базі Ardupilot.

        Ось отримані дані польоту:
        - Загальна дистанція: {metrics.get('total_distance_m', 0):.0f} метрів
        - Тривалість польоту: {metrics.get('duration_s', 0):.1f} секунд ({metrics.get('duration_s', 0)/60:.1f} хв)
        - Максимальна горизонтальна швидкість: {metrics.get('max_speed_h_kmh', 0):.1f} км/год
        - Максимальна вертикальна швидкість: {metrics.get('max_speed_v_ms', 0):.1f} м/с
        - Максимальний набір висоти: {metrics.get('max_altitude_gain_m', 0):.0f} метрів
        - Максимальне прискорення: {metrics.get('max_acceleration_ms2', 0):.1f} м/с²
        - Максимальна швидкість (з IMU): {metrics.get('max_speed_imu_ms', 0):.1f} м/с

        Проаналізуй ці дані українською мовою:
        1. Чи є аномалії? (різкі прискорення, надто висока швидкість, різкі зміни висоти)
        2. Загальна оцінка польоту (успішний/нестабільний/аварійний)
        3. Рекомендації щодо покращення або на що звернути увагу

        Відповідай коротко (3-5 речень).
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Помилка при генерації AI звіту: {str(e)}"

# Завантаження файлу
uploaded_file = st.file_uploader(
    "Виберіть BIN файл польоту",
    type=['bin']
)

if uploaded_file is not None:
    st.success(f"Файл {uploaded_file.name} завантажено")

    # Зберігаємо завантажений файл тимчасово на диск
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Парсимо лог і отримуємо метрики
    with st.spinner("Аналізую лог..."):
        try:
            parser = TelemetryParser(tmp_path)
            df_gps, df_imu = parser.parse()
            metrics = get_metrics(df_gps, df_imu)

            # Виводимо метрики у вигляді 4 колонок
            st.subheader("Метрики польоту")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Тривалість", f"{metrics.get('duration_s', 0):.1f} с")
            with col2:
                st.metric("Дистанція", f"{metrics.get('total_distance_m', 0):.0f} м")
            with col3:
                st.metric("Макс. швидкість (гориз.)", f"{metrics.get('max_speed_h_kmh', 0):.1f} км/год")
            with col4:
                st.metric("Макс. набір висоти", f"{metrics.get('max_altitude_gain_m', 0):.0f} м")

            # Додаткові метрики в розгорнутому вигляді
            with st.expander("Детальні метрики"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Вертикальна швидкість (макс.)", f"{metrics.get('max_speed_v_ms', 0):.1f} м/с")
                with col_b:
                    st.metric("Прискорення (макс.)", f"{metrics.get('max_acceleration_ms2', 0):.1f} м/с²")
                with col_c:
                    st.metric("Швидкість з IMU (макс.)", f"{metrics.get('max_speed_imu_ms', 0):.1f} м/с")

        except Exception as e:
            st.error(f"Помилка при аналізі: {str(e)}")
            metrics = {}

    # Видаляємо тимчасовий файл
    try:
        os.unlink(tmp_path)
    except PermissionError:
        pass  # Файл уже видалений або ще використовується

    st.markdown("---")
    st.subheader("AI-аналіз аномалій польоту")

    if st.button("Згенерувати AI звіт", type="primary"):
        if not groq_api_key:
            st.error("API ключ не налаштовано.")
        elif not metrics:
            st.error("Немає даних для аналізу")
        else:
            with st.spinner("Штучний інтелект аналізує..."):
                report = generate_ai_report(metrics, groq_api_key)
                st.success("Звіт готовий")
                st.info(report)

else:
    st.info("Завантаж BIN файл, щоб почати")
