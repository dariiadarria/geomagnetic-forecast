import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import subprocess
import os
from datetime import datetime
import time
import warnings
import glob

warnings.filterwarnings('ignore')


def load_css():
    with open('styles.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = 'notebooks/models/space_weather_classifier_base_model.pkl'
    try:
        model_info = joblib.load(model_path)
        model = model_info['model']
        accuracy = model_info.get('accuracy_on_test', 'N/A')
        f1_score = model_info.get('f1_weighted_on_test', 'N/A')
        features_used = model_info.get('features_used', 'N/A')
        return model, accuracy, f1_score, features_used
    except FileNotFoundError:
        st.error(f"Файл модели не найден: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        st.stop()


def get_status_for_feature(feature_name, value):
    if feature_name == 'wind_speed':
        if value > 700:
            return "🔴 Высокая"
        elif value > 600:
            return "🟡 Умеренная"
        return "🟢 Низкая"
    elif feature_name == 'wind_density':
        if value > 15:
            return "🔴 Высокая"
        elif value > 10:
            return "🟡 Умеренная"
        return "🟢 Низкая"
    elif feature_name == 'kp_index':
        if value > 4:
            return "🔴 Высокая"
        elif value > 2:
            return "🟡 Умеренная"
        return "🟢 Низкая"
    return "⚪ Неизвестный"


def run_data_pipeline():
    with st.spinner("Шаг 1/2: Сбор данных из api noaa..."):
        try:
            subprocess.run(["python", "src/data_collector.py"], check=False)
        except Exception as e:
            st.error(f"Ошибка при запуске data_collector.py: {e}")
            return None

    with st.spinner("Шаг 2/2: Подготовка данных для прогноза..."):
        try:
            subprocess.run(["python", "src/noaa_preparator.py"], check=False)
        except Exception as e:
            st.error(f"Ошибка при запуске noaa_preparator.py: {e}")
            return None

    noaa_ready_path = 'data/processed/noaa_ready.csv'
    if os.path.exists(noaa_ready_path):
        st.success("Данные успешно обновлены")
        return noaa_ready_path
    else:
        st.error("Файл с данными не создан")
        return None


st.set_page_config(
    page_title="Система мониторинга геомагнитной угрозы",
    page_icon="🌌",
    layout="wide"
)

load_css()

model, model_accuracy, model_f1, model_features = load_model()
st.session_state['loaded_model'] = model

st.markdown("<h1 class='main-header'>🌌 Система мониторинга и прогнозирования геомагнитной угрозы</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Настройки")
    update_interval = st.slider("Интервал обновления (минуты)", 1, 60, 15)
    auto_update = st.checkbox("Автоматическое обновление", value=False)

    if st.button("🔄 Обновить данные из api"):
        st.session_state['update_triggered'] = True
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 Информация о модели")
    st.info("**Модель:** RandomForestClassifier")
    st.write(f"**Точность (test):** {model_accuracy:.3f}" if isinstance(model_accuracy, float) else f"**Точность (test):** {model_accuracy}")
    st.write(f"**F1-weighted (test):** {model_f1:.3f}" if isinstance(model_f1, float) else f"**F1-weighted (test):** {model_f1}")
    st.write(f"**Признаки:** {', '.join(model_features) if isinstance(model_features, list) else model_features}")

    st.markdown("---")
    st.markdown("### 📊 Источники данных")
    st.info("""
    - noaa space weather api
    - Солнечный ветер и магнитное поле
    - Планетарный kp-индекс
    """)


if 'current_data_path' not in st.session_state:
    st.session_state.current_data_path = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_probabilities' not in st.session_state:
    st.session_state.current_probabilities = None
if 'current_features' not in st.session_state:
    st.session_state.current_features = None
if 'update_triggered' not in st.session_state:
    st.session_state.update_triggered = False
if 'api_data_cache' not in st.session_state:
    st.session_state.api_data_cache = {}


if st.session_state.update_triggered:
    st.session_state.update_triggered = False
    data_path = run_data_pipeline()
    if data_path:
        st.session_state.current_data_path = data_path
        try:
            df_current = pd.read_csv(data_path)
            features_for_model = df_current[['kp_index', 'wind_speed', 'wind_density']].values
            st.session_state.current_features = features_for_model

            loaded_model = st.session_state['loaded_model']
            prediction = loaded_model.predict(features_for_model)[0]
            probabilities = loaded_model.predict_proba(features_for_model)[0]

            st.session_state.current_prediction = prediction
            st.session_state.current_probabilities = probabilities

            api_data_cache = {}
            patterns = {
                'solar_wind_plasma': 'data/solar_wind_plasma_*.csv',
                'solar_wind_mag': 'data/solar_wind_mag_*.csv',
                'kp_index': 'data/kp_index_*.csv',
                'dst_index': 'data/dst_index_*.csv'
            }
            for name, pattern in patterns.items():
                files = glob.glob(pattern)
                if files:
                    latest_file = max(files, key=os.path.getmtime)
                    try:
                        df = pd.read_csv(latest_file)
                        if 'time_tag' in df.columns:
                            df['time_tag'] = pd.to_datetime(df['time_tag'], errors='coerce')
                        df = df.dropna(subset=['time_tag'])
                        api_data_cache[name] = df
                    except Exception:
                        pass
            st.session_state.api_data_cache = api_data_cache
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")


tab1, tab2 = st.tabs(["📊 Мониторинг", "🤖 Прогноз"])

with tab1:
    st.header("📊 Текущее состояние космической погоды")
    if st.session_state.current_data_path and st.session_state.current_features is not None:
        df_display = pd.read_csv(st.session_state.current_data_path)
        kp_val = df_display['kp_index'].iloc[0]
        ws_val = df_display['wind_speed'].iloc[0]
        wd_val = df_display['wind_density'].iloc[0]
        timestamp_val = df_display['timestamp'].iloc[0] if 'timestamp' in df_display.columns else "N/A"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_kp = get_status_for_feature('kp_index', kp_val)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Kp-индекс</h3>
                <h2>{kp_val:.2f}</h2>
                <p>{status_kp}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            status_ws = get_status_for_feature('wind_speed', ws_val)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Скорость ветра</h3>
                <h2>{ws_val:.1f} км/с</h2>
                <p>{status_ws}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            status_wd = get_status_for_feature('wind_density', wd_val)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Плотность</h3>
                <h2>{wd_val:.2f} част/см³</h2>
                <p>{status_wd}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Последнее обновление</h3>
                <h4>{timestamp_val}</h4>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Детали подготовленных данных"):
            st.dataframe(df_display)
    else:
        st.info("Данные не обновлены. Нажмите 'Обновить данные из api' в боковой панели.")

    if st.session_state.api_data_cache:
        st.subheader("📈 Динамика параметров")
        api_data = st.session_state.api_data_cache

        if 'solar_wind_plasma' in api_data:
            df_plasma = api_data['solar_wind_plasma']
            if not df_plasma.empty:
                numeric_cols = ['speed', 'density', 'temperature']
                for col in numeric_cols:
                    if col in df_plasma.columns:
                        df_plasma[col] = pd.to_numeric(df_plasma[col], errors='coerce')

                df_plasma = df_plasma.sort_values('time_tag')

                fig_plasma, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
                ax1.plot(df_plasma['time_tag'], df_plasma['speed'], label='Скорость (км/с)', color='blue')
                ax1.axhline(y=600, color='red', linestyle='--', alpha=0.5, label='Порог 600 км/с')
                ax1.set_ylabel('Скорость (км/с)')
                ax1.set_title('Скорость солнечного ветра')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(df_plasma['time_tag'], df_plasma['density'], label='Плотность (частиц/см³)', color='green')
                ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Порог 10 част/см³')
                ax2.set_ylabel('Плотность (частиц/см³)')
                ax2.set_title('Плотность солнечного ветра')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                ax3.plot(df_plasma['time_tag'], df_plasma['temperature'], label='Температура (K)', color='orange')
                ax3.set_ylabel('Температура (K)')
                ax3.set_title('Температура плазмы')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_xlabel('Время')

                plt.tight_layout()
                st.pyplot(fig_plasma)
                plt.close(fig_plasma)

        if 'solar_wind_mag' in api_data:
            df_mag = api_data['solar_wind_mag']
            if not df_mag.empty:
                numeric_cols = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
                for col in numeric_cols:
                    if col in df_mag.columns:
                        df_mag[col] = pd.to_numeric(df_mag[col], errors='coerce')

                df_mag = df_mag.sort_values('time_tag')

                fig_mag, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                ax1.plot(df_mag['time_tag'], df_mag['bz_gsm'], label='Bz (nT)', color='red')
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Bz = 0')
                ax1.axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='Порог Bz = -5 nT')
                ax1.set_ylabel('Bz (nT)')
                ax1.set_title('Компонента bz межпланетного магнитного поля')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(df_mag['time_tag'], df_mag['bt'], label='Bt (nT)', color='purple')
                ax2.set_ylabel('Bt (nT)')
                ax2.set_title('Общая величина магнитного поля')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_xlabel('Время')

                plt.tight_layout()
                st.pyplot(fig_mag)
                plt.close(fig_mag)

        if 'kp_index' in api_data:
            df_kp = api_data['kp_index']
            if not df_kp.empty:
                if 'Kp' in df_kp.columns:
                    df_kp['Kp'] = pd.to_numeric(df_kp['Kp'], errors='coerce')

                df_kp = df_kp.sort_values('time_tag')

                fig_kp, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df_kp['time_tag'], df_kp['Kp'], label='Kp-индекс', color='brown', marker='o', markersize=3)
                ax.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Порог Kp = 4')
                ax.set_ylabel('Kp-индекс')
                ax.set_title('Планетарный kp-индекс')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Время')

                plt.tight_layout()
                st.pyplot(fig_kp)
                plt.close(fig_kp)
    else:
        st.info("Графики недоступны. Обновите данные из api.")


with tab2:
    st.header("🤖 Прогноз геомагнитной угрозы")
    if st.session_state.current_prediction is not None and st.session_state.current_probabilities is not None:
        prediction = st.session_state.current_prediction
        probabilities = st.session_state.current_probabilities
        threat_levels = ['Низкая', 'Умеренная', 'Высокая']
        colors = ['🟢', '🟡', '🔴']
        alerts = ['alert-low', 'alert-medium', 'alert-high']

        if st.session_state.current_data_path:
            df_display = pd.read_csv(st.session_state.current_data_path)
            kp_val = df_display['kp_index'].iloc[0]
            ws_val = df_display['wind_speed'].iloc[0]
            wd_val = df_display['wind_density'].iloc[0]

            st.markdown("""
            <div class="prediction-card">
                <h3>📊 Текущие параметры космической погоды</h3>
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Kp-индекс", f"{kp_val:.2f}")
            with col2:
                st.metric("Скорость ветра", f"{ws_val:.1f} км/с")
            with col3:
                st.metric("Плотность", f"{wd_val:.2f} част/см³")

        st.markdown("""
        <div class="prediction-card">
            <h3>🎯 Результат прогноза</h3>
        </div>
        """, unsafe_allow_html=True)

        confidence = probabilities[prediction] * 100
        threat_color = alerts[prediction]
        st.markdown(f"""
        <div class="{threat_color}">
            <h3>{colors[prediction]} Уровень угрозы: {threat_levels[prediction]} ({confidence:.1f}% уверенности)</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="prediction-card">
            <h3>📈 Вероятности</h3>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Низкая", f"{probabilities[0]:.1%}")
        with col2:
            st.metric("Умеренная", f"{probabilities[1]:.1%}")
        with col3:
            st.metric("Высокая", f"{probabilities[2]:.1%}")

        fig2, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(threat_levels, probabilities * 100, color=['green', 'yellow', 'red'])
        ax.set_ylabel('Вероятность (%)')
        ax.set_title('Распределение вероятностей')
        ax.set_ylim(0, 110)
        for bar, prob in zip(bars, probabilities * 100):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig2)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Факторы риска")
            risk_factors = []
            if ws_val > 600:
                risk_factors.append(f"⚡ Высокая скорость ветра ({ws_val:.1f} км/с)")
            if kp_val > 4:
                risk_factors.append(f"⚡ Высокий kp-индекс ({kp_val:.2f})")
            if wd_val > 10:
                risk_factors.append(f"💨 Повышенная плотность ({wd_val:.2f} част/см³)")
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.info("✅ Параметры в пределах нормы")
        with col2:
            st.subheader("🎯 Рекомендации")
            if prediction == 0:
                st.success("""
                **Штатный режим:**
                - Обычная работа систем
                - Плановое обслуживание разрешено
                - Мониторинг в обычном режиме
                """)
            elif prediction == 1:
                st.warning("""
                **Повышенная готовность:**
                - Усиленный мониторинг
                - Готовность к отключению не критичных систем
                - Уведомление персонала
                """)
            else:
                st.error("""
                **Высокая угроза:**
                - Активное предупреждение операторов
                - Подготовка к отключениям
                - Резервные системы в режиме готовности
                - Непрерывный мониторинг
                """)
    else:
        st.warning("Прогноз недоступен. Обновите данные из api.")


st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🌌 Система мониторинга геомагнитной угрозы")
with col2:
    st.caption(f"Последнее обновление данных: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    st.caption("Источник данных: noaa swpc api")


if auto_update:
    time.sleep(update_interval * 60)
    st.session_state['update_triggered'] = True
    st.rerun()