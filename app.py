"""
App de Valoración de Riesgos Financieros
Alumna: Alejandra Martínez Cuen (#13)
Activos: PFE, SHIB-USD, CL=F
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


# ---------- Configuración de la página ----------
st.set_page_config(
    page_title="Riesgos Financieros - Alejandra Martínez",
    page_icon="💹",
    layout="wide"
)


# ---------- Diccionario de activos ----------
ACTIVOS = {
    "PFE (Pfizer - Acción)": "PFE",
    "SHIB-USD (Shiba Inu - Criptomoneda)": "SHIB-USD",
    "CL=F (Crude Oil WTI - Materia Prima)": "CL=F"
}


# ---------- Función de descarga de datos (Fase 1) ----------
@st.cache_data(ttl=3600)
def obtener_historicos(ticker, anios):
    fin = datetime.today()
    inicio = fin - timedelta(days=365 * anios)

    try:
        df = yf.download(ticker, start=inicio, end=fin,
                         progress=False, auto_adjust=True, threads=False)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=f"{anios}y", auto_adjust=True)
        except Exception as e:
            st.error(f"No se pudo descargar {ticker}: {e}")
            return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return pd.DataFrame()

    return df[["Close"]].dropna()


# ---------- Encabezado ----------
st.title(" Valoración de Riesgos Financieros")
st.write("**Alumna:** Alejandra Martínez Cuen  |  **Activos analizados:** PFE · SHIB-USD · CL=F")
st.divider()


# ---------- Sidebar con parámetros ----------
st.sidebar.title(" Parámetros")

activo_seleccionado = st.sidebar.selectbox("Activo:", list(ACTIVOS.keys()))
ticker = ACTIVOS[activo_seleccionado]

anios = st.sidebar.slider("Años de historia:", 1, 5, 1)

nivel_confianza = st.sidebar.radio(
    "Nivel de confianza VaR:",
    [0.90, 0.95, 0.99],
    index=1,
    format_func=lambda x: f"{int(x*100)}%",
    horizontal=True
)

tasa_libre = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%):",
    0.0, 20.0, 4.5, 0.1
) / 100

umbral_var = st.sidebar.slider(
    "Umbral de alerta VaR (%):",
    1.0, 10.0, 3.0, 0.5
) / 100


# ---------- Descarga y preparación de datos ----------
with st.spinner(f"Descargando datos de {ticker}..."):
    datos = obtener_historicos(ticker, anios)

if datos.empty:
    st.error("No se pudieron obtener datos. Intenta con otro activo.")
    st.stop()

datos["Retorno"] = np.log(datos["Close"] / datos["Close"].shift(1))
datos = datos.dropna()
retornos = datos["Retorno"]


# ---------- Cálculo de indicadores (Fase 2) ----------
vol_diaria = retornos.std()
vol_anual = vol_diaria * np.sqrt(252)

z_score = stats.norm.ppf(1 - nivel_confianza)
var_param = retornos.mean() + z_score * vol_diaria

maximo_acum = datos["Close"].cummax()
drawdown_serie = (datos["Close"] - maximo_acum) / maximo_acum
max_drawdown = drawdown_serie.min()

retorno_anual = retornos.mean() * 252
sharpe = (retorno_anual - tasa_libre) / vol_anual if vol_anual != 0 else 0


# ============================================================
# SECCIÓN 1: ALERTA DE RIESGO (arriba, visible inmediatamente)
# ============================================================
st.header(f"Resultados para {activo_seleccionado}")

if abs(var_param) > umbral_var:
    st.error(
        f"🔴 **ALERTA DE RIESGO ALTO** — El VaR ({abs(var_param)*100:.2f}%) "
        f"supera el umbral establecido ({umbral_var*100:.1f}%).\n\n"
        f"**Medidas sugeridas:** diversificar la cartera, aplicar coberturas "
        f"con opciones o futuros, reducir exposición y establecer stop-loss."
    )
elif abs(var_param) > umbral_var * 0.7:
    st.warning(
        f"🟡 **Riesgo moderado** — El VaR está cerca del umbral. "
        f"Monitorear diariamente."
    )
else:
    st.success(
        f"🟢 **Riesgo controlado** — El VaR ({abs(var_param)*100:.2f}%) "
        f"se encuentra dentro del umbral aceptable ({umbral_var*100:.1f}%)."
    )


# ============================================================
# SECCIÓN 2: INDICADORES CLAVE
# ============================================================
st.subheader(" Indicadores de riesgo")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Volatilidad Anual", f"{vol_anual*100:.2f}%")
m2.metric(f"VaR {int(nivel_confianza*100)}%", f"{var_param*100:.2f}%")
m3.metric("Máx. Drawdown", f"{max_drawdown*100:.2f}%")
m4.metric("Ratio de Sharpe", f"{sharpe:.2f}")

st.divider()


# ============================================================
# SECCIÓN 3: DISTRIBUCIÓN DE RETORNOS (Fase 3 - histograma con VaR)
# ============================================================
st.subheader("📉 Distribución de retornos diarios")

grafico_hist = go.Figure()
grafico_hist.add_trace(go.Histogram(
    x=retornos, nbinsx=60, name="Retornos",
    marker_color="#6A4C93", opacity=0.75
))
grafico_hist.add_vline(
    x=var_param, line_width=3, line_dash="dash", line_color="red",
    annotation_text=f"VaR {int(nivel_confianza*100)}% = {var_param*100:.2f}%",
    annotation_position="top left"
)
grafico_hist.add_vline(
    x=retornos.mean(), line_width=2, line_dash="dot", line_color="green",
    annotation_text=f"Media = {retornos.mean()*100:.3f}%",
    annotation_position="top right"
)
grafico_hist.update_layout(
    xaxis_title="Retorno diario",
    yaxis_title="Frecuencia",
    height=400,
    bargap=0.05
)
st.plotly_chart(grafico_hist, use_container_width=True)

st.caption(
    "🔎 La región a la izquierda de la línea roja representa los escenarios "
    "de pérdida extrema (análisis de cola izquierda)."
)

st.divider()


# ============================================================
# SECCIÓN 4: COMPARATIVA PRECIOS vs RETORNOS
# ============================================================
st.subheader(" Precios históricos y frecuencia de retornos")

col_izq, col_der = st.columns(2)

with col_izq:
    fig_precio = go.Figure()
    fig_precio.add_trace(go.Scatter(
        x=datos.index, y=datos["Close"],
        mode="lines",
        line=dict(color="#2E86AB", width=2),
        name="Precio"
    ))
    fig_precio.update_layout(
        title="Evolución de precios",
        xaxis_title="Fecha",
        yaxis_title="Precio de cierre",
        height=380
    )
    st.plotly_chart(fig_precio, use_container_width=True)

with col_der:
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown_serie.index, y=drawdown_serie * 100,
        fill="tozeroy", mode="lines",
        line=dict(color="crimson"),
        name="Drawdown"
    ))
    fig_dd.update_layout(
        title="Evolución del Drawdown (%)",
        xaxis_title="Fecha",
        yaxis_title="Drawdown (%)",
        height=380
    )
    st.plotly_chart(fig_dd, use_container_width=True)

st.divider()


# ============================================================
# SECCIÓN 5: REPORTE COMPARATIVO DE LOS 3 ACTIVOS
# ============================================================
st.subheader(" Análisis comparativo de los 3 activos")

@st.cache_data(ttl=3600)
def metricas_activo(ticker, anios):
    df = obtener_historicos(ticker, anios)
    if df.empty:
        return None
    ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    vol = ret.std() * np.sqrt(252)
    var95 = ret.mean() + stats.norm.ppf(0.05) * ret.std()
    cummax = df["Close"].cummax()
    dd = ((df["Close"] - cummax) / cummax).min()
    sharpe_r = (ret.mean() * 252 - 0.045) / vol if vol != 0 else 0
    return {
        "Volatilidad anual": f"{vol*100:.2f}%",
        "VaR 95%": f"{var95*100:.2f}%",
        "Máx. Drawdown": f"{dd*100:.2f}%",
        "Sharpe": f"{sharpe_r:.2f}"
    }

tabla_comp = {}
for nombre, tkr in ACTIVOS.items():
    m = metricas_activo(tkr, anios)
    if m:
        tabla_comp[tkr] = m

df_comparativo = pd.DataFrame(tabla_comp).T
st.dataframe(df_comparativo, use_container_width=True)

st.markdown("""
**Conclusión del análisis:**

- **PFE (Pfizer):** acción farmacéutica del sector salud, considerada defensiva.
  Presenta volatilidad moderada y drawdowns contenidos por la estabilidad de la
  demanda del sector. Es el activo más estable del portafolio.
- **SHIB-USD (Shiba Inu):** criptomoneda meme con la mayor volatilidad y los
  mayores drawdowns del portafolio. Representa el activo de mayor riesgo, con
  colas muy pesadas en su distribución de retornos.
- **CL=F (Crude Oil WTI):** materia prima influida por factores geopolíticos y
  ciclos económicos. Volatilidad media-alta con drawdowns importantes en
  periodos de recesión.

Según los indicadores (volatilidad anualizada, VaR y máximo drawdown),
**SHIB-USD es el activo con mayor riesgo**, seguido de CL=F y finalmente PFE
como el más conservador. Un portafolio diversificado entre los tres permitiría
balancear rentabilidad esperada y riesgo.
""")