"""
App de Valoración de Riesgos Financieros
Alumna: Alejandra Martínez Cuen (#13)
Activos asignados: PFE, SHIB-USD, CL=F
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime, timedelta


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Valoración de Riesgos Financieros",
    page_icon="📊",
    layout="wide"
)

# ---------- Estilos ----------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.1rem;
        }

        .section-card {
            background: #ffffff;
            padding: 1rem 1.1rem;
            border-radius: 16px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            margin-bottom: 1rem;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }

        .report-box {
            background: #f8fafc;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 14px;
            padding: 1rem 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ACTIVOS ASIGNADOS
# ============================================================
ACTIVOS = {
    "PFE (Pfizer - Acción)": "PFE",
    "SHIB-USD (Shiba Inu - Criptomoneda)": "SHIB-USD",
    "CL=F (Crude Oil WTI - Materia Prima)": "CL=F"
}


# ============================================================
# FUNCIONES
# ============================================================
@st.cache_data(ttl=3600)
def obtener_historicos(ticker: str, anios: int) -> pd.DataFrame:
    """
    Descarga precios históricos ajustados del activo seleccionado.
    """
    fin = datetime.today()
    inicio = fin - timedelta(days=365 * anios)

    try:
        df = yf.download(
            ticker,
            start=inicio,
            end=fin,
            progress=False,
            auto_adjust=True,
            threads=False
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=f"{anios}y", auto_adjust=True)
        except Exception:
            return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            return pd.DataFrame()

    return df[["Close"]].dropna()


def calcular_metricas(df: pd.DataFrame, nivel_confianza: float, tasa_libre: float) -> dict:
    """
    Calcula métricas de riesgo principales.
    """
    datos = df.copy()
    datos["Retorno"] = np.log(datos["Close"] / datos["Close"].shift(1))
    datos = datos.dropna()

    retornos = datos["Retorno"]

    vol_diaria = retornos.std()
    vol_anual = vol_diaria * np.sqrt(252)

    z_score = stats.norm.ppf(1 - nivel_confianza)
    var_param = retornos.mean() + z_score * vol_diaria

    maximo_acum = datos["Close"].cummax()
    drawdown_serie = (datos["Close"] - maximo_acum) / maximo_acum
    max_drawdown = drawdown_serie.min()

    retorno_anual = retornos.mean() * 252
    sharpe = (retorno_anual - tasa_libre) / vol_anual if vol_anual != 0 else 0

    observaciones_cola = (retornos < var_param).sum()
    pct_cola = observaciones_cola / len(retornos) if len(retornos) > 0 else 0

    return {
        "datos": datos,
        "retornos": retornos,
        "vol_anual": vol_anual,
        "var_param": var_param,
        "drawdown_serie": drawdown_serie,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "retorno_anual": retorno_anual,
        "observaciones_cola": int(observaciones_cola),
        "pct_cola": pct_cola
    }


@st.cache_data(ttl=3600)
def tabla_comparativa(anios: int, nivel_confianza: float, tasa_libre: float) -> pd.DataFrame:
    """
    Genera tabla comparativa para los 3 activos.
    """
    registros = []

    for nombre, ticker in ACTIVOS.items():
        df = obtener_historicos(ticker, anios)
        if df.empty:
            continue

        met = calcular_metricas(df, nivel_confianza, tasa_libre)

        registros.append({
            "Activo": ticker,
            "Tipo": nombre.split(" - ")[-1].replace(")", ""),
            "Volatilidad anual (%)": met["vol_anual"] * 100,
            f"VaR {int(nivel_confianza * 100)}% (%)": met["var_param"] * 100,
            "Máx. Drawdown (%)": met["max_drawdown"] * 100,
            "Sharpe": met["sharpe"],
            "Retorno anual esperado (%)": met["retorno_anual"] * 100,
            "Días en cola izquierda": met["observaciones_cola"],
            "% días bajo VaR": met["pct_cola"] * 100
        })

    return pd.DataFrame(registros)


def detectar_mayor_riesgo(df_comp: pd.DataFrame, nivel_confianza: float) -> str:
    """
    Determina el activo más riesgoso con base en volatilidad, VaR y drawdown.
    """
    var_col = f"VaR {int(nivel_confianza * 100)}% (%)"
    temp = df_comp.copy()

    # Para VaR y Drawdown, más negativo = más riesgoso.
    temp["score_vol"] = temp["Volatilidad anual (%)"].rank(ascending=False, method="min")
    temp["score_var"] = temp[var_col].rank(ascending=True, method="min")
    temp["score_dd"] = temp["Máx. Drawdown (%)"].rank(ascending=True, method="min")

    temp["score_total"] = temp["score_vol"] + temp["score_var"] + temp["score_dd"]
    peor = temp.sort_values("score_total", ascending=True).iloc[0]["Activo"]
    return peor


# ============================================================
# ENCABEZADO
# ============================================================
st.markdown('<div class="main-title">Valoración de Riesgos Financieros</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Aplicación interactiva para evaluar volatilidad, VaR, drawdown y Sharpe de los activos asignados: PFE, SHIB-USD y CL=F.</div>',
    unsafe_allow_html=True
)
st.caption("Alumna: Alejandra Martínez Cuen · Actividad 2.1.1 y 2.1.2")
st.divider()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuración del análisis")

activo_seleccionado = st.sidebar.selectbox("Selecciona un activo", list(ACTIVOS.keys()))
ticker = ACTIVOS[activo_seleccionado]

anios = st.sidebar.slider("Años de historia", 1, 5, 1)

nivel_confianza = st.sidebar.radio(
    "Nivel de confianza del VaR",
    [0.90, 0.95, 0.99],
    index=1,
    format_func=lambda x: f"{int(x * 100)}%"
)

tasa_libre = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1
) / 100

umbral_var = st.sidebar.slider(
    "Umbral de alerta VaR (%)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5
) / 100

st.sidebar.markdown("---")
st.sidebar.info(
    "La app calcula indicadores de riesgo con datos históricos descargados desde Yahoo Finance."
)


# ============================================================
# DESCARGA Y CÁLCULO PRINCIPAL
# ============================================================
with st.spinner(f"Descargando datos de {ticker}..."):
    df_activo = obtener_historicos(ticker, anios)

if df_activo.empty:
    st.error("No se pudieron obtener datos del activo seleccionado.")
    st.stop()

met = calcular_metricas(df_activo, nivel_confianza, tasa_libre)
datos = met["datos"]
retornos = met["retornos"]
vol_anual = met["vol_anual"]
var_param = met["var_param"]
drawdown_serie = met["drawdown_serie"]
max_drawdown = met["max_drawdown"]
sharpe = met["sharpe"]
retorno_anual = met["retorno_anual"]
observaciones_cola = met["observaciones_cola"]
pct_cola = met["pct_cola"]


# ============================================================
# ALERTA DE RIESGO
# ============================================================
st.subheader(f"Resultados para {activo_seleccionado}")

if abs(var_param) > umbral_var:
    st.error(
        f"ALERTA DE RIESGO ALTO — El VaR {int(nivel_confianza * 100)}% es de "
        f"{abs(var_param) * 100:.2f}%, por encima del umbral definido de {umbral_var * 100:.2f}%. "
        f"Se recomienda considerar diversificación, coberturas, reducción de exposición o límites de pérdida."
    )
elif abs(var_param) > umbral_var * 0.7:
    st.warning(
        f"Riesgo moderado — El VaR {int(nivel_confianza * 100)}% es de "
        f"{abs(var_param) * 100:.2f}% y se encuentra cerca del umbral de alerta."
    )
else:
    st.success(
        f"Riesgo controlado — El VaR {int(nivel_confianza * 100)}% es de "
        f"{abs(var_param) * 100:.2f}% y se mantiene dentro del umbral aceptable."
    )


# ============================================================
# MÉTRICAS PRINCIPALES
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Indicadores clave de riesgo")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Volatilidad anualizada", f"{vol_anual * 100:.2f}%")
c2.metric(f"VaR {int(nivel_confianza * 100)}%", f"{var_param * 100:.2f}%")
c3.metric("Máximo drawdown", f"{max_drawdown * 100:.2f}%")
c4.metric("Ratio de Sharpe", f"{sharpe:.2f}")

st.markdown(
    f"""
    <div class="small-note">
    Retorno anual estimado: <b>{retorno_anual * 100:.2f}%</b> ·
    Días en cola izquierda: <b>{observaciones_cola}</b> ·
    Porcentaje de días por debajo del VaR: <b>{pct_cola * 100:.2f}%</b>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# COMPARATIVA: PRECIO VS FRECUENCIA DE RETORNOS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Comparativa visual: precio vs frecuencia de retornos")

col_izq, col_der = st.columns(2)

with col_izq:
    fig_precio = go.Figure()
    fig_precio.add_trace(go.Scatter(
        x=datos.index,
        y=datos["Close"],
        mode="lines",
        name="Precio de cierre",
        line=dict(color="#2563eb", width=2.5)
    ))
    fig_precio.update_layout(
        title="Evolución del precio",
        xaxis_title="Fecha",
        yaxis_title="Precio de cierre",
        height=390,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )
    st.plotly_chart(fig_precio, use_container_width=True)

with col_der:
    fig_hist_small = go.Figure()
    fig_hist_small.add_trace(go.Histogram(
        x=retornos,
        nbinsx=50,
        name="Frecuencia de retornos",
        marker_color="#7c3aed",
        opacity=0.82
    ))
    fig_hist_small.add_vline(
        x=var_param,
        line_width=3,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR {int(nivel_confianza * 100)}%",
        annotation_position="top left"
    )
    fig_hist_small.update_layout(
        title="Frecuencia de retornos diarios",
        xaxis_title="Retorno diario",
        yaxis_title="Frecuencia",
        height=390,
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_white"
    )
    st.plotly_chart(fig_hist_small, use_container_width=True)

st.markdown(
    """
    <div class="small-note">
    Esta comparación permite contrastar el comportamiento temporal del precio con la distribución estadística
    de sus retornos, identificando episodios de mayor dispersión y riesgo en la cola izquierda.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DISTRIBUCIÓN Y CONTROL VISUAL
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Distribución de retornos y línea de control VaR")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=retornos,
    nbinsx=60,
    name="Retornos diarios",
    marker_color="#6d28d9",
    opacity=0.78
))
fig_hist.add_vline(
    x=var_param,
    line_width=3,
    line_dash="dash",
    line_color="red",
    annotation_text=f"VaR {int(nivel_confianza * 100)}% = {var_param * 100:.2f}%",
    annotation_position="top left"
)
fig_hist.add_vline(
    x=retornos.mean(),
    line_width=2,
    line_dash="dot",
    line_color="green",
    annotation_text=f"Media = {retornos.mean() * 100:.3f}%",
    annotation_position="top right"
)

fig_hist.update_layout(
    xaxis_title="Retorno diario",
    yaxis_title="Frecuencia",
    height=450,
    bargap=0.05,
    template="plotly_white",
    margin=dict(l=20, r=20, t=50, b=20)
)
st.plotly_chart(fig_hist, use_container_width=True)

q01 = retornos.quantile(0.01)
q05 = retornos.quantile(0.05)
q95 = retornos.quantile(0.95)
q99 = retornos.quantile(0.99)

t1, t2, t3, t4 = st.columns(4)
t1.metric("Percentil 1%", f"{q01 * 100:.2f}%")
t2.metric("Percentil 5%", f"{q05 * 100:.2f}%")
t3.metric("Percentil 95%", f"{q95 * 100:.2f}%")
t4.metric("Percentil 99%", f"{q99 * 100:.2f}%")

st.caption(
    "La cola izquierda concentra los escenarios de pérdida extrema. El VaR funciona como línea de control para identificar si el activo presenta eventos severos con frecuencia relevante."
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DRAWDOWN
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Evolución del drawdown")

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown_serie.index,
    y=drawdown_serie * 100,
    mode="lines",
    fill="tozeroy",
    name="Drawdown",
    line=dict(color="crimson", width=2)
))
fig_dd.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Drawdown (%)",
    height=400,
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20)
)
st.plotly_chart(fig_dd, use_container_width=True)

st.markdown(
    """
    <div class="small-note">
    El drawdown muestra la caída acumulada desde el máximo histórico del periodo. Mientras más profundo sea,
    mayor es el esfuerzo requerido para recuperar el valor previo.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TABLA COMPARATIVA DE LOS 3 ACTIVOS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Análisis comparativo de los 3 activos")

df_comp = tabla_comparativa(anios, nivel_confianza, tasa_libre)

if df_comp.empty:
    st.warning("No fue posible construir la tabla comparativa.")
else:
    st.dataframe(
        df_comp.style.format({
            "Volatilidad anual (%)": "{:.2f}",
            f"VaR {int(nivel_confianza * 100)}% (%)": "{:.2f}",
            "Máx. Drawdown (%)": "{:.2f}",
            "Sharpe": "{:.2f}",
            "Retorno anual esperado (%)": "{:.2f}",
            "% días bajo VaR": "{:.2f}"
        }),
        use_container_width=True
    )

    # Gráfico comparativo
    var_col = f"VaR {int(nivel_confianza * 100)}% (%)"

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=df_comp["Activo"],
        y=df_comp["Volatilidad anual (%)"],
        name="Volatilidad anual (%)"
    ))
    fig_comp.add_trace(go.Bar(
        x=df_comp["Activo"],
        y=np.abs(df_comp[var_col]),
        name=f"|VaR {int(nivel_confianza * 100)}%| (%)"
    ))
    fig_comp.add_trace(go.Bar(
        x=df_comp["Activo"],
        y=np.abs(df_comp["Máx. Drawdown (%)"]),
        name="|Máx. Drawdown| (%)"
    ))

    fig_comp.update_layout(
        barmode="group",
        height=420,
        title="Comparación de indicadores de riesgo",
        xaxis_title="Activo",
        yaxis_title="Porcentaje (%)",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# REPORTE DE ANÁLISIS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Reporte breve de análisis")

if not df_comp.empty:
    var_col = f"VaR {int(nivel_confianza * 100)}% (%)"

    activo_mas_vol = df_comp.loc[df_comp["Volatilidad anual (%)"].idxmax(), "Activo"]
    activo_peor_var = df_comp.loc[df_comp[var_col].idxmin(), "Activo"]
    activo_peor_dd = df_comp.loc[df_comp["Máx. Drawdown (%)"].idxmin(), "Activo"]
    activo_mejor_sharpe = df_comp.loc[df_comp["Sharpe"].idxmax(), "Activo"]
    activo_mas_riesgoso = detectar_mayor_riesgo(df_comp, nivel_confianza)

    fila_riesgo = df_comp[df_comp["Activo"] == activo_mas_riesgoso].iloc[0]
    fila_estable = df_comp.sort_values("Volatilidad anual (%)", ascending=True).iloc[0]

    reporte = f"""
    En el periodo seleccionado de {anios} año(s), el análisis comparativo muestra diferencias claras entre los tres activos asignados.
    **{activo_mas_vol}** presenta la mayor volatilidad anualizada, mientras que **{activo_peor_var}** registra el VaR más severo al nivel de confianza elegido.
    Asimismo, **{activo_peor_dd}** exhibe el drawdown más profundo, lo que indica una mayor dificultad potencial de recuperación ante caídas fuertes.

    Considerando en conjunto la volatilidad, el VaR y el máximo drawdown, el activo con **mayor riesgo** en esta ejecución es **{activo_mas_riesgoso}**.
    En particular, este activo presenta una volatilidad anual de **{fila_riesgo['Volatilidad anual (%)']:.2f}%**, un
    VaR {int(nivel_confianza * 100)}% de **{fila_riesgo[var_col]:.2f}%** y un drawdown máximo de **{fila_riesgo['Máx. Drawdown (%)']:.2f}%**.

    Por otro lado, el activo relativamente más estable en términos de dispersión fue **{fila_estable['Activo']}**,
    con una volatilidad anual de **{fila_estable['Volatilidad anual (%)']:.2f}%**.
    En desempeño ajustado por riesgo, el mejor resultado de Sharpe corresponde a **{activo_mejor_sharpe}**.

    En conclusión, la comparación sugiere que los activos especulativos o altamente sensibles al mercado pueden generar colas más pesadas
    y pérdidas extremas más frecuentes, mientras que los activos más estables tienden a exhibir menor dispersión y drawdowns menos severos.
    """

    st.markdown(f'<div class="report-box">{reporte}</div>', unsafe_allow_html=True)
else:
    st.info("No fue posible generar el reporte comparativo.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()
st.caption(
    "Fuente de datos: Yahoo Finance mediante yfinance. "
    "Esta app tiene fines académicos y no constituye recomendación de inversión."
)