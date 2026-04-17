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
    page_icon="📈",
    layout="wide"
)

# ---------- Estilos oscuros ----------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #050b16 0%, #0b1220 100%);
            color: #f8fafc;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 1.2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
        }

        section[data-testid="stSidebar"] {
            background: #060d18;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .main-title {
            font-size: 3rem;
            font-weight: 800;
            color: #f9fafb;
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
            letter-spacing: -0.03em;
            line-height: 1.1;
        }

        .subtitle {
            color: #cbd5e1;
            font-size: 1.08rem;
            margin-bottom: 0.95rem;
            line-height: 1.5;
        }

        .market-chip {
            display: inline-block;
            padding: 0.38rem 0.78rem;
            margin-right: 0.45rem;
            margin-bottom: 0.4rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            color: #e5e7eb;
            font-size: 0.90rem;
        }

        .section-card {
            background: rgba(12, 19, 34, 0.88);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }

        .small-note {
            color: #d1d5db;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .report-box {
            background: rgba(10, 16, 28, 0.95);
            border: 1px solid rgba(34, 197, 94, 0.22);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            color: #f8fafc;
            line-height: 1.7;
            font-size: 1rem;
        }

        div[data-testid="metric-container"] {
            background: rgba(4, 10, 20, 0.82);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 1rem 1rem;
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }

        div[data-testid="metric-container"] label {
            color: #cbd5e1 !important;
            opacity: 1 !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            opacity: 1 !important;
        }

        div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
            color: #e5e7eb !important;
            opacity: 1 !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: inherit;
        }

        hr {
            border-color: rgba(255,255,255,0.08);
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
    var_col = f"VaR {int(nivel_confianza * 100)}% (%)"
    temp = df_comp.copy()

    temp["score_vol"] = temp["Volatilidad anual (%)"].rank(ascending=False, method="min")
    temp["score_var"] = temp[var_col].rank(ascending=True, method="min")
    temp["score_dd"] = temp["Máx. Drawdown (%)"].rank(ascending=True, method="min")

    temp["score_total"] = temp["score_vol"] + temp["score_var"] + temp["score_dd"]
    peor = temp.sort_values("score_total", ascending=True).iloc[0]["Activo"]
    return peor


def layout_mercado(titulo: str, alto: int = 400) -> dict:
    return dict(
        title=dict(
            text=titulo,
            font=dict(size=22, color="#f8fafc")
        ),
        height=alto,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0a1222",
        font=dict(color="#f8fafc", size=14),
        margin=dict(l=20, r=20, t=65, b=20),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title_font=dict(color="#e5e7eb"),
            tickfont=dict(color="#e5e7eb")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            title_font=dict(color="#e5e7eb"),
            tickfont=dict(color="#e5e7eb")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#f8fafc")
        )
    )


# ============================================================
# ENCABEZADO
# ============================================================
st.markdown(
    """
    <div class="main-title">Valoración de Riesgos Financieros</div>
    <div class="subtitle">
        Dashboard interactivo con enfoque de mercado para evaluar volatilidad, VaR, drawdown
        y desempeño ajustado por riesgo.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="market-chip">PFE · Equity</span>
    <span class="market-chip">SHIB-USD · Crypto</span>
    <span class="market-chip">CL=F · Commodity</span>
    """,
    unsafe_allow_html=True
)

st.caption("Alumna: Alejandra Martínez Cuen · Actividad 2.1.1 y 2.1.2")
st.divider()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Parámetros de mercado")

activo_seleccionado = st.sidebar.selectbox("Activo", list(ACTIVOS.keys()))
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
st.sidebar.caption("Datos históricos obtenidos desde Yahoo Finance.")


# ============================================================
# DESCARGA Y CÁLCULO
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
        f"{abs(var_param) * 100:.2f}%, por encima del umbral de {umbral_var * 100:.2f}%."
    )
elif abs(var_param) > umbral_var * 0.7:
    st.warning(
        f"Riesgo moderado — El VaR {int(nivel_confianza * 100)}% es de "
        f"{abs(var_param) * 100:.2f}% y se acerca al umbral configurado."
    )
else:
    st.success(
        f"Riesgo controlado — El VaR {int(nivel_confianza * 100)}% es de "
        f"{abs(var_param) * 100:.2f}% y permanece dentro del rango aceptable."
    )


# ============================================================
# MÉTRICAS
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Indicadores clave")

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
    % de días por debajo del VaR: <b>{pct_cola * 100:.2f}%</b>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# COMPARATIVA PRECIO VS FRECUENCIA
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
        line=dict(color="#22c55e", width=2.6)
    ))
    fig_precio.update_layout(**layout_mercado("Evolución del precio", 390))
    fig_precio.update_xaxes(title="Fecha")
    fig_precio.update_yaxes(title="Precio de cierre")
    st.plotly_chart(fig_precio, use_container_width=True)

with col_der:
    fig_hist_small = go.Figure()
    fig_hist_small.add_trace(go.Histogram(
        x=retornos,
        nbinsx=50,
        name="Frecuencia",
        marker_color="#38bdf8",
        opacity=0.82
    ))
    fig_hist_small.add_vrect(
        x0=retornos.min(),
        x1=var_param,
        fillcolor="rgba(239, 68, 68, 0.18)",
        line_width=0,
        layer="below"
    )
    fig_hist_small.add_vline(
        x=var_param,
        line_width=3,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"VaR {int(nivel_confianza * 100)}%",
        annotation_position="top left"
    )
    fig_hist_small.update_layout(**layout_mercado("Frecuencia de retornos diarios", 390))
    fig_hist_small.update_xaxes(title="Retorno diario")
    fig_hist_small.update_yaxes(title="Frecuencia")
    st.plotly_chart(fig_hist_small, use_container_width=True)

st.markdown(
    """
    <div class="small-note">
    Esta sección compara visualmente el comportamiento del precio frente a la frecuencia
    de sus retornos, tal como solicita la actividad.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# HISTOGRAMA PRINCIPAL
# ============================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Distribución de retornos diarios")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=retornos,
    nbinsx=60,
    name="Retornos diarios",
    marker_color="#00c2ff",
    opacity=0.80,
    hovertemplate="Retorno: %{x:.2%}<br>Frecuencia: %{y}<extra></extra>"
))

fig_hist.add_vrect(
    x0=retornos.min(),
    x1=var_param,
    fillcolor="rgba(239, 68, 68, 0.20)",
    line_width=0,
    layer="below",
    annotation_text="Zona de pérdida extrema",
    annotation_position="top left"
)

fig_hist.add_vline(
    x=var_param,
    line_width=3,
    line_dash="dash",
    line_color="#ff3b30",
    annotation_text=f"VaR {int(nivel_confianza * 100)}% = {var_param * 100:.2f}%",
    annotation_position="top left"
)

fig_hist.add_vline(
    x=retornos.mean(),
    line_width=2,
    line_dash="dot",
    line_color="#22c55e",
    annotation_text=f"Media = {retornos.mean() * 100:.3f}%",
    annotation_position="top right"
)

fig_hist.update_layout(**layout_mercado("Distribución y línea de control VaR", 450))
fig_hist.update_xaxes(title="Retorno diario", tickformat=".0%")
fig_hist.update_yaxes(title="Frecuencia")
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
    "El histograma muestra la distribución de retornos diarios, marca claramente la línea del VaR en rojo y destaca la cola izquierda."
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
    line=dict(color="#f97316", width=2.3),
    fillcolor="rgba(249, 115, 22, 0.18)"
))
fig_dd.update_layout(**layout_mercado("Profundidad de caídas del activo", 400))
fig_dd.update_xaxes(title="Fecha")
fig_dd.update_yaxes(title="Drawdown (%)")
st.plotly_chart(fig_dd, use_container_width=True)

st.markdown(
    """
    <div class="small-note">
    El drawdown complementa el análisis de riesgo porque muestra la severidad de las caídas acumuladas desde máximos previos.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TABLA COMPARATIVA
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

    var_col = f"VaR {int(nivel_confianza * 100)}% (%)"

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=df_comp["Activo"],
        y=df_comp["Volatilidad anual (%)"],
        name="Volatilidad anual",
        marker_color="#38bdf8"
    ))
    fig_comp.add_trace(go.Bar(
        x=df_comp["Activo"],
        y=np.abs(df_comp[var_col]),
        name=f"|VaR {int(nivel_confianza * 100)}%|",
        marker_color="#ef4444"
    ))
    fig_comp.add_trace(go.Bar(
        x=df_comp["Activo"],
        y=np.abs(df_comp["Máx. Drawdown (%)"]),
        name="|Máx. Drawdown|",
        marker_color="#f59e0b"
    ))

    fig_comp.update_layout(**layout_mercado("Comparación de indicadores de riesgo", 420))
    fig_comp.update_layout(barmode="group")
    fig_comp.update_xaxes(title="Activo")
    fig_comp.update_yaxes(title="Porcentaje (%)")
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# REPORTE
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

    reporte_html = f"""
    <div class="report-box">
        <p>
            En el periodo seleccionado de <b>{anios} año(s)</b>, el análisis confirma diferencias importantes entre los tres activos.
            <b>{activo_mas_vol}</b> presenta la mayor volatilidad anualizada, mientras que <b>{activo_peor_var}</b> muestra
            el VaR más severo bajo el nivel de confianza elegido. A su vez, <b>{activo_peor_dd}</b> registra el drawdown más profundo.
        </p>

        <p>
            Considerando de forma conjunta la volatilidad, el VaR y el máximo drawdown, el activo con <b>mayor riesgo</b>
            en esta ejecución es <b>{activo_mas_riesgoso}</b>. Este activo presenta una volatilidad anual de
            <b>{fila_riesgo['Volatilidad anual (%)']:.2f}%</b>, un VaR {int(nivel_confianza * 100)}% de
            <b>{fila_riesgo[var_col]:.2f}%</b> y un drawdown máximo de
            <b>{fila_riesgo['Máx. Drawdown (%)']:.2f}%</b>.
        </p>

        <p>
            Por otra parte, el activo relativamente más estable fue <b>{fila_estable['Activo']}</b>,
            con una volatilidad anual de <b>{fila_estable['Volatilidad anual (%)']:.2f}%</b>.
            En términos de retorno ajustado por riesgo, el mejor ratio de Sharpe corresponde a
            <b>{activo_mejor_sharpe}</b>.
        </p>
    </div>
    """

    st.markdown(reporte_html, unsafe_allow_html=True)
else:
    st.info("No fue posible generar el reporte comparativo.")

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PIE
# ============================================================
st.divider()
st.caption(
    "Fuente de datos: Yahoo Finance mediante yfinance. Esta app tiene fines académicos y no constituye recomendación de inversión."
)