"""
MinePlanner PERT-CPM — Aplicativo de planificación y optimización de
operaciones mineras (subterránea y superficial).

Ejecutar con:
    streamlit run app.py

Autor: Generado para planificación minera profesional.
Arquitectura modular: core/ (cálculo), viz/ (gráficos), io_utils/ (E/S y
validación), views/ (interfaz por módulo).
"""
from __future__ import annotations

import streamlit as st

from io_utils import state
from views import components as C
from views import general, actividades, analisis, optimizacion, mineria, simulacion, dashboard

st.set_page_config(
    page_title="MinePlanner PERT-CPM",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Menú lateral: (etiqueta, función)
MENU = {
    "🏠 Inicio": general.inicio,
    "⚙️ Configuración del proyecto": general.configuracion,
    "📋 Registro de actividades": actividades.registro,
    "📊 Análisis PERT": analisis.analisis_pert,
    "🧮 Análisis CPM": analisis.analisis_cpm,
    "🎯 Ruta crítica": analisis.ruta_critica,
    "⚡ Optimización y Crashing": optimizacion.optimizacion,
    "🏗️ Minería subterránea": mineria.subterranea,
    "🚜 Minería superficial": mineria.superficial,
    "🎲 Simulación Monte Carlo": simulacion.simulacion,
    "📈 Dashboard ejecutivo": dashboard.dashboard,
    "📤 Exportar resultados": general.exportar,
}


def main():
    state.init_state()
    C.inject_css()

    with st.sidebar:
        st.markdown("## ⛏️ MinePlanner")
        st.caption("PERT · CPM · Optimización minera")
        cfg = state.get_config()
        st.markdown(f"**Proyecto:** {cfg['nombre']}")
        st.markdown(f"**Operación:** {cfg['tipo']}")
        st.divider()
        seleccion = st.radio("Navegación", list(MENU.keys()), label_visibility="collapsed")
        st.divider()
        try:
            df = state.get_actividades()
            from core import cpm
            _, ruta, dur, _ = cpm.compute_cpm(df)
            st.metric("Duración estimada", f"{dur:g}")
            st.metric("Actividades", len(df))
            st.caption(f"Ruta crítica: {' → '.join(ruta)}")
        except Exception:
            st.caption("Complete el registro de actividades para ver métricas.")
        st.divider()
        st.caption("© MinePlanner — Ingeniería de Minas")

    # Renderizar la vista seleccionada
    MENU[seleccion]()


if __name__ == "__main__":
    main()
