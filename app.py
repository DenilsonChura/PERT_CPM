"""
MinePlanner PERT-CPM — Aplicativo profesional de planificación, programación y
optimización de operaciones mineras (subterránea y superficial) mediante las
metodologías PERT y CPM.

Archivo único, sin dependencias de módulos locales. Ejecutar con:
    streamlit run app.py

Contenido:
  - Modelo de datos y validación
  - Cálculo PERT y CPM (ruta crítica, holguras)
  - Optimización / Crashing (OR-Tools) y curva tiempo-costo
  - Simulación Monte Carlo (scipy) y simulación de acarreo (SimPy)
  - Plantillas mineras e indicadores (KPIs)
  - Visualización de red (NetworkX + Plotly) y diagrama de Gantt
  - Exportación a Excel, CSV y PDF
  - Interfaz Streamlit con 12 módulos
"""
from __future__ import annotations

import io
import math
import string
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="MinePlanner PERT-CPM",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# 1. MODELO DE DATOS
# ===========================================================================
COL_ID = "ID"
COL_NOMBRE = "Actividad"
COL_PRED = "Predecesoras"
COL_O = "Optimista"
COL_M = "Mas_probable"
COL_P = "Pesimista"
COL_DUR = "Duracion_CPM"
COL_COSTO = "Costo"
COL_REC = "Recursos"

COLUMNAS = [COL_ID, COL_NOMBRE, COL_PRED, COL_O, COL_M, COL_P, COL_DUR, COL_COSTO, COL_REC]

TOL = 1e-6  # tolerancia para holgura ~ 0


@dataclass
class Actividad:
    """Representación tipada de una actividad del proyecto."""
    id: str
    nombre: str = ""
    predecesoras: List[str] = field(default_factory=list)
    optimista: float = 0.0
    mas_probable: float = 0.0
    pesimista: float = 0.0
    duracion_cpm: Optional[float] = None
    costo: float = 0.0
    recursos: str = ""


def parse_predecesoras(valor) -> List[str]:
    """Convierte el texto de predecesoras en una lista limpia de IDs."""
    if valor is None:
        return []
    if isinstance(valor, (list, tuple)):
        items = list(valor)
    else:
        texto = str(valor).strip()
        if not texto or texto.lower() in {"nan", "none", "-"}:
            return []
        texto = texto.replace(";", ",").replace("|", ",")
        items = texto.split() if "," not in texto else texto.split(",")
    return [str(x).strip() for x in items if str(x).strip()]


def _num(valor, defecto: float = 0.0) -> float:
    """Conversión robusta a float."""
    try:
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            return defecto
        return float(valor)
    except (TypeError, ValueError):
        return defecto


def normalizar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el DataFrame de actividades para el motor de cálculo."""
    df = df.copy()
    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = None

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df = df[df[COL_ID].notna() & (df[COL_ID] != "") & (df[COL_ID].str.lower() != "nan")]
    df = df.drop_duplicates(subset=[COL_ID], keep="first").reset_index(drop=True)

    for col in (COL_O, COL_M, COL_P, COL_DUR, COL_COSTO):
        df[col] = df[col].apply(_num)

    df["_preds"] = df[COL_PRED].apply(parse_predecesoras)
    df["_te"] = (df[COL_O] + 4 * df[COL_M] + df[COL_P]) / 6.0

    def _dur_efectiva(row):
        d = row[COL_DUR]
        return float(d) if d and d > 0 else float(row["_te"])

    df["_dur"] = df.apply(_dur_efectiva, axis=1)
    return df


def df_vacio() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNAS})


# ===========================================================================
# 2. PERT
# ===========================================================================
def tiempo_esperado(o: float, m: float, p: float) -> float:
    """TE = (O + 4M + P) / 6."""
    return (o + 4.0 * m + p) / 6.0


def varianza(o: float, p: float) -> float:
    """Var = ((P - O) / 6)^2."""
    return ((p - o) / 6.0) ** 2


def calcular_pert(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla PERT por actividad (TE, varianza, desviación)."""
    dfn = normalizar_df(df)
    filas = []
    for _, r in dfn.iterrows():
        te = tiempo_esperado(r[COL_O], r[COL_M], r[COL_P])
        var = varianza(r[COL_O], r[COL_P])
        filas.append({
            COL_ID: r[COL_ID], COL_NOMBRE: r[COL_NOMBRE],
            COL_O: r[COL_O], COL_M: r[COL_M], COL_P: r[COL_P],
            "Tiempo_esperado": round(te, 3),
            "Varianza": round(var, 3),
            "Desviacion_estandar": round(math.sqrt(var), 3),
        })
    return pd.DataFrame(filas)


def estadisticos_ruta(df_pert: pd.DataFrame, ruta_critica_ids: List[str]) -> dict:
    """Media y desviación del proyecto sumando sobre la ruta crítica."""
    sub = df_pert[df_pert[COL_ID].isin(ruta_critica_ids)]
    media = float(sub["Tiempo_esperado"].sum())
    var_total = float(sub["Varianza"].sum())
    return {
        "media": round(media, 3),
        "varianza": round(var_total, 3),
        "desviacion": round(math.sqrt(var_total), 3) if var_total >= 0 else 0.0,
    }


# ===========================================================================
# 3. CPM
# ===========================================================================
def construir_grafo(dfn: pd.DataFrame) -> nx.DiGraph:
    """DiGraph Activity-on-Node a partir del DataFrame normalizado."""
    G = nx.DiGraph()
    for _, r in dfn.iterrows():
        G.add_node(r[COL_ID], duration=float(r["_dur"]), nombre=str(r[COL_NOMBRE]))
    ids = set(dfn[COL_ID])
    for _, r in dfn.iterrows():
        for p in r["_preds"]:
            if p in ids and p != r[COL_ID]:
                G.add_edge(p, r[COL_ID])
    return G


def _primer_ciclo(G: nx.DiGraph) -> List[str]:
    try:
        ciclo = nx.find_cycle(G, orientation="original")
        nodos = [u for u, _v, _d in ciclo]
        nodos.append(ciclo[-1][1])
        return nodos
    except nx.NetworkXNoCycle:
        return []


def compute_cpm(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], float, nx.DiGraph]:
    """Análisis CPM: pases adelante/atrás, holguras y ruta crítica.

    Devuelve (tabla, ruta_critica, duracion_proyecto, grafo).
    Lanza ValueError si hay dependencias circulares.
    """
    dfn = normalizar_df(df)
    G = construir_grafo(dfn)

    if not nx.is_directed_acyclic_graph(G):
        ciclo = _primer_ciclo(G)
        raise ValueError(f"Dependencias circulares detectadas: {' -> '.join(ciclo)}")

    orden = list(nx.topological_sort(G))

    ES: Dict[str, float] = {}
    EF: Dict[str, float] = {}
    for n in orden:
        preds = list(G.predecessors(n))
        ES[n] = max((EF[p] for p in preds), default=0.0)
        EF[n] = ES[n] + G.nodes[n]["duration"]

    duracion_proyecto = max(EF.values(), default=0.0)

    LS: Dict[str, float] = {}
    LF: Dict[str, float] = {}
    for n in reversed(orden):
        succ = list(G.successors(n))
        LF[n] = min((LS[s] for s in succ), default=duracion_proyecto)
        LS[n] = LF[n] - G.nodes[n]["duration"]

    filas = []
    for n in orden:
        succ = list(G.successors(n))
        holgura_total = LS[n] - ES[n]
        if succ:
            holgura_libre = min(ES[s] for s in succ) - EF[n]
        else:
            holgura_libre = duracion_proyecto - EF[n]
        critica = abs(holgura_total) <= TOL
        filas.append({
            COL_ID: n, COL_NOMBRE: G.nodes[n]["nombre"],
            "Duracion": round(G.nodes[n]["duration"], 3),
            "Inicio_temprano": round(ES[n], 3),
            "Fin_temprano": round(EF[n], 3),
            "Inicio_tardio": round(LS[n], 3),
            "Fin_tardio": round(LF[n], 3),
            "Holgura_total": round(holgura_total, 3),
            "Holgura_libre": round(max(holgura_libre, 0.0), 3),
            "Critica": critica,
        })

    tabla = pd.DataFrame(filas)
    ruta = _ruta_critica(G, ES, EF, LS)

    for n in G.nodes:
        G.nodes[n].update(ES=ES[n], EF=EF[n], LS=LS[n], LF=LF[n],
                          critica=(abs((LS[n] - ES[n])) <= TOL))

    return tabla, ruta, round(duracion_proyecto, 3), G


def _ruta_critica(G: nx.DiGraph, ES: Dict[str, float], EF: Dict[str, float],
                  LS: Dict[str, float]) -> List[str]:
    """Reconstruye una ruta crítica por retroceso desde el nodo de mayor EF."""
    if not G.nodes:
        return []
    fin = max(EF, key=EF.get)
    ruta = [fin]
    actual = fin
    while True:
        preds = list(G.predecessors(actual))
        criticos = [p for p in preds
                    if abs(EF[p] - ES[actual]) <= TOL and abs((LS[p] - ES[p])) <= TOL]
        if not criticos:
            break
        siguiente = max(criticos, key=EF.get)
        ruta.append(siguiente)
        actual = siguiente
    return list(reversed(ruta))


# ===========================================================================
# 4. VALIDACIÓN
# ===========================================================================
def validar(df: pd.DataFrame) -> List[str]:
    """Lista de errores/advertencias del modelo (vacía si todo ok)."""
    errores: List[str] = []
    dfn = normalizar_df(df)
    if dfn.empty:
        return ["No hay actividades registradas."]

    ids_orig = df[COL_ID].astype(str).str.strip() if COL_ID in df.columns else pd.Series([], dtype=str)
    dups = ids_orig[ids_orig.duplicated() & (ids_orig != "")].unique()
    if len(dups):
        errores.append(f"IDs duplicados: {', '.join(map(str, dups))}")

    ids = set(dfn[COL_ID])
    for _, r in dfn.iterrows():
        faltantes = [p for p in r["_preds"] if p not in ids]
        if faltantes:
            errores.append(f"Actividad {r[COL_ID]}: predecesora(s) inexistente(s) {', '.join(faltantes)}")
    for _, r in dfn.iterrows():
        if r[COL_ID] in r["_preds"]:
            errores.append(f"Actividad {r[COL_ID]} se referencia a sí misma.")
    for _, r in dfn.iterrows():
        o, m, p = r[COL_O], r[COL_M], r[COL_P]
        if not (o <= m <= p):
            errores.append(f"Actividad {r[COL_ID]}: se esperaba O ≤ M ≤ P (O={o}, M={m}, P={p}).")

    G = nx.DiGraph()
    G.add_nodes_from(ids)
    for _, r in dfn.iterrows():
        for p in r["_preds"]:
            if p in ids:
                G.add_edge(p, r[COL_ID])
    if not nx.is_directed_acyclic_graph(G):
        try:
            ciclo = nx.find_cycle(G)
            ruta = " -> ".join([u for u, _v in ciclo] + [ciclo[-1][1]])
            errores.append(f"Dependencia circular detectada: {ruta}")
        except nx.NetworkXNoCycle:
            errores.append("Dependencia circular detectada.")
    return errores


# ===========================================================================
# 5. CRASHING (OPTIMIZACIÓN TIEMPO-COSTO)
# ===========================================================================
def costo_marginal(dur_normal: float, dur_minima: float,
                   costo_normal: float, costo_acelerado: float) -> Optional[float]:
    reduccion = dur_normal - dur_minima
    if reduccion <= 0:
        return None
    return (costo_acelerado - costo_normal) / reduccion


def crash_preparar(df_crash: pd.DataFrame) -> pd.DataFrame:
    """Calcula pendiente y reducción máxima por actividad."""
    df = df_crash.copy()
    slopes, max_red = [], []
    for _, r in df.iterrows():
        s = costo_marginal(float(r["Duracion_normal"]), float(r["Duracion_minima"]),
                           float(r["Costo_normal"]), float(r["Costo_acelerado"]))
        slopes.append(s)
        max_red.append(max(float(r["Duracion_normal"]) - float(r["Duracion_minima"]), 0.0))
    df["Costo_marginal"] = slopes
    df["Reduccion_maxima"] = max_red
    return df


def crash_candidatas(df_crash: pd.DataFrame, ruta_ids: List[str]) -> pd.DataFrame:
    df = crash_preparar(df_crash)
    df = df[df[COL_ID].isin(ruta_ids)]
    df = df[df["Costo_marginal"].notna() & (df["Reduccion_maxima"] > 0)]
    return df.sort_values("Costo_marginal").reset_index(drop=True)


def _grafo_desde_crash(df_crash: pd.DataFrame, duraciones: Dict[str, float]) -> nx.DiGraph:
    G = nx.DiGraph()
    ids = set(df_crash[COL_ID].astype(str))
    for _, r in df_crash.iterrows():
        G.add_node(str(r[COL_ID]), duration=float(duraciones[str(r[COL_ID])]))
    for _, r in df_crash.iterrows():
        for p in parse_predecesoras(r.get("Predecesoras")):
            if p in ids:
                G.add_edge(p, str(r[COL_ID]))
    return G


def _duracion_red(G: nx.DiGraph) -> float:
    orden = list(nx.topological_sort(G))
    EF: Dict[str, float] = {}
    for n in orden:
        es = max((EF[p] for p in G.predecessors(n)), default=0.0)
        EF[n] = es + G.nodes[n]["duration"]
    return max(EF.values(), default=0.0)


def crash_duracion_minima(df: pd.DataFrame) -> float:
    dmin = {str(r[COL_ID]): float(r["Duracion_minima"]) for _, r in df.iterrows()}
    return _duracion_red(_grafo_desde_crash(df, dmin))


def crash_lp(df_crash: pd.DataFrame, duracion_objetivo: float) -> dict:
    """Optimización exacta con OR-Tools (GLOP): mínimo costo para el objetivo."""
    from ortools.linear_solver import pywraplp

    df = crash_preparar(df_crash)
    ids = [str(x) for x in df[COL_ID].tolist()]
    dur_normal = {str(r[COL_ID]): float(r["Duracion_normal"]) for _, r in df.iterrows()}
    max_red = {str(r[COL_ID]): float(r["Reduccion_maxima"]) for _, r in df.iterrows()}
    slope = {str(r[COL_ID]): (r["Costo_marginal"] or 0.0) for _, r in df.iterrows()}
    preds = {str(r[COL_ID]): parse_predecesoras(r.get("Predecesoras")) for _, r in df.iterrows()}

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {"estado": "sin_solver", "detalle": "GLOP no disponible"}

    INF = solver.infinity()
    x = {i: solver.NumVar(0.0, max_red[i], f"x_{i}") for i in ids}
    s = {i: solver.NumVar(0.0, INF, f"s_{i}") for i in ids}
    T = solver.NumVar(0.0, INF, "T")

    for i in ids:
        di = dur_normal[i] - x[i]
        for p in preds[i]:
            if p in ids:
                solver.Add(s[i] >= s[p] + (dur_normal[p] - x[p]))
        solver.Add(T >= s[i] + di)
    solver.Add(T <= duracion_objetivo)
    solver.Minimize(solver.Sum(slope[i] * x[i] for i in ids))
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {"estado": "infactible",
                "detalle": "No se puede alcanzar la duración objetivo con las reducciones disponibles.",
                "duracion_minima_posible": round(crash_duracion_minima(df), 3)}

    reducciones = {i: round(x[i].solution_value(), 3) for i in ids}
    costo_incremental = round(solver.Objective().Value(), 2)
    filas = []
    for _, r in df.iterrows():
        i = str(r[COL_ID])
        filas.append({
            COL_ID: i, COL_NOMBRE: r.get(COL_NOMBRE, i),
            "Duracion_normal": dur_normal[i], "Reduccion": reducciones[i],
            "Duracion_final": round(dur_normal[i] - reducciones[i], 3),
            "Costo_marginal": slope[i],
            "Costo_incremental": round(slope[i] * reducciones[i], 2),
        })
    return {"estado": "optimo", "costo_incremental": costo_incremental,
            "duracion_lograda": round(T.solution_value(), 3),
            "tabla": pd.DataFrame(filas)}


def crash_heuristico(df_crash: pd.DataFrame, pasos: int = 999) -> pd.DataFrame:
    """Curva tiempo-costo reduciendo iterativamente la crítica más barata."""
    df = crash_preparar(df_crash)
    dur = {str(r[COL_ID]): float(r["Duracion_normal"]) for _, r in df.iterrows()}
    dmin = {str(r[COL_ID]): float(r["Duracion_minima"]) for _, r in df.iterrows()}
    slope = {str(r[COL_ID]): (r["Costo_marginal"] or float("inf")) for _, r in df.iterrows()}
    costo_base = float(df["Costo_normal"].sum())

    def estado_red():
        G = _grafo_desde_crash(df, dur)
        orden = list(nx.topological_sort(G))
        ES, EF, LS, LF = {}, {}, {}, {}
        for n in orden:
            ES[n] = max((EF[p] for p in G.predecessors(n)), default=0.0)
            EF[n] = ES[n] + G.nodes[n]["duration"]
        Dproj = max(EF.values(), default=0.0)
        for n in reversed(orden):
            LF[n] = min((LS[s] for s in G.successors(n)), default=Dproj)
            LS[n] = LF[n] - G.nodes[n]["duration"]
        criticos = [n for n in orden if abs((LS[n] - ES[n])) <= TOL]
        return criticos, Dproj

    registros = []
    criticos, Dproj = estado_red()
    costo_acum = costo_base
    registros.append({"Iteracion": 0, "Duracion": round(Dproj, 3),
                      "Costo_total": round(costo_acum, 2), "Actividad_reducida": "-"})

    for it in range(1, pasos + 1):
        cand = [n for n in criticos if dur[n] - dmin[n] > TOL and slope[n] != float("inf")]
        if not cand:
            break
        elegida = min(cand, key=lambda n: slope[n])
        dur[elegida] = max(dur[elegida] - 1.0, dmin[elegida])
        costo_acum += slope[elegida]
        criticos, Dproj = estado_red()
        registros.append({"Iteracion": it, "Duracion": round(Dproj, 3),
                          "Costo_total": round(costo_acum, 2), "Actividad_reducida": elegida})
    return pd.DataFrame(registros)


# ===========================================================================
# 6. SIMULACIÓN MONTE CARLO
# ===========================================================================
def _muestrear(o: float, m: float, p: float, n: int, dist: str, rng) -> np.ndarray:
    o, m, p = float(o), float(m), float(p)
    if p <= o:
        return np.full(n, m if m > 0 else o)
    if dist == "beta":
        alpha = 1 + 4 * (m - o) / (p - o)
        beta_ = 1 + 4 * (p - m) / (p - o)
        return o + stats.beta.rvs(alpha, beta_, size=n, random_state=rng) * (p - o)
    c = (m - o) / (p - o) if p > o else 0.5
    c = min(max(c, 1e-6), 1 - 1e-6)
    return stats.triang.rvs(c, loc=o, scale=(p - o), size=n, random_state=rng)


def montecarlo_simular(df: pd.DataFrame, n_iter: int = 10000, dist: str = "triangular",
                       objetivo: float | None = None, semilla: int = 42) -> dict:
    """Simulación Monte Carlo de la duración del proyecto."""
    dfn = normalizar_df(df)
    rng = np.random.default_rng(semilla)

    muestras: Dict[str, np.ndarray] = {}
    for _, r in dfn.iterrows():
        muestras[r[COL_ID]] = _muestrear(r[COL_O], r[COL_M], r[COL_P], n_iter, dist, rng)

    G = construir_grafo(dfn)
    orden = list(nx.topological_sort(G))
    EF = {n: np.zeros(n_iter) for n in orden}
    for n in orden:
        preds = list(G.predecessors(n))
        es = np.maximum.reduce([EF[p] for p in preds]) if preds else np.zeros(n_iter)
        EF[n] = es + muestras[n]

    duraciones = np.maximum.reduce([EF[n] for n in orden]) if orden else np.zeros(n_iter)
    p50, p80, p90 = np.percentile(duraciones, [50, 80, 90])
    prob = float(np.mean(duraciones <= objetivo)) if objetivo is not None else None

    return {"duraciones": duraciones, "media": float(np.mean(duraciones)),
            "desviacion": float(np.std(duraciones)), "min": float(np.min(duraciones)),
            "max": float(np.max(duraciones)), "P50": float(p50), "P80": float(p80),
            "P90": float(p90), "objetivo": objetivo, "prob_cumplimiento": prob,
            "n_iter": n_iter, "dist": dist}


# ===========================================================================
# 7. MINERÍA: PLANTILLAS, KPIs Y SIMPY
# ===========================================================================
def _fila(id_, nombre, pred, o, m, p, costo, rec) -> Dict:
    return {COL_ID: id_, COL_NOMBRE: nombre, COL_PRED: pred, COL_O: o, COL_M: m,
            COL_P: p, COL_DUR: 0, COL_COSTO: costo, COL_REC: rec}


PLANTILLA_SUBTERRANEA: List[Dict] = [
    _fila("A", "Topografía y marcado", "", 0.3, 0.5, 1.0, 800, "Topógrafo"),
    _fila("B", "Perforación de frente", "A", 1.5, 2.0, 3.5, 3500, "Jumbo"),
    _fila("C", "Carguío de explosivos", "B", 0.5, 0.8, 1.5, 1200, "Cuadrilla"),
    _fila("D", "Voladura", "C", 0.2, 0.3, 0.5, 900, "Explosivos"),
    _fila("E", "Ventilación", "D", 0.4, 0.6, 1.2, 400, "Ventiladores"),
    _fila("F", "Desatado de rocas", "E", 0.5, 0.8, 1.5, 700, "Scaler"),
    _fila("G", "Sostenimiento (pernos/shotcrete)", "F", 1.0, 1.5, 3.0, 4200, "Robot shotcrete"),
    _fila("H", "Limpieza (mucking)", "F", 1.0, 1.8, 3.0, 2600, "Scoop"),
    _fila("I", "Transporte a superficie", "H", 0.8, 1.2, 2.5, 2200, "Camión bajo perfil"),
    _fila("J", "Preparación de tajo", "G,I", 2.0, 3.0, 5.0, 5000, "Cuadrilla preparación"),
]

PLANTILLA_SUPERFICIAL: List[Dict] = [
    _fila("A", "Desbroce y limpieza de banco", "", 1.0, 1.5, 3.0, 4000, "Tractor D10"),
    _fila("B", "Perforación de bancos", "A", 2.0, 3.0, 5.0, 12000, "Perforadora DTH"),
    _fila("C", "Carguío de explosivos", "B", 0.5, 1.0, 2.0, 6000, "Camión fábrica"),
    _fila("D", "Voladura", "C", 0.2, 0.3, 0.6, 8000, "Explosivos"),
    _fila("E", "Carguío de mineral", "D", 2.0, 3.5, 6.0, 15000, "Pala/Excavadora"),
    _fila("F", "Transporte a chancadora", "E", 2.0, 3.0, 5.0, 18000, "Flota camiones"),
    _fila("G", "Transporte a botadero", "E", 1.5, 2.5, 4.0, 9000, "Flota camiones"),
    _fila("H", "Chancado primario", "F", 1.5, 2.0, 3.5, 7000, "Chancadora"),
    _fila("I", "Gestión de stockpile", "H", 0.5, 1.0, 2.0, 2500, "Cargador frontal"),
    _fila("J", "Conformación de botadero", "G", 1.0, 1.5, 3.0, 3000, "Tractor"),
]


def plantilla_df(tipo: str) -> pd.DataFrame:
    datos = PLANTILLA_SUBTERRANEA if tipo == "subterranea" else PLANTILLA_SUPERFICIAL
    return pd.DataFrame(datos, columns=COLUMNAS)


def kpis_subterranea(metros_perforados, guardias, avance_guardia, disponibilidad,
                     utilizacion, toneladas, equipos, dias) -> Dict[str, float]:
    dias = max(dias, 1e-9); guardias = max(guardias, 1e-9); equipos = max(equipos, 1e-9)
    return {
        "m/dia": round(metros_perforados / dias, 2),
        "t/dia": round(toneladas / dias, 2),
        "t/guardia": round(toneladas / guardias, 2),
        "avance_m_guardia": round(avance_guardia, 2),
        "productividad_equipo_t": round(toneladas / equipos, 2),
        "disponibilidad_%": round(disponibilidad * 100, 1),
        "utilizacion_%": round(utilizacion * 100, 1),
        "efectividad_%": round(disponibilidad * utilizacion * 100, 1),
    }


def kpis_superficial(bcm, toneladas, horas, camiones, distancia_km, velocidad,
                     tiempo_ciclo_min, disponibilidad, utilizacion) -> Dict[str, float]:
    horas = max(horas, 1e-9); camiones = max(camiones, 1e-9)
    ciclos_hora = 60.0 / tiempo_ciclo_min if tiempo_ciclo_min > 0 else 0.0
    return {
        "BCM/h": round(bcm / horas, 2),
        "t/h": round(toneladas / horas, 2),
        "t/camion": round(toneladas / camiones, 2),
        "ciclos/h": round(ciclos_hora, 2),
        "productividad_flota_t/h": round((toneladas / horas) / camiones, 2),
        "disponibilidad_%": round(disponibilidad * 100, 1),
        "utilizacion_%": round(utilizacion * 100, 1),
        "efectividad_%": round(disponibilidad * utilizacion * 100, 1),
        "velocidad_km/h": round(velocidad, 1),
        "distancia_km": round(distancia_km, 2),
    }


def simular_ciclo_acarreo(n_camiones, capacidad_t, t_carga, t_viaje_ida, t_descarga,
                          t_viaje_vuelta, n_palas, horas_turno, semilla=7) -> Dict[str, float]:
    """Simulación SimPy del ciclo carguío-acarreo (tiempos en minutos)."""
    import random
    import simpy

    random.seed(semilla)
    horizonte = horas_turno * 60.0
    estado = {"ciclos": 0, "toneladas": 0.0}

    def camion(env, palas):
        while True:
            with palas.request() as req:
                yield req
                yield env.timeout(max(random.gauss(t_carga, t_carga * 0.1), 0.1))
            yield env.timeout(max(random.gauss(t_viaje_ida, t_viaje_ida * 0.15), 0.1))
            yield env.timeout(max(random.gauss(t_descarga, t_descarga * 0.1), 0.1))
            yield env.timeout(max(random.gauss(t_viaje_vuelta, t_viaje_vuelta * 0.15), 0.1))
            estado["ciclos"] += 1
            estado["toneladas"] += capacidad_t

    env = simpy.Environment()
    palas = simpy.Resource(env, capacity=max(int(n_palas), 1))
    for _ in range(max(int(n_camiones), 1)):
        env.process(camion(env, palas))
    env.run(until=horizonte)

    horas = max(horas_turno, 1e-9)
    return {"ciclos_totales": estado["ciclos"],
            "toneladas_turno": round(estado["toneladas"], 1),
            "t/h": round(estado["toneladas"] / horas, 2),
            "t/camion": round(estado["toneladas"] / max(n_camiones, 1), 1),
            "productividad_flota_t/h": round((estado["toneladas"] / horas) / max(n_camiones, 1), 2)}


# ===========================================================================
# 8. VISUALIZACIÓN (RED Y GANTT)
# ===========================================================================
def _layout_por_capas(G: nx.DiGraph) -> dict:
    nivel = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        nivel[n] = 0 if not preds else max(nivel[p] for p in preds) + 1
    capas: dict = {}
    for n, l in nivel.items():
        capas.setdefault(l, []).append(n)
    pos = {}
    for l, nodos in capas.items():
        k = len(nodos)
        for i, n in enumerate(sorted(nodos)):
            y = (k - 1) / 2.0 - i
            pos[n] = (l * 2.2, y * 1.6)
    return pos


def figura_red(G: nx.DiGraph, ruta: List[str]) -> go.Figure:
    pos = _layout_por_capas(G)
    criticas = set(zip(ruta[:-1], ruta[1:]))
    set_ruta = set(ruta)

    edge_traces = []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        es_crit = (u, v) in criticas
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(width=3 if es_crit else 1.4, color="#e63946" if es_crit else "#8d99ae"),
            hoverinfo="none", showlegend=False))

    node_x, node_y, textos, hover, colores, bordes = [], [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]; node_x.append(x); node_y.append(y)
        d = G.nodes[n]
        textos.append(f"<b>{n}</b><br>{d.get('duration', 0):.1f}")
        hover.append(f"<b>{n} — {d.get('nombre', '')}</b><br>"
                     f"Duración: {d.get('duration', 0):.2f}<br>"
                     f"ES: {d.get('ES', 0):.1f} | EF: {d.get('EF', 0):.1f}<br>"
                     f"LS: {d.get('LS', 0):.1f} | LF: {d.get('LF', 0):.1f}")
        es_crit = n in set_ruta
        colores.append("#e63946" if es_crit else "#457b9d")
        bordes.append("#9d0208" if es_crit else "#1d3557")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=textos,
        textposition="middle center", textfont=dict(color="white", size=11),
        hovertext=hover, hoverinfo="text",
        marker=dict(size=48, color=colores, line=dict(width=2.5, color=bordes), symbol="circle"),
        showlegend=False)

    fig = go.Figure(data=edge_traces + [node_trace])
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        es_crit = (u, v) in criticas
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.5,
                           arrowcolor="#e63946" if es_crit else "#8d99ae", opacity=0.9, standoff=26)

    fig.update_layout(title="Red PERT-CPM (ruta crítica en rojo)", showlegend=False,
                      hovermode="closest", margin=dict(l=20, r=20, t=50, b=20),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      template="plotly_dark", height=520)
    return fig


def construir_gantt(tabla_cpm: pd.DataFrame, fecha_inicio: datetime,
                    df_actividades: pd.DataFrame | None = None,
                    avance: dict | None = None) -> go.Figure:
    recursos_map = {}
    if df_actividades is not None and COL_ID in df_actividades.columns:
        for _, r in df_actividades.iterrows():
            recursos_map[str(r[COL_ID])] = str(r.get("Recursos", ""))

    filas = []
    for _, r in tabla_cpm.iterrows():
        ini = fecha_inicio + timedelta(days=float(r["Inicio_temprano"]))
        fin = fecha_inicio + timedelta(days=float(r["Fin_temprano"]))
        if fin <= ini:
            fin = ini + timedelta(hours=6)
        pct = avance.get(str(r[COL_ID]), 0) if avance else 0
        filas.append({"ID": r[COL_ID], "Actividad": f"{r[COL_ID]} — {r[COL_NOMBRE]}",
                      "Inicio": ini, "Fin": fin,
                      "Tipo": "Crítica" if r["Critica"] else "No crítica",
                      "Recursos": recursos_map.get(str(r[COL_ID]), ""),
                      "Avance": pct, "Holgura": r["Holgura_total"]})

    dfg = pd.DataFrame(filas)
    if dfg.empty:
        return go.Figure()
    dfg = dfg.sort_values("Inicio")

    fig = px.timeline(dfg, x_start="Inicio", x_end="Fin", y="Actividad", color="Tipo",
                      color_discrete_map={"Crítica": "#e63946", "No crítica": "#457b9d"},
                      hover_data={"Recursos": True, "Avance": True, "Holgura": True,
                                  "Inicio": True, "Fin": True, "Tipo": False})
    fig.update_yaxes(autorange="reversed")

    for _, row in dfg.iterrows():
        if row["Avance"] and row["Avance"] > 0:
            dur = (row["Fin"] - row["Inicio"])
            fin_avance = row["Inicio"] + dur * (row["Avance"] / 100.0)
            fig.add_shape(type="rect", xref="x", yref="y", x0=row["Inicio"], x1=fin_avance,
                          y0=row["Actividad"], y1=row["Actividad"],
                          fillcolor="rgba(46, 204, 113, 0.55)", line=dict(width=0), layer="above")

    fig.update_layout(title="Diagrama de Gantt (fechas tempranas)", template="plotly_dark",
                      height=max(360, 34 * len(dfg)), margin=dict(l=20, r=20, t=50, b=20),
                      legend_title_text="")
    return fig


# ===========================================================================
# 9. EXPORTACIÓN (EXCEL / CSV / PDF)
# ===========================================================================
def exportar_excel(hojas: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre, df in hojas.items():
            hoja = (nombre or "Hoja")[:31]
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(
                writer, sheet_name=hoja, index=False)
    return buffer.getvalue()


def exportar_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _estilo_tabla(compacto: bool = False):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle
    fs = 7 if compacto else 9
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#adb5bd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f3f5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])


def exportar_pdf(config: dict, resumen: Dict[str, str], tabla_cpm: pd.DataFrame,
                 ruta: List[str], kpis: Dict[str, str] | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            leftMargin=1.5 * cm, rightMargin=1.5 * cm)
    estilos = getSampleStyleSheet()
    titulo = ParagraphStyle("Titulo", parent=estilos["Title"], fontSize=18,
                            textColor=colors.HexColor("#1d3557"))
    h2 = ParagraphStyle("H2", parent=estilos["Heading2"], textColor=colors.HexColor("#457b9d"))
    normal = estilos["Normal"]

    elems = [Paragraph("Reporte Ejecutivo — Planificación Minera PERT-CPM", titulo),
             Spacer(1, 0.3 * cm),
             Paragraph(f"Proyecto: <b>{config.get('nombre', 'Sin nombre')}</b> &nbsp;|&nbsp; "
                       f"Operación: <b>{config.get('tipo', '-')}</b> &nbsp;|&nbsp; "
                       f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", normal),
             Spacer(1, 0.4 * cm), Paragraph("Resumen del proyecto", h2)]

    t = Table([["Indicador", "Valor"]] + [[k, v] for k, v in resumen.items()],
              colWidths=[8 * cm, 8 * cm])
    t.setStyle(_estilo_tabla()); elems += [t, Spacer(1, 0.4 * cm)]

    if kpis:
        elems.append(Paragraph("Indicadores operativos (KPIs)", h2))
        tk = Table([["KPI", "Valor"]] + [[k, v] for k, v in kpis.items()],
                   colWidths=[8 * cm, 8 * cm])
        tk.setStyle(_estilo_tabla()); elems += [tk, Spacer(1, 0.4 * cm)]

    elems += [Paragraph("Ruta crítica", h2),
              Paragraph(" → ".join(ruta) if ruta else "N/D", normal),
              Spacer(1, 0.4 * cm), Paragraph("Análisis CPM por actividad", h2)]

    cols = [c for c in ["ID", "Actividad", "Duracion", "Inicio_temprano", "Fin_temprano",
                        "Holgura_total", "Critica"] if c in tabla_cpm.columns]
    filas = [["ID", "Actividad", "Dur", "ES", "EF", "H.Total", "Crítica"]]
    for _, r in tabla_cpm.iterrows():
        filas.append([str(r[c]) if c != "Critica" else ("Sí" if r[c] else "No") for c in cols])
    tc = Table(filas, repeatRows=1)
    tc.setStyle(_estilo_tabla(compacto=True))
    for i, (_, r) in enumerate(tabla_cpm.iterrows(), start=1):
        if r.get("Critica"):
            tc.setStyle(TableStyle([("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ffd6d6"))]))
    elems.append(tc)

    doc.build(elems)
    return buffer.getvalue()


# ===========================================================================
# 10. ESTADO DE SESIÓN
# ===========================================================================
def init_state():
    ss = st.session_state
    if "config" not in ss:
        ss.config = {"nombre": "Proyecto Minero Demo", "tipo": "Subterránea",
                     "horizonte": "Semanal", "turnos": 2, "horas_turno": 8.0,
                     "fecha_inicio": date.today(), "fecha_objetivo": date.today() + timedelta(days=30)}
    if "actividades" not in ss:
        ss.actividades = plantilla_df("subterranea")
    if "avance" not in ss:
        ss.avance = {}
    # Nonce del editor: al cambiar, Streamlit reconstruye el Data Editor desde cero.
    # Necesario al reemplazar el dataset (plantillas / importación), de lo contrario
    # el editor reaplica ediciones antiguas sobre los datos nuevos.
    if "_editor_nonce" not in ss:
        ss._editor_nonce = 0
    # Firma del último archivo importado: evita reimportar en cada recarga.
    if "_ultimo_import" not in ss:
        ss._ultimo_import = None


def _bump_editor():
    """Fuerza la reconstrucción del Data Editor tras reemplazar el dataset."""
    st.session_state._editor_nonce = st.session_state.get("_editor_nonce", 0) + 1


def cargar_plantilla(tipo: str, etiqueta: str):
    """Carga una plantilla interna y resetea el estado dependiente."""
    set_actividades(plantilla_df(tipo))
    st.session_state.config["tipo"] = etiqueta
    st.session_state.avance = {}
    _bump_editor()


def get_actividades() -> pd.DataFrame:
    return st.session_state.actividades


def set_actividades(df: pd.DataFrame):
    st.session_state.actividades = df


def get_config() -> dict:
    return st.session_state.config


# ===========================================================================
# 11. COMPONENTES DE INTERFAZ
# ===========================================================================
CSS = """
<style>
  .kpi-card { background: linear-gradient(135deg, #1d3557 0%, #2a4d6e 100%);
      border-radius: 14px; padding: 16px 18px; color: #fff;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25); border: 1px solid #ffffff14; }
  .kpi-card .label { font-size: 0.78rem; opacity: 0.8; letter-spacing: .4px;
      text-transform: uppercase; }
  .kpi-card .value { font-size: 1.7rem; font-weight: 700; margin-top: 4px; }
  .kpi-card .delta { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }
  .kpi-crit { background: linear-gradient(135deg, #9d0208 0%, #d00000 100%); }
  .kpi-ok   { background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%); }
  .kpi-warn { background: linear-gradient(135deg, #7f5539 0%, #b08968 100%); }
  .section-title { border-left: 5px solid #e63946; padding-left: 12px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi_card(col, label: str, value, delta: str = "", tipo: str = ""):
    clase = {"crit": "kpi-crit", "ok": "kpi-ok", "warn": "kpi-warn"}.get(tipo, "")
    col.markdown(f"""<div class="kpi-card {clase}">
        <div class="label">{label}</div><div class="value">{value}</div>
        <div class="delta">{delta}</div></div>""", unsafe_allow_html=True)


def cabecera(titulo: str, subtitulo: str = ""):
    st.markdown(f"<h2 class='section-title'>{titulo}</h2>", unsafe_allow_html=True)
    if subtitulo:
        st.caption(subtitulo)


def guard_actividades() -> bool:
    df = get_actividades()
    if df is None or df.empty:
        st.warning("No hay actividades registradas. Vaya a **Registro de actividades**.")
        return False
    errores = validar(df)
    if errores:
        st.error("Se encontraron problemas en el modelo de actividades:")
        for e in errores:
            st.markdown(f"- {e}")
        return False
    return True


# ===========================================================================
# 12. VISTAS
# ===========================================================================
def vista_inicio():
    cabecera("⛏️ MinePlanner PERT-CPM",
             "Planificación, programación y optimización de operaciones mineras")
    st.markdown("""
Bienvenido a **MinePlanner**, una plataforma profesional para la planificación de
operaciones de **minería subterránea** y **minería superficial (cielo abierto)**
mediante las metodologías **PERT** y **CPM**.

**¿Qué puede hacer con esta herramienta?**
- Modelar actividades y relaciones de precedencia.
- Calcular tiempos esperados (PERT) y la ruta crítica (CPM).
- Analizar holguras y optimizar la duración vía *crashing*.
- Evaluar el riesgo del cronograma con **simulación Monte Carlo**.
- Usar plantillas mineras y calcular KPIs operativos.
- Exportar reportes ejecutivos en **Excel, CSV y PDF**.
    """)

    df = get_actividades(); cfg = get_config()
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Proyecto", cfg["nombre"][:18], cfg["tipo"])
    kpi_card(c2, "Actividades", len(df))
    try:
        _, ruta, dur, _ = compute_cpm(df)
        kpi_card(c3, "Duración estimada", f"{dur:g}", "unidades de tiempo", tipo="ok")
        kpi_card(c4, "Actividades críticas", len(ruta), " → ".join(ruta), tipo="crit")
    except Exception:
        kpi_card(c3, "Duración estimada", "—")
        kpi_card(c4, "Actividades críticas", "—")

    st.divider(); st.subheader("Comenzar rápido")

    st.markdown("**Opción A — plantillas incluidas**")
    col1, col2 = st.columns(2)
    if col1.button("🏗️ Cargar plantilla — Subterránea", use_container_width=True):
        cargar_plantilla("subterranea", "Subterránea")
        st.success("Plantilla subterránea cargada."); st.rerun()
    if col2.button("🚜 Cargar plantilla — Superficial", use_container_width=True):
        cargar_plantilla("superficial", "Superficial")
        st.success("Plantilla superficial cargada."); st.rerun()

    st.markdown("**Opción B — subir su propia plantilla (.xlsx)**")
    widget_importar("inicio")


def vista_configuracion():
    cabecera("⚙️ Configuración del proyecto")
    cfg = get_config()
    c1, c2 = st.columns(2)
    cfg["nombre"] = c1.text_input("Nombre del proyecto", cfg["nombre"])
    cfg["tipo"] = c2.selectbox("Tipo de operación", ["Subterránea", "Superficial"],
                               index=0 if cfg["tipo"] == "Subterránea" else 1)
    c3, c4 = st.columns(2)
    cfg["horizonte"] = c3.selectbox("Horizonte de planificación",
                                    ["Diario", "Semanal", "Mensual", "Anual"],
                                    index=["Diario", "Semanal", "Mensual", "Anual"].index(cfg["horizonte"]))
    cfg["turnos"] = c4.number_input("Número de turnos", 1, 4, int(cfg["turnos"]))
    c5, c6 = st.columns(2)
    cfg["horas_turno"] = c5.number_input("Horas por turno", 1.0, 24.0, float(cfg["horas_turno"]), 0.5)
    cfg["fecha_inicio"] = c5.date_input("Fecha de inicio", cfg["fecha_inicio"])
    cfg["fecha_objetivo"] = c6.date_input("Fecha objetivo", cfg["fecha_objetivo"])
    c6.metric("Horas productivas por día", f"{cfg['turnos'] * cfg['horas_turno']:g} h")
    st.success("Configuración guardada automáticamente en la sesión.")


def _siguiente_id(df: pd.DataFrame) -> str:
    existentes = set(df[COL_ID].astype(str)) if COL_ID in df.columns else set()
    for letra in string.ascii_uppercase:
        if letra not in existentes:
            return letra
    i = 1
    while f"A{i}" in existentes:
        i += 1
    return f"A{i}"


def _mapear_columnas(df: pd.DataFrame) -> pd.DataFrame:
    alias = {"id": COL_ID, "actividad": COL_NOMBRE, "nombre": COL_NOMBRE,
             "predecesoras": COL_PRED, "predecesora": COL_PRED, "pred": COL_PRED,
             "optimista": COL_O, "o": COL_O, "mas probable": COL_M, "más probable": COL_M,
             "mas_probable": COL_M, "m": COL_M, "pesimista": COL_P, "p": COL_P,
             "duracion_cpm": COL_DUR, "duración cpm": COL_DUR, "duracion": COL_DUR,
             "costo": COL_COSTO, "recursos": COL_REC}
    ren = {c: alias[str(c).strip().lower()] for c in df.columns if str(c).strip().lower() in alias}
    df = df.rename(columns=ren)
    for c in COLUMNAS:
        if c not in df.columns:
            df[c] = 0 if c in (COL_O, COL_M, COL_P, COL_DUR, COL_COSTO) else ""
    return df[COLUMNAS]


def widget_importar(contexto: str) -> bool:
    """Cargador de plantillas .xlsx reutilizable.

    Importa el archivo UNA sola vez. El file_uploader de Streamlit sigue
    devolviendo el archivo en cada recarga del script, por lo que se compara una
    firma (nombre + tamaño) contra la última importación para no reimportar en
    bucle. No llama a st.rerun(): el Data Editor se reconstruye mediante el nonce.
    """
    st.markdown("##### 📥 Importar plantilla desde Excel")
    st.caption("Columnas esperadas: ID · Actividad · Predecesoras · Optimista · "
               "Mas_probable · Pesimista · Duracion_CPM · Costo · Recursos")
    archivo = st.file_uploader(
        "Archivo .xlsx", type=["xlsx"], key=f"uploader_{contexto}",
        label_visibility="collapsed")

    if archivo is None:
        return False

    firma = f"{archivo.name}|{archivo.size}"
    if st.session_state.get("_ultimo_import") == firma:
        st.caption(f"✔️ **{archivo.name}** ya está cargado "
                   f"({len(get_actividades())} actividades). "
                   "Pulse la ✕ del archivo para subir otro.")
        return False

    try:
        imp = _mapear_columnas(pd.read_excel(archivo))
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return False

    if imp.empty or imp[COL_ID].astype(str).str.strip().eq("").all():
        st.error("El archivo no contiene actividades legibles. "
                 "Verifique que la primera fila sean los encabezados.")
        return False

    set_actividades(imp)
    st.session_state._ultimo_import = firma
    st.session_state.avance = {}
    _bump_editor()
    st.success(f"✅ Importadas **{len(imp)}** actividades desde **{archivo.name}**.")

    errores = validar(imp)
    if errores:
        st.warning("Se importó, pero revise estas observaciones:")
        for e in errores[:5]:
            st.markdown(f"- {e}")
    return True


def vista_actividades():
    cabecera("📋 Registro de actividades",
             "Agregue, edite o elimine actividades y sus predecesoras")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])
    if c1.button("➕ Fila vacía", use_container_width=True):
        df = get_actividades().copy()
        nuevo_id = _siguiente_id(df)
        fila = {c: (nuevo_id if c == COL_ID else (0 if c in (COL_O, COL_M, COL_P, COL_DUR, COL_COSTO) else ""))
                for c in COLUMNAS}
        set_actividades(pd.concat([df, pd.DataFrame([fila])], ignore_index=True))
        _bump_editor(); st.rerun()
    if c2.button("🏗️ Plantilla subterránea", use_container_width=True):
        cargar_plantilla("subterranea", "Subterránea"); st.rerun()
    if c3.button("🚜 Plantilla superficial", use_container_width=True):
        cargar_plantilla("superficial", "Superficial"); st.rerun()
    if c4.button("🗑️ Vaciar tabla", use_container_width=True):
        set_actividades(df_vacio()); _bump_editor(); st.rerun()

    widget_importar("registro")

    st.divider()
    df = get_actividades().copy()
    for c in COLUMNAS:
        if c not in df.columns:
            df[c] = 0 if c in (COL_O, COL_M, COL_P, COL_DUR, COL_COSTO) else ""

    config_cols = {
        COL_ID: st.column_config.TextColumn("ID", required=True, width="small"),
        COL_NOMBRE: st.column_config.TextColumn("Actividad", width="large"),
        COL_PRED: st.column_config.TextColumn("Predecesoras", help="IDs separados por coma, p. ej. A,B"),
        COL_O: st.column_config.NumberColumn("Optimista (O)", min_value=0.0, format="%.2f"),
        COL_M: st.column_config.NumberColumn("Más probable (M)", min_value=0.0, format="%.2f"),
        COL_P: st.column_config.NumberColumn("Pesimista (P)", min_value=0.0, format="%.2f"),
        COL_DUR: st.column_config.NumberColumn("Duración CPM", min_value=0.0, format="%.2f",
                                               help="0 = usar tiempo esperado PERT"),
        COL_COSTO: st.column_config.NumberColumn("Costo", min_value=0.0, format="%.2f"),
        COL_REC: st.column_config.TextColumn("Recursos"),
    }
    editado = st.data_editor(df[COLUMNAS], num_rows="dynamic", use_container_width=True,
                             column_config=config_cols, height=420,
                             key=f"editor_actividades_{st.session_state.get('_editor_nonce', 0)}")
    set_actividades(editado)

    errores = validar(editado)
    if errores:
        st.error("⚠️ Problemas detectados:")
        for e in errores:
            st.markdown(f"- {e}")
    else:
        st.success(f"✅ Modelo válido — {len(editado)} actividades sin dependencias circulares.")

    with st.expander("📖 Ayuda de columnas"):
        st.markdown(
            "- **ID**: identificador único (A, B, 10, T-01...).\n"
            "- **Predecesoras**: IDs que deben terminar antes (relación fin-inicio).\n"
            "- **O / M / P**: estimaciones optimista, más probable y pesimista para PERT.\n"
            "- **Duración CPM**: si es 0, el sistema usa el tiempo esperado PERT.\n"
            "- **Costo / Recursos**: para el diagrama de Gantt y el análisis económico.")


def vista_pert():
    cabecera("📊 Análisis PERT", "Tiempo esperado, varianza y probabilidades")
    if not guard_actividades():
        return
    df = get_actividades()
    tabla = calcular_pert(df)
    _, ruta, dur, _ = compute_cpm(df)
    est = estadisticos_ruta(tabla, ruta)

    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Duración esperada (μ)", f"{est['media']:g}", "sobre ruta crítica", tipo="ok")
    kpi_card(c2, "Varianza del proyecto", f"{est['varianza']:g}")
    kpi_card(c3, "Desviación estándar (σ)", f"{est['desviacion']:g}")

    st.subheader("Resultados por actividad")
    st.dataframe(tabla.rename(columns={"Tiempo_esperado": "Tiempo esperado (TE)",
                                       "Desviacion_estandar": "Desv. estándar"}),
                 use_container_width=True, hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        fig = px.bar(tabla, x="ID", y="Tiempo_esperado", color="Varianza",
                     color_continuous_scale="Reds",
                     title="Histograma de duración esperada por actividad",
                     labels={"Tiempo_esperado": "Tiempo esperado"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        mu, sigma = est["media"], max(est["desviacion"], 1e-6)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        fig2 = go.Figure(go.Scatter(x=x, y=stats.norm.pdf(x, mu, sigma),
                                    fill="tozeroy", line=dict(color="#e63946")))
        fig2.add_vline(x=mu, line_dash="dash", line_color="white", annotation_text=f"μ={mu:g}")
        fig2.update_layout(title="Distribución de probabilidad del proyecto",
                           template="plotly_dark", height=380,
                           xaxis_title="Duración", yaxis_title="Densidad")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Curva acumulada de probabilidad (S-curve)")
    mu, sigma = est["media"], max(est["desviacion"], 1e-6)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    figc = go.Figure(go.Scatter(x=x, y=stats.norm.cdf(x, mu, sigma) * 100,
                                line=dict(color="#2a9d8f", width=3)))
    for pct, col in [(50, "#f4a261"), (80, "#e76f51"), (90, "#e63946")]:
        val = stats.norm.ppf(pct / 100, mu, sigma)
        figc.add_vline(x=val, line_dash="dot", line_color=col, annotation_text=f"P{pct}={val:.1f}")
    figc.update_layout(template="plotly_dark", height=380, xaxis_title="Duración",
                       yaxis_title="Probabilidad acumulada (%)")
    st.plotly_chart(figc, use_container_width=True)
    st.info(f"La duración del proyecto se modela como normal con media **{mu:g}** y "
            f"desviación **{sigma:g}** (TLC sobre la ruta crítica). Para un análisis más "
            "realista use **Simulación Monte Carlo**.")


def vista_cpm():
    cabecera("🧮 Análisis CPM", "Tiempos tempranos/tardíos, holguras y red")
    if not guard_actividades():
        return
    df = get_actividades()
    try:
        tabla, ruta, dur, G = compute_cpm(df)
    except ValueError as e:
        st.error(str(e)); return

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Duración del proyecto", f"{dur:g}", "unidades de tiempo", tipo="ok")
    kpi_card(c2, "Actividades", len(tabla))
    kpi_card(c3, "Actividades críticas", int(tabla["Critica"].sum()), tipo="crit")
    kpi_card(c4, "Holgura total acumulada", f"{tabla['Holgura_total'].sum():g}")

    st.subheader("Tabla CPM completa")
    styled = tabla.rename(columns={"Inicio_temprano": "IT (ES)", "Fin_temprano": "FT (EF)",
                                   "Inicio_tardio": "IL (LS)", "Fin_tardio": "FL (LF)",
                                   "Holgura_total": "H. Total", "Holgura_libre": "H. Libre"})

    def resaltar(row):
        return (["background-color: #7a1420; color: white"] * len(row)
                if row["Critica"] else [""] * len(row))

    st.dataframe(styled.style.apply(resaltar, axis=1), use_container_width=True, hide_index=True)
    st.subheader("Red PERT-CPM")
    st.plotly_chart(figura_red(G, ruta), use_container_width=True)


def vista_ruta_critica():
    cabecera("🎯 Ruta crítica", "Actividades sin holgura y diagrama de Gantt")
    if not guard_actividades():
        return
    df = get_actividades()
    tabla, ruta, dur, G = compute_cpm(df)
    cfg = get_config()

    st.markdown(f"### 🔴 Ruta crítica:  `{'  →  '.join(ruta)}`")
    col = st.columns(3)
    kpi_card(col[0], "Duración de la ruta crítica", f"{dur:g}", tipo="crit")
    kpi_card(col[1], "N.º de actividades críticas", len(ruta))
    kpi_card(col[2], "% del total", f"{100 * len(ruta) / max(len(tabla), 1):.0f}%")

    st.subheader("Actividades críticas")
    st.dataframe(tabla[tabla["Critica"]][["ID", "Actividad", "Duracion", "Inicio_temprano",
                                          "Fin_temprano", "Holgura_total"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Diagrama de Gantt")
    fecha = datetime.combine(cfg["fecha_inicio"], datetime.min.time())
    st.plotly_chart(construir_gantt(tabla, fecha, df, st.session_state.get("avance", {})),
                    use_container_width=True)

    with st.expander("Gestionar % de avance por actividad"):
        avance = st.session_state.get("avance", {})
        cols = st.columns(4)
        for i, (_, r) in enumerate(tabla.iterrows()):
            with cols[i % 4]:
                avance[str(r["ID"])] = st.slider(f"{r['ID']}", 0, 100,
                                                 int(avance.get(str(r["ID"]), 0)), key=f"av_{r['ID']}")
        st.session_state["avance"] = avance


def vista_optimizacion():
    cabecera("⚡ Optimización y Crashing",
             "Reduzca la duración al menor costo posible (tiempo-costo)")
    if not guard_actividades():
        return
    df = get_actividades()
    tabla, ruta, dur, G = compute_cpm(df)
    st.info(f"Duración normal del proyecto (CPM): **{dur:g}** unidades. "
            f"Ruta crítica: **{' → '.join(ruta)}**")

    st.subheader("1) Parámetros de aceleración por actividad")
    st.caption("Defina duración mínima y costos normal/acelerado. "
               "La duración normal se toma de la duración CPM/PERT actual.")
    pred_series = (df.set_index(COL_ID).reindex(tabla[COL_ID])[COL_PRED].values
                   if COL_PRED in df.columns else "")
    costo_series = (df.set_index(COL_ID).reindex(tabla[COL_ID])[COL_COSTO].fillna(0)
                    if COL_COSTO in df.columns else pd.Series([0.0] * len(tabla)))
    base = pd.DataFrame({
        COL_ID: tabla[COL_ID].values, COL_NOMBRE: tabla[COL_NOMBRE].values,
        "Predecesoras": pred_series,
        "Duracion_normal": tabla["Duracion"].values,
        "Duracion_minima": (tabla["Duracion"] * 0.6).round(1).values,
        "Costo_normal": np.array(costo_series),
        "Costo_acelerado": (np.array(costo_series) * 1.6).round(1),
    })

    editado = st.data_editor(base, use_container_width=True, hide_index=True, key="editor_crash",
        column_config={
            COL_ID: st.column_config.TextColumn("ID", disabled=True),
            COL_NOMBRE: st.column_config.TextColumn("Actividad", disabled=True),
            "Duracion_normal": st.column_config.NumberColumn("Dur. normal", format="%.1f"),
            "Duracion_minima": st.column_config.NumberColumn("Dur. mínima", format="%.1f"),
            "Costo_normal": st.column_config.NumberColumn("Costo normal", format="%.0f"),
            "Costo_acelerado": st.column_config.NumberColumn("Costo acelerado", format="%.0f"),
        })

    prep = crash_preparar(editado)
    st.subheader("2) Costo marginal y candidatas")
    cand = crash_candidatas(editado, ruta)
    cc1, cc2 = st.columns([1.4, 1])
    with cc1:
        st.markdown("**Costo marginal (pendiente) por actividad**")
        st.dataframe(prep[[COL_ID, COL_NOMBRE, "Duracion_normal", "Duracion_minima",
                           "Reduccion_maxima", "Costo_marginal"]],
                     use_container_width=True, hide_index=True)
    with cc2:
        st.markdown("**Candidatas críticas (menor costo primero)**")
        if cand.empty:
            st.warning("No hay actividades críticas reducibles.")
        else:
            st.dataframe(cand[[COL_ID, "Costo_marginal", "Reduccion_maxima"]],
                         use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3) Optimización a duración objetivo (OR-Tools)")
    dmin = crash_duracion_minima(prep)
    objetivo = st.slider("Duración objetivo del proyecto", float(round(dmin, 1)),
                         float(round(dur, 1)), float(round((dur + dmin) / 2, 1)), 0.5)
    if st.button("🚀 Optimizar (mínimo costo)", type="primary"):
        res = crash_lp(editado, objetivo)
        if res["estado"] != "optimo":
            st.error(f"No factible: {res.get('detalle', '')} "
                     f"(mínimo posible ≈ {res.get('duracion_minima_posible', dmin):g})")
        else:
            k1, k2, k3 = st.columns(3)
            kpi_card(k1, "Duración lograda", f"{res['duracion_lograda']:g}", f"antes: {dur:g}", tipo="ok")
            kpi_card(k2, "Reducción total", f"{dur - res['duracion_lograda']:g}", "unidades")
            kpi_card(k3, "Incremento de costo", f"+{res['costo_incremental']:,.0f}", tipo="warn")
            st.markdown("**Plan de aceleración**")
            st.dataframe(res["tabla"], use_container_width=True, hide_index=True)
            st.markdown("**Comparativa antes / después**")
            st.dataframe(pd.DataFrame({"Escenario": ["Normal", "Optimizado"],
                                       "Duración": [dur, res["duracion_lograda"]],
                                       "Costo aceleración": [0, res["costo_incremental"]]}),
                         use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4) Curva tiempo-costo completa (heurística iterativa)")
    if st.button("📈 Generar curva tiempo-costo"):
        curva = crash_heuristico(editado)
        fig = go.Figure(go.Scatter(x=curva["Duracion"], y=curva["Costo_total"],
                                   mode="lines+markers", line=dict(color="#e63946", width=3),
                                   name="Costo total"))
        fig.update_layout(template="plotly_dark", height=420, title="Curva tiempo-costo del proyecto",
                          xaxis_title="Duración del proyecto", yaxis_title="Costo total acumulado")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(curva, use_container_width=True, hide_index=True)


def _simulador_acarreo(contexto: str):
    st.divider()
    st.subheader("🔄 Simulación de ciclo de acarreo (SimPy)")
    st.caption("Simulación de eventos discretos: camiones compitiendo por palas/equipos "
               "de carguío durante un turno. Tiempos en minutos.")
    c1, c2, c3, c4 = st.columns(4)
    ncam = c1.number_input("Camiones", 1, 60, 6, key=f"sc_{contexto}_1")
    cap = c1.number_input("Capacidad (t)", 1.0, 400.0, 40.0, key=f"sc_{contexto}_2")
    tcarga = c2.number_input("T. carga", 0.5, 60.0, 4.0, key=f"sc_{contexto}_3")
    tida = c2.number_input("T. viaje ida", 0.5, 120.0, 8.0, key=f"sc_{contexto}_4")
    tdesc = c3.number_input("T. descarga", 0.5, 30.0, 2.0, key=f"sc_{contexto}_5")
    tvuelta = c3.number_input("T. viaje vuelta", 0.5, 120.0, 6.0, key=f"sc_{contexto}_6")
    npalas = c4.number_input("Palas/equipos carguío", 1, 20, 2, key=f"sc_{contexto}_7")
    horas = c4.number_input("Horas de turno", 1.0, 24.0, 10.0, key=f"sc_{contexto}_8")
    if st.button("▶️ Ejecutar simulación", key=f"run_{contexto}"):
        r = simular_ciclo_acarreo(int(ncam), cap, tcarga, tida, tdesc, tvuelta, int(npalas), horas)
        kc = st.columns(4)
        kpi_card(kc[0], "Ciclos completados", r["ciclos_totales"], tipo="ok")
        kpi_card(kc[1], "Producción del turno", f"{r['toneladas_turno']:,.0f} t")
        kpi_card(kc[2], "Ritmo", f"{r['t/h']} t/h")
        kpi_card(kc[3], "Prod. flota", f"{r['productividad_flota_t/h']} t/h·camión", tipo="warn")
        fig = go.Figure(go.Bar(x=["Ciclos", "t/h", "t/camión"],
                               y=[r["ciclos_totales"], r["t/h"], r["t/camion"]],
                               marker_color=["#457b9d", "#2a9d8f", "#e63946"]))
        fig.update_layout(template="plotly_dark", height=320,
                          title="Resultados de la simulación de acarreo")
        st.plotly_chart(fig, use_container_width=True)


def vista_subterranea():
    cabecera("🏗️ Minería subterránea",
             "Plantillas de labores, variables e indicadores operativos")
    st.subheader("Plantillas de labores")
    labores = ["Desarrollo horizontal", "Rampas", "Chimeneas", "Producción",
               "Sostenimiento", "Preparación de tajos"]
    st.write("Ciclo típico: perforación → carguío → voladura → ventilación → "
             "desatado → sostenimiento → limpieza → transporte.")
    cols = st.columns(3)
    for i, l in enumerate(labores):
        cols[i % 3].markdown(f"- **{l}**")
    if st.button("📥 Cargar plantilla de actividades subterránea", type="primary"):
        cargar_plantilla("subterranea", "Subterránea")
        st.success("Plantilla cargada. Vaya a **Análisis CPM** para ver resultados.")

    st.divider()
    st.subheader("Variables e indicadores")
    c1, c2, c3 = st.columns(3)
    metros = c1.number_input("Metros perforados", 0.0, 100000.0, 320.0)
    dias = c1.number_input("Días del periodo", 1.0, 365.0, 30.0)
    guardias = c2.number_input("N.º de guardias", 1.0, 1000.0, 60.0)
    avance = c2.number_input("Avance por guardia (m)", 0.0, 50.0, 3.2)
    toneladas = c3.number_input("Toneladas producidas", 0.0, 1e7, 45000.0)
    equipos = c3.number_input("N.º de equipos", 1.0, 100.0, 4.0)
    c4, c5 = st.columns(2)
    disp = c4.slider("Disponibilidad mecánica", 0.0, 1.0, 0.85)
    util = c5.slider("Utilización", 0.0, 1.0, 0.75)
    k = kpis_subterranea(metros, guardias, avance, disp, util, toneladas, equipos, dias)
    st.subheader("KPIs")
    kc = st.columns(4)
    kpi_card(kc[0], "Avance", f"{k['m/dia']} m/día", tipo="ok")
    kpi_card(kc[1], "Producción", f"{k['t/dia']} t/día")
    kpi_card(kc[2], "Por guardia", f"{k['t/guardia']} t")
    kpi_card(kc[3], "Efectividad", f"{k['efectividad_%']}%", "Disp × Util", tipo="warn")
    st.dataframe(pd.DataFrame(k.items(), columns=["Indicador", "Valor"]),
                 use_container_width=True, hide_index=True)
    _simulador_acarreo("subterranea")


def vista_superficial():
    cabecera("🚜 Minería superficial (cielo abierto)",
             "Plantillas de operaciones, variables e indicadores de flota")
    st.subheader("Plantillas de operaciones")
    ops = ["Perforación de bancos", "Voladura", "Carguío", "Transporte",
           "Chancado", "Botaderos", "Stockpile"]
    cols = st.columns(4)
    for i, o in enumerate(ops):
        cols[i % 4].markdown(f"- **{o}**")
    if st.button("📥 Cargar plantilla de actividades superficial", type="primary"):
        cargar_plantilla("superficial", "Superficial")
        st.success("Plantilla cargada. Vaya a **Análisis CPM** para ver resultados.")

    st.divider()
    st.subheader("Variables e indicadores de flota")
    c1, c2, c3 = st.columns(3)
    bcm = c1.number_input("BCM movidos", 0.0, 1e7, 50000.0)
    toneladas = c1.number_input("Toneladas movidas", 0.0, 1e7, 135000.0)
    horas = c2.number_input("Horas operativas", 1.0, 100000.0, 100.0)
    camiones = c2.number_input("N.º de camiones", 1.0, 200.0, 6.0)
    distancia = c3.number_input("Distancia de acarreo (km)", 0.0, 100.0, 3.5)
    velocidad = c3.number_input("Velocidad promedio (km/h)", 1.0, 80.0, 25.0)
    c4, c5, c6 = st.columns(3)
    ciclo = c4.number_input("Tiempo de ciclo (min)", 1.0, 240.0, 18.0)
    disp = c5.slider("Disponibilidad de flota", 0.0, 1.0, 0.85)
    util = c6.slider("Utilización de equipos", 0.0, 1.0, 0.80)
    k = kpis_superficial(bcm, toneladas, horas, camiones, distancia, velocidad, ciclo, disp, util)
    st.subheader("KPIs")
    kc = st.columns(4)
    kpi_card(kc[0], "Movimiento", f"{k['BCM/h']} BCM/h", tipo="ok")
    kpi_card(kc[1], "Producción", f"{k['t/h']} t/h")
    kpi_card(kc[2], "Por camión", f"{k['t/camion']} t")
    kpi_card(kc[3], "Efectividad", f"{k['efectividad_%']}%", "Disp × Util", tipo="warn")
    st.dataframe(pd.DataFrame(k.items(), columns=["Indicador", "Valor"]),
                 use_container_width=True, hide_index=True)
    _simulador_acarreo("superficial")


def vista_montecarlo():
    cabecera("🎲 Simulación Monte Carlo",
             "Riesgo del cronograma a partir de los tiempos PERT")
    if not guard_actividades():
        return
    df = get_actividades()
    _, ruta, dur_det, _ = compute_cpm(df)
    c1, c2, c3 = st.columns(3)
    n_iter = c1.select_slider("Iteraciones", [1000, 5000, 10000, 20000, 50000], value=10000)
    dist = c2.selectbox("Distribución", ["triangular", "beta"],
                        format_func=lambda x: "Triangular" if x == "triangular" else "Beta-PERT")
    objetivo = c3.number_input("Duración objetivo", 0.0, 1e6, float(round(dur_det, 1)), 1.0)

    if st.button("▶️ Ejecutar simulación", type="primary"):
        with st.spinner("Simulando..."):
            st.session_state["mc_res"] = montecarlo_simular(df, n_iter, dist, objetivo)

    res = st.session_state.get("mc_res")
    if not res:
        st.info("Configure los parámetros y ejecute la simulación."); return

    k = st.columns(4)
    kpi_card(k[0], "Media", f"{res['media']:.1f}", f"det.: {dur_det:g}", tipo="ok")
    kpi_card(k[1], "P80", f"{res['P80']:.1f}", "80% de confianza", tipo="warn")
    kpi_card(k[2], "P90", f"{res['P90']:.1f}", "90% de confianza", tipo="crit")
    prob = res["prob_cumplimiento"]
    kpi_card(k[3], "Prob. cumplir objetivo", f"{prob*100:.1f}%" if prob is not None else "—",
             tipo="ok" if (prob or 0) >= 0.8 else "warn")

    k2 = st.columns(3)
    kpi_card(k2[0], "P50 (mediana)", f"{res['P50']:.1f}")
    kpi_card(k2[1], "Mínimo", f"{res['min']:.1f}")
    kpi_card(k2[2], "Máximo", f"{res['max']:.1f}")

    dur = res["duraciones"]
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(go.Histogram(x=dur, nbinsx=50, marker_color="#457b9d"))
        for key, col in [("P50", "#f4a261"), ("P80", "#e76f51"), ("P90", "#e63946")]:
            fig.add_vline(x=res[key], line_dash="dash", line_color=col, annotation_text=key)
        if res["objetivo"]:
            fig.add_vline(x=res["objetivo"], line_color="#2a9d8f", annotation_text="Objetivo")
        fig.update_layout(template="plotly_dark", height=400,
                          title=f"Distribución de la duración ({res['n_iter']:,} iter.)",
                          xaxis_title="Duración", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        orden = np.sort(dur)
        acum = np.arange(1, len(orden) + 1) / len(orden) * 100
        figc = go.Figure(go.Scatter(x=orden, y=acum, line=dict(color="#2a9d8f", width=3)))
        for key, col in [("P50", "#f4a261"), ("P80", "#e76f51"), ("P90", "#e63946")]:
            figc.add_vline(x=res[key], line_dash="dot", line_color=col, annotation_text=key)
        if res["objetivo"]:
            figc.add_vline(x=res["objetivo"], line_color="white", annotation_text="Objetivo")
        figc.update_layout(template="plotly_dark", height=400, title="Curva acumulada (S-curve)",
                           xaxis_title="Duración", yaxis_title="Probabilidad acumulada (%)")
        st.plotly_chart(figc, use_container_width=True)

    interp = (f"Existe un **{prob*100:.1f}%** de probabilidad de completar el proyecto en "
              f"**{objetivo:g}** unidades o menos. " if prob is not None else "")
    st.info(interp + f"Para una planificación conservadora considere el **P80 = {res['P80']:.1f}** "
            f"o el **P90 = {res['P90']:.1f}** como fecha comprometida.")


def vista_dashboard():
    cabecera("📈 Dashboard ejecutivo", "Vista consolidada del proyecto para centros de control")
    if not guard_actividades():
        return
    df = get_actividades(); cfg = get_config()
    tabla, ruta, dur, G = compute_cpm(df)
    tpert = calcular_pert(df)
    est = estadisticos_ruta(tpert, ruta)

    dias_objetivo = (cfg["fecha_objetivo"] - cfg["fecha_inicio"]).days
    indice = (dias_objetivo / dur) if dur > 0 else 0
    costo_total = float(df["Costo"].sum()) if "Costo" in df.columns else 0.0
    avance = st.session_state.get("avance", {})
    avance_prom = (sum(avance.values()) / len(avance)) if avance else 0

    k = st.columns(4)
    kpi_card(k[0], "Duración del proyecto", f"{dur:g}", cfg["horizonte"], tipo="ok")
    kpi_card(k[1], "N.º de actividades", len(tabla))
    kpi_card(k[2], "Actividades críticas", int(tabla["Critica"].sum()), tipo="crit")
    kpi_card(k[3], "Índice de cumplimiento", f"{indice:.2f}", "≥1 dentro de plazo",
             tipo="ok" if indice >= 1 else "warn")
    k2 = st.columns(4)
    kpi_card(k2[0], "Costo acumulado", f"{costo_total:,.0f}")
    kpi_card(k2[1], "Avance promedio", f"{avance_prom:.0f}%", tipo="ok" if avance_prom >= 50 else "warn")
    kpi_card(k2[2], "Duración esperada (PERT)", f"{est['media']:g}")
    kpi_card(k2[3], "Desviación σ", f"{est['desviacion']:g}")

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Duración por actividad (barras)**")
        fig = px.bar(tabla, x="ID", y="Duracion", color="Critica",
                     color_discrete_map={True: "#e63946", False: "#457b9d"},
                     labels={"Duracion": "Duración"})
        fig.update_layout(template="plotly_dark", height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        st.markdown("**Distribución críticas vs no críticas (circular)**")
        n_crit = int(tabla["Critica"].sum())
        fig = go.Figure(go.Pie(labels=["Críticas", "No críticas"],
                               values=[n_crit, len(tabla) - n_crit], hole=0.5,
                               marker_colors=["#e63946", "#457b9d"]))
        fig.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Costo acumulado por secuencia (línea)**")
        seq = tabla.sort_values("Inicio_temprano").copy()
        costos = (df.set_index("ID").reindex(seq["ID"])["Costo"].fillna(0).values
                  if "Costo" in df.columns else [0] * len(seq))
        seq["Costo_acum"] = pd.Series(costos).cumsum().values
        fig = px.line(seq, x="ID", y="Costo_acum", markers=True)
        fig.update_traces(line_color="#2a9d8f")
        fig.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with g4:
        st.markdown("**Índice de cumplimiento (gauge)**")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=indice * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 150]},
                   "bar": {"color": "#2a9d8f" if indice >= 1 else "#e63946"},
                   "steps": [{"range": [0, 80], "color": "#5c1a1a"},
                             {"range": [80, 100], "color": "#7f5539"},
                             {"range": [100, 150], "color": "#1b4332"}],
                   "threshold": {"line": {"color": "white", "width": 3},
                                 "thickness": 0.75, "value": 100}},
            title={"text": "Plazo objetivo / Duración"}))
        fig.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Resumen ejecutivo**")
    st.dataframe(pd.DataFrame({
        "Indicador": ["Proyecto", "Operación", "Horizonte", "Ruta crítica",
                      "Duración (CPM)", "Duración esperada (PERT)", "Fecha objetivo"],
        "Valor": [cfg["nombre"], cfg["tipo"], cfg["horizonte"], " → ".join(ruta),
                  f"{dur:g}", f"{est['media']:g}", str(cfg["fecha_objetivo"])]}),
        use_container_width=True, hide_index=True)


def vista_exportar():
    cabecera("📤 Exportar resultados", "Descargue los análisis en Excel, CSV o PDF")
    if not guard_actividades():
        return
    df = get_actividades(); cfg = get_config()
    tabla_cpm, ruta, dur, G = compute_cpm(df)
    tabla_pert = calcular_pert(df)
    est = estadisticos_ruta(tabla_pert, ruta)
    tabla_ruta = tabla_cpm[tabla_cpm["Critica"]]

    resumen = {"Nombre del proyecto": cfg["nombre"], "Tipo de operación": cfg["tipo"],
               "Número de actividades": str(len(df)), "Duración del proyecto (CPM)": f"{dur:g}",
               "Duración esperada (PERT)": f"{est['media']:g}",
               "Desviación estándar (PERT)": f"{est['desviacion']:g}",
               "Actividades críticas": str(int(tabla_cpm['Critica'].sum())),
               "Ruta crítica": " → ".join(ruta)}

    st.subheader("Resumen que se exportará")
    st.table(pd.DataFrame(resumen.items(), columns=["Indicador", "Valor"]))

    hojas = {"PERT": tabla_pert, "CPM": tabla_cpm, "Ruta_critica": tabla_ruta,
             "Actividades": df,
             "Resumen": pd.DataFrame(resumen.items(), columns=["Indicador", "Valor"])}

    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Excel (.xlsx)", data=exportar_excel(hojas),
                       file_name=f"{cfg['nombre']}_reporte.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    tabla_sel = c2.selectbox("Tabla para CSV", list(hojas.keys()))
    c2.download_button("⬇️ CSV", data=exportar_csv(hojas[tabla_sel]),
                       file_name=f"{cfg['nombre']}_{tabla_sel}.csv", mime="text/csv",
                       use_container_width=True)
    try:
        pdf = exportar_pdf(cfg, resumen, tabla_cpm, ruta)
        c3.download_button("⬇️ PDF ejecutivo", data=pdf,
                           file_name=f"{cfg['nombre']}_ejecutivo.pdf", mime="application/pdf",
                           use_container_width=True)
    except Exception as e:
        c3.error(f"PDF no disponible: {e}")


# ===========================================================================
# 13. NAVEGACIÓN PRINCIPAL
# ===========================================================================
MENU = {
    "🏠 Inicio": vista_inicio,
    "⚙️ Configuración del proyecto": vista_configuracion,
    "📋 Registro de actividades": vista_actividades,
    "📊 Análisis PERT": vista_pert,
    "🧮 Análisis CPM": vista_cpm,
    "🎯 Ruta crítica": vista_ruta_critica,
    "⚡ Optimización y Crashing": vista_optimizacion,
    "🏗️ Minería subterránea": vista_subterranea,
    "🚜 Minería superficial": vista_superficial,
    "🎲 Simulación Monte Carlo": vista_montecarlo,
    "📈 Dashboard ejecutivo": vista_dashboard,
    "📤 Exportar resultados": vista_exportar,
}


def main():
    init_state()
    inject_css()
    with st.sidebar:
        st.markdown("## ⛏️ MinePlanner")
        st.caption("PERT · CPM · Optimización minera")
        cfg = get_config()
        st.markdown(f"**Proyecto:** {cfg['nombre']}")
        st.markdown(f"**Operación:** {cfg['tipo']}")
        st.divider()
        seleccion = st.radio("Navegación", list(MENU.keys()), label_visibility="collapsed")
        st.divider()
        try:
            df = get_actividades()
            _, ruta, dur, _ = compute_cpm(df)
            st.metric("Duración estimada", f"{dur:g}")
            st.metric("Actividades", len(df))
            st.caption(f"Ruta crítica: {' → '.join(ruta)}")
        except Exception:
            st.caption("Complete el registro de actividades para ver métricas.")
        st.divider()
        st.caption("© MinePlanner — Ingeniería de Minas")
    MENU[seleccion]()


if __name__ == "__main__":
    main()
