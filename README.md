# ⛏️ MinePlanner PERT-CPM

Aplicativo profesional en **Python + Streamlit** para la **planificación,
programación y optimización** de operaciones mineras (subterránea y superficial)
mediante las metodologías **PERT** y **CPM**.

Orientado a ingenieros de minas, planificadores, supervisores y gerentes de
operaciones.

---

## 🚀 Ejecución

```bash
# 1. Crear entorno (opcional pero recomendado)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`.

---

## 🧭 Módulos (menú lateral)

| # | Módulo | Descripción |
|---|--------|-------------|
| 1 | **Inicio** | Panel de bienvenida y carga rápida de plantillas. |
| 2 | **Configuración del proyecto** | Nombre, tipo de operación, horizonte, turnos, fechas. |
| 3 | **Registro de actividades** | Tabla editable (Data Editor), import Excel, validación de ciclos. |
| 4 | **Análisis PERT** | Tiempo esperado, varianza, σ, histograma y curvas de probabilidad. |
| 5 | **Análisis CPM** | ES/EF/LS/LF, holguras total y libre, red interactiva. |
| 6 | **Ruta crítica** | Actividades críticas y diagrama de Gantt con % de avance. |
| 7 | **Optimización y Crashing** | Costo marginal, optimización exacta (OR-Tools) y curva tiempo-costo. |
| 8 | **Minería subterránea** | Plantillas de labores, KPIs (m/día, t/guardia) y simulación de ciclo. |
| 9 | **Minería superficial** | Plantillas de operaciones, KPIs de flota (BCM/h, t/h) y simulación. |
| 10 | **Simulación Monte Carlo** | Distribución de la duración, P50/P80/P90 y probabilidad de cumplimiento. |
| 11 | **Dashboard ejecutivo** | Tarjetas KPI, barras, líneas, circular y gauge. |
| 12 | **Exportar resultados** | Descarga en Excel, CSV y PDF ejecutivo. |

---

## 🏗️ Arquitectura

```
mine_pert_cpm/
├── app.py                  # Entrypoint + navegación lateral
├── requirements.txt
├── README.md
├── core/                   # Lógica de cálculo (sin Streamlit, testeable)
│   ├── models.py           #   modelo de datos y normalización
│   ├── pert.py             #   TE, varianza, estadísticos
│   ├── cpm.py              #   pases adelante/atrás, holguras, ruta crítica
│   ├── crashing.py         #   costo marginal + LP (OR-Tools) + heurística
│   ├── montecarlo.py       #   simulación (triangular / Beta-PERT, scipy)
│   └── mining.py           #   plantillas, KPIs y simulación SimPy
├── viz/                    # Visualizaciones
│   ├── network_diagram.py  #   red PERT-CPM (NetworkX + Plotly)
│   └── gantt.py            #   diagrama de Gantt (Plotly)
├── io_utils/               # Entrada/salida y validación
│   ├── validation.py       #   ciclos, predecesoras, consistencia PERT
│   ├── exporters.py        #   Excel / CSV / PDF (reportlab)
│   └── state.py            #   estado de sesión
├── views/                  # Interfaz por módulo
│   ├── components.py       #   tarjetas KPI, estilos, guardas
│   ├── general.py          #   inicio, configuración, exportar
│   ├── actividades.py      #   registro de actividades
│   ├── analisis.py         #   PERT, CPM, ruta crítica
│   ├── optimizacion.py     #   crashing
│   ├── mineria.py          #   subterránea / superficial
│   ├── simulacion.py       #   Monte Carlo
│   └── dashboard.py        #   dashboard ejecutivo
├── data/                   # Datos de ejemplo (CSV)
└── tests/                  # Pruebas del núcleo de cálculo
    └── test_core.py
```

---

## 🧪 Pruebas

```bash
python tests/test_core.py
```

Verifican, contra un caso clásico conocido, el cálculo de PERT, la ruta crítica
(A-B-D-F = 14), las holguras, la optimización de crashing, la simulación Monte
Carlo, las plantillas mineras y la simulación SimPy de acarreo.

---

## 📐 Fundamento metodológico

**PERT**
- Tiempo esperado:  `TE = (O + 4M + P) / 6`
- Varianza:  `Var = ((P − O) / 6)²`

**CPM**
- Pase adelante: `ES = máx(EF predecesoras)`, `EF = ES + duración`
- Pase atrás: `LF = mín(LS sucesoras)`, `LS = LF − duración`
- Holgura total: `HT = LS − ES`.  Holgura libre: `HL = mín(ES sucesoras) − EF`
- **Ruta crítica**: actividades con holgura total = 0.

**Crashing (tiempo-costo)**
- Costo marginal (pendiente): `(Costo_acelerado − Costo_normal) / (Dur_normal − Dur_mínima)`
- Optimización exacta mediante programación lineal (solver **GLOP** de OR-Tools):
  minimiza el costo de aceleración sujeto a una duración objetivo.

---

## 📦 Librerías

`streamlit`, `pandas`, `numpy`, `networkx`, `plotly`, `matplotlib`, `scipy`,
`ortools`, `simpy`, `openpyxl`, `reportlab`.

---

## 💡 Notas de uso

- En **Registro de actividades**, la columna *Duración CPM = 0* indica al sistema
  que use automáticamente el tiempo esperado PERT.
- Las **plantillas mineras** cargan un proyecto completo listo para analizar.
- El **Gantt** usa fechas de calendario a partir de la *fecha de inicio* configurada.
- Para una fecha comprometida conservadora, use el **P80** o **P90** de la
  simulación Monte Carlo.
