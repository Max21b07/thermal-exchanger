"""Streamlit application for industrial batch cooling simulation.

PFD-style interface with block-based parameter entry.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import ProductBatch, CoolingWater, HeatExchanger, Architecture
from simulation import BatchCoolingSimulator, SimulationConfig, run_all_modes, create_comparison_dataframe

st.set_page_config(
    page_title="Industrial Batch Cooling Simulator",
    page_icon="🏭",
    layout="wide"
)


# Mode names
MODE_NAMES = {
    "mode1_no_control": "1 - Sans contrôle",
    "mode2_water_control": "2 - Contrôle débit eau",
    "mode3_bypass": "3 - Bypass produit",
    "mode4_single_pass": "4 - Passage unique"
}

MODE_COLORS = {
    "mode1_no_control": "#e74c3c",
    "mode2_water_control": "#3498db",
    "mode3_bypass": "#27ae60",
    "mode4_single_pass": "#9b59b6"
}

MODE_DESCRIPTIONS = {
    "mode1_no_control": "Débit fixe, pas de contrôle",
    "mode2_water_control": "PID contrôle débit eau → T° sortie ≤ 36°C",
    "mode3_bypass": "Eau max, PID contrôle fraction produit",
    "mode4_single_pass": "Passage unique, contrôle débit → T° sortie = 60°C"
}


def industrial_preset():
    """Return industrial case preset parameters."""
    return {
        "batch_mass": 45000,       # kg
        "Cp_product": 2200,        # J/(kg·K)
        "density": 900,            # kg/m³
        "T_initial": 150,          # °C
        "recirculation_flow": 50,   # kg/s (180 t/h)
        "water_flow": 111,         # kg/s (400 t/h)
        "water_inlet": 28,         # °C
        "water_max_outlet": 36,    # °C
        "exchanger_area": 100,     # m²
        "U_value": 600,            # W/(m²·K)
        "max_batch_time": 3600     # 60 minutes
    }


def render_pfd_schematic():
    """Render a simple PFD schematic using Plotly."""
    fig = go.Figure()

    # Tank (cuve)
    fig.add_shape(type="rect", x0=0.1, y0=0.3, x1=0.25, y1=0.7,
                  fillcolor="#ffcccb", line=dict(color="#e74c3c", width=2),
                  name="Cuve")

    # Exchanger
    fig.add_shape(type="rect", x0=0.55, y0=0.35, x1=0.75, y1=0.65,
                  fillcolor="#ffe4b5", line=dict(color="#f39c12", width=2),
                  name="Échangeur")

    # Water connection (bottom)
    fig.add_shape(type="rect", x0=0.55, y0=0.1, x1=0.75, y1=0.25,
                  fillcolor="#add8e6", line=dict(color="#3498db", width=2),
                  name="Eau")

    # Arrows and annotations
    annotations = [
        dict(x=0.17, y=0.8, text="Cuve", showarrow=False, font=dict(size=12)),
        dict(x=0.65, y=0.8, text="Échangeur", showarrow=False, font=dict(size=12)),
        dict(x=0.65, y=0.05, text="Eau", showarrow=False, font=dict(size=10)),
        dict(x=0.4, y=0.5, text="Produit", showarrow=False, font=dict(size=9)),
        dict(x=0.5, y=0.2, text="Eau froide", showarrow=False, font=dict(size=9)),
        dict(x=0.8, y=0.5, text="Produit\nrefroidi", showarrow=False, font=dict(size=9)),
    ]

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        plot_bgcolor="white"
    )

    return fig


def render_block_parameter(key: str, label: str, icon: str,
                           fields: dict, default_values: dict,
                           col=None):
    """Render a parameter block with fields."""
    if col:
        with col:
            with st.container():
                st.markdown(f"**{icon} {label}**")
                values = {}
                for field_key, field_info in fields.items():
                    default = default_values.get(field_key, field_info.get("default", 0))
                    values[field_key] = st.number_input(
                        field_info["label"],
                        min_value=field_info.get("min", 0),
                        max_value=field_info.get("max", 1e6),
                        value=default,
                        step=field_info.get("step", 1),
                        key=f"{key}_{field_key}",
                        label_visibility="collapsed"
                    )
                return values
    else:
        st.markdown(f"**{icon} {label}**")
        values = {}
        for field_key, field_info in fields.items():
            default = default_values.get(field_key, field_info.get("default", 0))
            values[field_key] = st.number_input(
                field_info["label"],
                min_value=field_info.get("min", 0),
                max_value=field_info.get("max", 1e6),
                value=default,
                step=field_info.get("step", 1),
                key=f"{key}_{field_key}",
                label_visibility="collapsed"
            )
        return values


def plot_temperature_evolution(results_dict: dict, target_temp: float):
    """Plot temperature evolution for all modes."""
    fig = go.Figure()

    for name, results in results_dict.items():
        color = MODE_COLORS.get(name, "#333")
        label = MODE_NAMES.get(name, name)

        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_tank,
            name=f"{label} - Cuve",
            line=dict(color=color, width=2.5),
            hovertemplate="t=%{x:.0f}s<br>T=%{y:.1f}°C"
        ))

    fig.add_hline(y=target_temp, line_dash="dash", line_color="gray",
                  annotation_text=f"Cible: {target_temp}°C")

    fig.update_layout(
        title="Température Cuve / Produit vs Temps",
        xaxis_title="Temps [s]",
        yaxis_title="Température [°C]",
        hovermode="x unified",
        height=400
    )

    return fig


def plot_water_temperature(results_dict: dict, max_allowed: float):
    """Plot water outlet temperature for all modes."""
    fig = go.Figure()

    for name, results in results_dict.items():
        color = MODE_COLORS.get(name, "#333")
        label = MODE_NAMES.get(name, name)

        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_water_out,
            name=label,
            line=dict(color=color, width=2, dash="dot"),
            hovertemplate="t=%{x:.0f}s<br>T_eau=%{y:.1f}°C"
        ))

    fig.add_hline(y=max_allowed, line_dash="dash", line_color="red",
                  annotation_text=f"Max: {max_allowed}°C")

    fig.update_layout(
        title="Température Sortie Eau de Refroidissement",
        xaxis_title="Temps [s]",
        yaxis_title="T° eau [°C]",
        hovermode="x unified",
        height=350
    )

    return fig


def plot_power_and_energy(results_dict: dict):
    """Plot power transfer and cumulative energy."""
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=["Puissance Thermique [kW]", "Énergie Cumulée [MJ]"])

    for name, results in results_dict.items():
        color = MODE_COLORS.get(name, "#333")
        label = MODE_NAMES.get(name, name)

        fig.add_trace(go.Scatter(
            x=results.t, y=results.Q_W / 1000,
            name=label,
            line=dict(color=color, width=2),
            showlegend=False
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=results.t, y=results.energy_J / 1e6,
            name=label,
            line=dict(color=color, width=2),
            showlegend=True
        ), row=1, col=2)

    fig.update_layout(height=350, hovermode="x unified")
    fig.update_xaxes(title_text="Temps [s]", row=1, col=1)
    fig.update_xaxes(title_text="Temps [s]", row=1, col=2)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="MJ", row=1, col=2)

    return fig


def plot_overlay_all(results_dict: dict, target_temp: float, max_water: float):
    """Create comprehensive overlay plot."""
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=("T° Cuve [°C]", "T° Eau Sortie [°C]",
                                      "Puissance [kW]", "Énergie [MJ]"))

    for name, results in results_dict.items():
        color = MODE_COLORS.get(name, "#333")

        # T° Cuve
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_tank,
            line=dict(color=color, width=2), showlegend=False
        ), row=1, col=1)

        # T° Eau
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_water_out,
            line=dict(color=color, width=2, dash="dot"), showlegend=False
        ), row=1, col=2)

        # Power
        fig.add_trace(go.Scatter(
            x=results.t, y=results.Q_W / 1000,
            line=dict(color=color, width=2), showlegend=False
        ), row=2, col=1)

        # Energy
        fig.add_trace(go.Scatter(
            x=results.t, y=results.energy_J / 1e6,
            line=dict(color=color, width=2), showlegend=False
        ), row=2, col=2)

    # Reference lines
    fig.add_hline(y=target_temp, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=max_water, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=600, showlegend=True,
                      title_text="Vue Overlay - Tous les Scénarios")

    fig.update_xaxes(title_text="Temps [s]", row=2, col=1)
    fig.update_xaxes(title_text="Temps [s]", row=2, col=2)

    return fig


def display_comparison_table(comparison: dict, max_time: float):
    """Display comparison table with color coding."""
    data = []
    for name, metrics in comparison.items():
        row = {
            "Mode": metrics["name"],
            "Temps à 60°C": metrics["time_to_60C"],
            "Contrainte eau": metrics["constraint_ok"],
            "Énergie [MJ]": metrics["total_energy_MJ"],
            "Puiss. moy. [kW]": metrics["avg_power_kW"],
            "Débit eau [t/h]": metrics["avg_water_flow_th"],
            "Score": f"{metrics['overall_score']:.0f}/100"
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Style function
    def style_row(row):
        styles = [""] * len(row)
        # Constraint column (index 2)
        if "❌" in str(row["Contrainte eau"]):
            styles[2] = "background-color: #ffe6e6"
        else:
            styles[2] = "background-color: #e6ffe6"
        return styles

    st.dataframe(
        df.style.apply(style_row, axis=1),
        use_container_width=True,
        height=180
    )

    return df


def display_recommendation(comparison: dict):
    """Display engineering recommendation."""
    # Find best by overall score
    best = max(comparison.items(), key=lambda x: x[1]["overall_score"])

    # Find fastest respecting constraint
    respecting = [(k, v) for k, v in comparison.items() if v["constraint_violation_C"] == 0]
    fastest_robust = min(respecting, key=lambda x: x[1]["time_to_60C_s"]) if respecting else None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🏆 Meilleur score", best[1]["name"],
                 f"{best[1]['overall_score']:.0f}/100")

    with col2:
        if fastest_robust:
            st.metric("⚡ Plus rapide (robuste)", fastest_robust[1]["name"],
                     fastest_robust[1]["time_to_60C"])
        else:
            st.metric("⚡ Plus rapide", "Aucun", "Contrainte non respectée")

    with col3:
        violated = [(k, v) for k, v in comparison.items() if v["constraint_violation_C"] > 0]
        st.metric("⚠️ Modes en violation", str(len(violated)),
                 f"+{violated[0][1]['constraint_violation_C']:.1f}°C") if violated else st.metric("⚠️ Modes en violation", "0", "Tous OK")

    st.divider()

    # Detailed analysis
    st.subheader("📋 Analyse Détaillée")

    for name, metrics in comparison.items():
        with st.expander(f"{metrics['name']}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Temps à 60°C:** {metrics['time_to_60C']}")
                st.write(f"**Énergie totale:** {metrics['total_energy_MJ']} MJ")
                st.write(f"**Puissance moyenne:** {metrics['avg_power_kW']} kW")

            with col_b:
                st.write(f"**Contrainte eau:** {metrics['constraint_ok']}")
                st.write(f"**Débit eau moyen:** {metrics['avg_water_flow_th']} t/h")
                st.write(f"**Score global:** {metrics['overall_score']:.0f}/100")

            # Score breakdown
            st.progress(metrics['overall_score'] / 100,
                        text=f"Performance: {metrics['overall_score']:.0f}%")

    # Conclusion
    st.divider()
    if fastest_robust:
        st.success(f"""
        **Recommandation:** Le mode **'{fastest_robust[1]['name']}'** est recommandé.

        - Atteint 60°C en {fastest_robust[1]['time_to_60C']}
        - Respecte la contrainte T° eau ≤ 36°C
        - Bon compromis performance/robustesse
        """)
    else:
        st.warning("""
        **Attention:** Aucun mode ne respecte la contrainte de température eau maximale.
        Considerer: augmenter le débit eau ou la surface d'échange.
        """)


def export_csv(results_dict: dict):
    """Export all results to CSV."""
    rows = []
    for name, results in results_dict.items():
        for i in range(len(results.t)):
            rows.append({
                "mode": name,
                "time_s": results.t[i],
                "T_tank_C": results.T_tank[i],
                "T_water_out_C": results.T_water_out[i],
                "Q_W": results.Q_W[i],
                "energy_J": results.energy_J[i]
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def main():
    st.title("🏭 Industrial Batch Cooling Simulator")
    st.markdown("""
    **Contexte:** Refroidissement batch industriel via échangeur tubes et calandre.
    Comparison de 4 architectures de refroidissement sous contrainte utilités.
    """)

    # Industrial preset button
    col_preset, col_spacer = st.columns([1, 4])
    with col_preset:
        if st.button("🏭 Industrial Case Preset (45t batch)", use_container_width=True):
            st.session_state.preset = industrial_preset()
            st.rerun()

    # Initialize preset
    if 'preset' not in st.session_state:
        st.session_state.preset = {
            "batch_mass": 500, "Cp_product": 2200, "density": 900,
            "T_initial": 150, "recirculation_flow": 1.0,
            "water_flow": 10, "water_inlet": 28, "water_max_outlet": 36,
            "exchanger_area": 10, "U_value": 500, "max_batch_time": 3600
        }

    preset = st.session_state.preset

    # === PFD Schematic ===
    st.subheader("📊 Schéma Procédé (PFD)")
    fig_pfd = render_pfd_schematic()
    st.plotly_chart(fig_pfd, use_container_width=True)

    st.divider()

    # === Parameter Blocks ===
    st.subheader("🔧 Configuration des Paramètres")

    col_prod, col_hex, col_water, col_ctrl = st.columns(4)

    # Product block
    with col_prod:
        st.markdown("### 📦 Produit (Cuve)")
        batch_mass = st.number_input("Masse batch [kg]", 100, 100000, preset["batch_mass"], 100, key="batch_mass")
        Cp = st.number_input("Cp [J/(kg·K)]", 1000, 4000, preset["Cp_product"], 50, key="Cp")
        density = st.number_input("Densité [kg/m³]", 500, 1500, preset["density"], 10, key="density")
        T_init = st.number_input("T° initiale [°C]", 50, 250, preset["T_initial"], 5, key="T_init")
        recirculation = st.number_input("Débit recirculation [kg/s]", 1, 200, preset["recirculation_flow"], 1, key="recirculation")

    # Exchanger block
    with col_hex:
        st.markdown("### 🔥 Échangeur")
        A = st.number_input("Surface A [m²]", 1, 300, preset["exchanger_area"], 1, key="A")
        U = st.number_input("Coefficient U [W/(m²·K)]", 100, 2000, preset["U_value"], 50, key="U")

    # Water block
    with col_water:
        st.markdown("### 💧 Eau Refroidissement")
        water_flow = st.number_input("Débit eau [kg/s]", 1, 500, preset["water_flow"], 1, key="water_flow")
        T_water_in = st.number_input("T° entrée [°C]", 15, 40, preset["water_inlet"], 1, key="T_water_in")
        T_water_max = st.number_input("T° sortie max [°C]", 30, 50, preset["water_max_outlet"], 1, key="T_water_max")

    # Control block
    with col_ctrl:
        st.markdown("### ⏱️ Contraintes & Mode")
        target_temp = st.number_input("T° cible [°C]", 30, 120, 60, 5, key="target")
        max_time = st.number_input("Temps max batch [s]", 300, 7200, preset["max_batch_time"], 60, key="max_time")
        max_time_min = max_time / 60

        st.write(f"**Contrainte:** {max_time_min:.0f} min")

        mode_options = ["Tous les modes (comparaison)", "Mode 1 - Sans contrôle",
                       "Mode 2 - Contrôle débit eau", "Mode 3 - Bypass produit",
                       "Mode 4 - Passage unique"]
        selected_mode = st.selectbox("Mode à simuler", mode_options, key="mode_select")

    # Update preset with current values for next load
    st.session_state.preset = {
        "batch_mass": batch_mass, "Cp_product": Cp, "density": density,
        "T_initial": T_init, "recirculation_flow": recirculation,
        "water_flow": water_flow, "water_inlet": T_water_in, "water_max_outlet": T_water_max,
        "exchanger_area": A, "U_value": U, "max_batch_time": max_time
    }

    st.divider()

    # === Run Simulation ===
    col_run, col_export = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("▶️ Lancer Simulation", type="primary", use_container_width=True)

    with col_export:
        if st.session_state.get('results') and st.button("📊 Exporter CSV", use_container_width=True):
            csv = export_csv(st.session_state.results)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="batch_cooling_results.csv",
                mime="text/csv"
            )

    # Run simulation
    if run_clicked:
        with st.spinner("Simulation en cours..."):
            # Create config
            batch = ProductBatch(
                mass=batch_mass, Cp=Cp, density=density,
                initial_temp=T_init, recirculation_flow=recirculation
            )
            water = CoolingWater(
                flow_rate=water_flow, inlet_temp=T_water_in,
                max_outlet_temp=T_water_max
            )
            exchanger = HeatExchanger(area=A, U=U)

            config = SimulationConfig(
                batch=batch, water=water, exchanger=exchanger,
                max_batch_time=max_time, target_temp=target_temp
            )

            # Run selected mode(s)
            if selected_mode == "Tous les modes (comparaison)":
                results_dict = run_all_modes(config)
            else:
                mode_map = {
                    "Mode 1 - Sans contrôle": "mode1_no_control",
                    "Mode 2 - Contrôle débit eau": "mode2_water_control",
                    "Mode 3 - Bypass produit": "mode3_bypass",
                    "Mode 4 - Passage unique": "mode4_single_pass"
                }
                mode_key = mode_map.get(selected_mode, "mode1_no_control")

                from model import Architecture
                arch_map = {
                    "mode1_no_control": Architecture.RECIRC_NO_CONTROL,
                    "mode2_water_control": Architecture.RECIRC_WATER_CONTROL,
                    "mode3_bypass": Architecture.RECIRC_BYPASS,
                    "mode4_single_pass": Architecture.SINGLE_PASS
                }

                cfg = SimulationConfig(
                    batch=batch, water=water, exchanger=exchanger,
                    max_batch_time=max_time, target_temp=target_temp,
                    mode=arch_map[mode_key]
                )
                sim = BatchCoolingSimulator(cfg)
                results_dict = {mode_key: sim.run()}

            st.session_state.results = results_dict
            st.session_state.comparison = create_comparison_dataframe(results_dict, max_time)
            st.session_state.config = config

    # === Display Results ===
    if st.session_state.get('results'):
        results_dict = st.session_state.results
        comparison = st.session_state.comparison

        # Quick metrics row
        st.divider()
        st.subheader("📈 Métriques Clés")

        cols = st.columns(len(results_dict))
        for col, (name, results) in zip(cols, results_dict.items()):
            with col:
                time_str = f"{results.time_to_60C_s:.0f}s" if results.time_to_60C_s else "—"
                st.metric(MODE_NAMES.get(name, name), time_str,
                         f"{results.avg_power_kW:.0f} kW moy")

        # Detailed constraint check
        st.write("**Respect contrainte eau (36°C max):**")
        constraint_cols = st.columns(len(results_dict))
        for col, (name, results) in zip(constraint_cols, results_dict.items()):
            with col:
                if results.constraint_satisfied:
                    st.success(f"✅ {MODE_NAMES.get(name, name)}")
                else:
                    st.error(f"❌ {MODE_NAMES.get(name, name)}: +{results.constraint_violation_max_C:.1f}°C")

        # Plots
        st.divider()
        st.subheader("📉 Graphiques")

        tab_overlay, tab_temp, tab_water, tab_power = st.tabs([
            "Vue Overlay", "Températures Cuve", "Eau Refroidissement", "Puissance"
        ])

        with tab_overlay:
            fig = plot_overlay_all(results_dict, target_temp, T_water_max)
            st.plotly_chart(fig, use_container_width=True)

        with tab_temp:
            fig = plot_temperature_evolution(results_dict, target_temp)
            st.plotly_chart(fig, use_container_width=True)

        with tab_water:
            fig = plot_water_temperature(results_dict, T_water_max)
            st.plotly_chart(fig, use_container_width=True)

        with tab_power:
            fig = plot_power_and_energy(results_dict)
            st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        st.divider()
        st.subheader("📊 Tableau Comparatif")

        display_comparison_table(comparison, max_time)

        # Recommendation
        display_recommendation(comparison)

        # Detailed data
        st.divider()
        st.subheader("📋 Données Détaillées")

        for name, results in results_dict.items():
            with st.expander(f"Données - {MODE_NAMES.get(name, name)}"):
                df = pd.DataFrame({
                    "t [s]": results.t,
                    "T_cuve [°C]": results.T_tank,
                    "T_eau_sortie [°C]": results.T_water_out,
                    "Q [kW]": results.Q_W / 1000,
                    "Énergie [MJ]": results.energy_J / 1e6
                })
                st.dataframe(df, use_container_width=True, height=200)

    else:
        # Default state
        st.divider()
        st.info("👆 Configurez les paramètres ci-dessus et cliquez sur 'Lancer Simulation'")
        st.markdown("""
        **Mode d'emploi:**
        1. Utilisez le preset "Industrial Case" ou saisissez vos paramètres
        2. Sélectionnez le mode à simuler (ou tous)
        3. Cliquez sur Lancer
        4. Comparez les résultats et contraintes
        """)


if __name__ == "__main__":
    main()