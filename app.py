"""Streamlit application for industrial batch cooling simulation."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model import ProductFluid, CoolingWater, HeatExchanger, Architecture
from simulation import BatchCoolingSimulator, SimulationConfig, create_comparison_table

st.set_page_config(
    page_title="Refroidissement Batch Industriel",
    page_icon="🏭",
    layout="wide"
)


def init_session_state():
    """Initialize session state."""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'comparison' not in st.session_state:
        st.session_state.comparison = None
    if 'config' not in st.session_state:
        st.session_state.config = None


def render_sidebar():
    """Render sidebar with all input parameters."""
    st.sidebar.header("🔧 Paramètres du Procédé")

    # Product parameters
    st.sidebar.subheader("📦 Produit (Cuve)")
    mass = st.sidebar.number_input("Masse batch [kg]", 50, 10000, 500, 10)
    Cp_product = st.sidebar.number_input("Cp produit [J/(kg·K)]", 1000, 4000, 2200, 50)
    density = st.sidebar.number_input("Densité [kg/m³]", 500, 1500, 900, 10)
    initial_temp = st.sidebar.number_input("T° initiale [°C]", 100, 200, 150, 5)
    flow_rate_product = st.sidebar.number_input("Débit recirculation [kg/s]", 0.1, 10.0, 1.0, 0.1)

    # Water parameters
    st.sidebar.subheader("💧 Eau de Refroidissement")
    flow_rate_water = st.sidebar.number_input("Débit eau [kg/s]", 0.1, 20.0, 3.0, 0.5)
    water_inlet_temp = st.sidebar.number_input("T° entrée eau [°C]", 20, 40, 28, 1)
    max_water_temp = st.sidebar.number_input("T° sortie max [°C]", 30, 50, 36, 1)

    # Exchanger parameters
    st.sidebar.subheader("🔥 Échangeur")
    area = st.sidebar.number_input("Surface A [m²]", 0.5, 50.0, 10.0, 0.5)
    U = st.sidebar.number_input("Coefficient U [W/(m²·K)]", 50, 2000, 400, 50)
    fouling = st.sidebar.number_input("Fouling factor [m²·K/W]", 0.0, 0.001, 0.0001, 0.00005)

    # Simulation parameters
    st.sidebar.subheader("⏱️ Simulation")
    t_end = st.sidebar.number_input("Durée max [s]", 60, 7200, 1800, 60)
    target_temp = st.sidebar.number_input("T° cible [°C]", 40, 100, 60, 5)

    # Scenario selection
    st.sidebar.subheader("🎯 Scénario")
    scenario_options = {
        "Tous les scénarios (comparaison)": "all",
        "1 - Recirculation sans contrôle": "recirc_no_control",
        "2 - Recirculation avec contrôle T° eau": "recirc_water_control",
        "3 - Recirculation avec bypass": "recirc_bypass",
        "4 - Passage unique (single pass)": "single_pass"
    }
    selected_scenario = st.sidebar.selectbox("Scénario", list(scenario_options.keys()))

    return {
        "product": ProductFluid(
            mass=mass, Cp=Cp_product, density=density,
            initial_temp=initial_temp, flow_rate=flow_rate_product
        ),
        "water": CoolingWater(
            flow_rate=flow_rate_water, inlet_temp=water_inlet_temp,
            max_outlet_temp=max_water_temp, Cp=4184.0
        ),
        "exchanger": HeatExchanger(
            area=area, U=U, fouling_factor=fouling
        ),
        "scenario": scenario_options[selected_scenario],
        "t_end": t_end,
        "target_temp": target_temp
    }


def run_single_simulation(params: dict, arch: Architecture) -> dict:
    """Run a single scenario simulation."""
    config = SimulationConfig(
        product=params["product"],
        water=params["water"],
        exchanger=params["exchanger"],
        architecture=arch,
        t_end=params["t_end"],
        target_temp=params["target_temp"]
    )

    sim = BatchCoolingSimulator(config)
    return sim.run_simulation()


def plot_temperature_comparison(results_dict: dict, target_temp: float):
    """Plot temperature comparison for all scenarios."""
    fig = go.Figure()

    colors = {
        "recirc_no_control": "#e74c3c",
        "recirc_water_control": "#3498db",
        "recirc_bypass": "#27ae60",
        "single_pass": "#9b59b6"
    }

    names = {
        "recirc_no_control": "Sans contrôle",
        "recirc_water_control": "Contrôle T° eau",
        "recirc_bypass": "Bypass produit",
        "single_pass": "Passage unique"
    }

    for name, results in results_dict.items():
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_product,
            name=f"{names[name]} - Cuve",
            line=dict(color=colors[name], width=2),
            opacity=0.9
        ))

    # Add target line
    fig.add_hline(y=target_temp, line_dash="dash", line_color="gray",
                  annotation_text=f"Cible: {target_temp}°C")

    fig.update_layout(
        title="Comparaison - Température Cuve/Produit",
        xaxis_title="Temps [s]",
        yaxis_title="Température [°C]",
        hovermode="x unified",
        height=400
    )

    return fig


def plot_water_temperature(results_dict: dict, max_allowed: float):
    """Plot water outlet temperatures."""
    fig = go.Figure()

    colors = {
        "recirc_no_control": "#e74c3c",
        "recirc_water_control": "#3498db",
        "recirc_bypass": "#27ae60",
        "single_pass": "#9b59b6"
    }

    names = {
        "recirc_no_control": "Sans contrôle",
        "recirc_water_control": "Contrôle T° eau",
        "recirc_bypass": "Bypass produit",
        "single_pass": "Passage unique"
    }

    for name, results in results_dict.items():
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_water_out,
            name=f"{names[name]} - Eau",
            line=dict(color=colors[name], width=2, dash="dot")
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


def plot_power_transfer(results_dict: dict):
    """Plot heat transfer power."""
    fig = go.Figure()

    colors = {
        "recirc_no_control": "#e74c3c",
        "recirc_water_control": "#3498db",
        "recirc_bypass": "#27ae60",
        "single_pass": "#9b59b6"
    }

    names = {
        "recirc_no_control": "Sans contrôle",
        "recirc_water_control": "Contrôle T° eau",
        "recirc_bypass": "Bypass produit",
        "single_pass": "Passage unique"
    }

    for name, results in results_dict.items():
        fig.add_trace(go.Scatter(
            x=results.t, y=results.Q / 1000,
            name=names[name],
            line=dict(color=colors[name], width=2),
            fill='tozeroy' if name == "recirc_no_control" else None
        ))

    fig.update_layout(
        title="Puissance Thermique Transférée",
        xaxis_title="Temps [s]",
        yaxis_title="Puissance [kW]",
        hovermode="x unified",
        height=300
    )

    return fig


def plot_overlay_all(results_dict: dict, target_temp: float):
    """Create overlay plot with all temperatures."""
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=("Température Produit", "Température Eau",
                                      "Puissance Thermique", "Énergie Cumulée"))

    colors = {
        "recirc_no_control": "#e74c3c",
        "recirc_water_control": "#3498db",
        "recirc_bypass": "#27ae60",
        "single_pass": "#9b59b6"
    }

    names = {
        "recirc_no_control": "Sans contrôle",
        "recirc_water_control": "Contrôle T° eau",
        "recirc_bypass": "Bypass produit",
        "single_pass": "Passage unique"
    }

    for name, results in results_dict.items():
        color = colors[name]
        label = names[name]

        # Product temperature (row 1, col 1)
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_product, name=label,
            line=dict(color=color, width=2), showlegend=False
        ), row=1, col=1)

        # Water temperature (row 1, col 2)
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_water_out, name=label,
            line=dict(color=color, width=2, dash="dot"), showlegend=True
        ), row=1, col=2)

        # Power (row 2, col 1)
        fig.add_trace(go.Scatter(
            x=results.t, y=results.Q/1000, name=label,
            line=dict(color=color, width=2), showlegend=False
        ), row=2, col=1)

        # Energy (row 2, col 2)
        fig.add_trace(go.Scatter(
            x=results.t, y=results.energy/1e6, name=label,
            line=dict(color=color, width=2), showlegend=False
        ), row=2, col=2)

    # Add target line to first subplot
    fig.add_hline(y=target_temp, line_dash="dash", line_color="gray", row=1, col=1)

    # Add max water temp line to second subplot
    fig.add_hline(y=36, line_dash="dash", line_color="red", row=1, col=2)

    fig.update_layout(height=700, showlegend=True,
                      title_text="Vue Overlay - Tous les Scénarios")

    fig.update_xaxes(title_text="Temps [s]", row=2, col=1)
    fig.update_xaxes(title_text="Temps [s]", row=2, col=2)
    fig.update_yaxes(title_text="T° [°C]", row=1, col=1)
    fig.update_yaxes(title_text="T° [°C]", row=1, col=2)
    fig.update_yaxes(title_text="Puissance [kW]", row=2, col=1)
    fig.update_yaxes(title_text="Énergie [MJ]", row=2, col=2)

    return fig


def display_comparison_table(comparison: dict, target_temp: float):
    """Display comparison table with metrics."""
    data = []
    for name, metrics in comparison.items():
        names = {
            "recirc_no_control": "Sans contrôle",
            "recirc_water_control": "Contrôle T° eau",
            "recirc_bypass": "Bypass produit",
            "single_pass": "Passage unique"
        }

        constraint_icon = "✅" if metrics["constraint_met"] else "❌"

        data.append({
            "Scénario": names.get(name, name),
            "Temps à 60°C [s]": metrics["time_to_60°C"],
            "Énergie [MJ]": f"{metrics['total_energy_MJ']:.2f}",
            "Puissance moy. [kW]": f"{metrics['avg_power_kW']:.1f}",
            "T° eau max [°C]": metrics["max_water_temp"],
            "Contrainte eau": constraint_icon,
            "Coût proxy": f"{metrics['pumping_proxy']:.2f}"
        })

    df = pd.DataFrame(data)

    # Style the dataframe
    def highlight_constraint(row):
        if row["Contrainte eau"] == "✅":
            return ["background-color: #d4edda"] * len(row)
        else:
            return ["background-color: #f8d7da"] * len(row)

    st.dataframe(
        df.style.apply(highlight_constraint, axis=1),
        use_container_width=True,
        height=200
    )

    return df


def display_recommendation(comparison: dict):
    """Display engineering recommendation based on results."""
    st.subheader("📋 Recommandation Procédure")

    # Find best scenario for different criteria
    fastest = min(comparison.items(), key=lambda x: x[1]["time_to_60°C_s"])
    most_efficient = min(comparison.items(), key=lambda x: x[1]["total_energy_MJ"])
    respects_constraint = [(k, v) for k, v in comparison.items() if v["constraint_met"]]
    most_robust = min(respects_constraint, key=lambda x: x[1]["time_to_60°C_s"]) if respects_constraint else None

    names = {
        "recirc_no_control": "Sans contrôle",
        "recirc_water_control": "Contrôle T° eau",
        "recirc_bypass": "Bypass produit",
        "single_pass": "Passage unique"
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⏱️ Plus rapide", names.get(fastest[0], fastest[0]),
                 f"{fastest[1]['time_to_60°C']}")
        st.caption(f"Énergie: {fastest[1]['total_energy_MJ']:.1f} MJ")

    with col2:
        if most_robust:
            st.metric("🛡️ Plus robuste", names.get(most_robust[0], most_robust[0]),
                     f"Contrainte respectée")
        else:
            st.metric("🛡️ Plus robuste", "Aucun", "Tous dépassent la contrainte")

    with col3:
        st.metric("⚡ Plus économe", names.get(most_efficient[0], most_efficient[0]),
                 f"{most_efficient[1]['total_energy_MJ']:.1f} MJ")

    st.markdown("---")

    # Overall recommendation
    if most_robust:
        st.success(f"""
        **Conclusion:** L'architecture **'{names.get(most_robust[0])}'** est recommandée.

        - Atteint 60°C en {most_robust[1]['time_to_60°C']}
        - Respecte la contrainte T° eau ≤ 36°C
        - Solution balance performance/robustesse
        """)
    else:
        st.warning("""
        **Attention:** Aucun scénario ne respecte la contrainte de température eau maximale.
        Considerer: augmentation du débit eau ou surface d'échange.
        """)


def export_to_csv(results_dict: dict, comparison: dict):
    """Export results to CSV."""
    csv_data = []

    for name, results in results_dict.items():
        for i in range(len(results.t)):
            csv_data.append({
                "scenario": name,
                "time_s": results.t[i],
                "T_product_C": results.T_product[i],
                "T_water_out_C": results.T_water_out[i],
                "Q_W": results.Q[i],
                "energy_J": results.energy[i] if i < len(results.energy) else 0
            })

    df = pd.DataFrame(csv_data)
    return df.to_csv(index=False)


def main():
    init_session_state()

    st.title("🏭 Refroidissement Batch - Simulation Échangeur Thermique")
    st.markdown("""
    **Contexte:** Refroidissement d'additifs pour huiles de lubrification moteur.
    Batch à 150°C → 60°C via échangeur tubes et calandre, eau de réfrigération à 28°C.
    """)

    params = render_sidebar()

    # Run button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button("▶️ Lancer Simulation", type="primary", use_container_width=True)

    with col2:
        if st.button("📊 Exporter CSV", use_container_width=True):
            if st.session_state.results:
                csv = export_to_csv(st.session_state.results, st.session_state.comparison)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="batch_cooling_results.csv",
                    mime="text/csv"
                )

    if run_clicked:
        with st.spinner("Simulation en cours..."):
            results_dict = {}

            if params["scenario"] == "all":
                architectures = [
                    Architecture.RECIRC_NO_CONTROL,
                    Architecture.RECIRC_WATER_TEMP_CONTROL,
                    Architecture.RECIRC_BYPASS,
                    Architecture.SINGLE_PASS
                ]
            else:
                arch_map = {
                    "recirc_no_control": Architecture.RECIRC_NO_CONTROL,
                    "recirc_water_control": Architecture.RECIRC_WATER_TEMP_CONTROL,
                    "recirc_bypass": Architecture.RECIRC_BYPASS,
                    "single_pass": Architecture.SINGLE_PASS
                }
                architectures = [arch_map[params["scenario"]]]

            for arch in architectures:
                results_dict[arch.value] = run_single_simulation(params, arch)

            st.session_state.results = results_dict
            st.session_state.comparison = create_comparison_table(results_dict)
            st.session_state.config = params

    # Display results
    if st.session_state.results:
        results_dict = st.session_state.results
        comparison = st.session_state.comparison

        st.divider()
        st.subheader("📈 Résultats")

        # Quick metrics
        cols = st.columns(len(results_dict))
        names = {
            "recirc_no_control": "Sans contrôle",
            "recirc_water_control": "Contrôle T° eau",
            "recirc_bypass": "Bypass",
            "single_pass": "Single pass"
        }

        for col, (name, results) in zip(cols, results_dict.items()):
            with col:
                time_str = results.time_to_target if results.time_to_target else "—"
                if results.time_to_target:
                    time_str = f"{results.time_to_target:.0f}s"
                st.metric(f"{names.get(name, name)}", time_str, f"{results.avg_power/1000:.1f} kW moy")

        # Plots
        tab_single, tab_overlay, tab_water, tab_power = st.tabs([
            "Scénarios Individuels", "Vue Overlay", "Eau Refroidissement", "Puissance"
        ])

        with tab_single:
            fig = plot_temperature_comparison(results_dict, params["target_temp"])
            st.plotly_chart(fig, use_container_width=True)

        with tab_overlay:
            fig = plot_overlay_all(results_dict, params["target_temp"])
            st.plotly_chart(fig, use_container_width=True)

        with tab_water:
            fig = plot_water_temperature(results_dict, params["water"].max_outlet_temp)
            st.plotly_chart(fig, use_container_width=True)

        with tab_power:
            fig = plot_power_transfer(results_dict)
            st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        st.divider()
        st.subheader("📊 Analyse Comparative")

        df = display_comparison_table(comparison, params["target_temp"])

        display_recommendation(comparison)

        # Detailed data
        st.divider()
        st.subheader("📋 Données Détaillées")

        for name, results in results_dict.items():
            with st.expander(f"Données - {names.get(name, name)}"):
                df_detail = pd.DataFrame({
                    "t [s]": results.t,
                    "T_prod [°C]": results.T_product,
                    "T_eau_sortie [°C]": results.T_water_out,
                    "Q [kW]": results.Q / 1000,
                    "Énergie [MJ]": results.energy / 1e6
                })
                st.dataframe(df_detail, use_container_width=True, height=200)

    else:
        # Default info display
        st.divider()
        st.subheader("ℹ️ Configuration")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Produit**")
            st.write(f"- Masse: {params['product'].mass} kg")
            st.write(f"- Cp: {params['product'].Cp} J/(kg·K)")
            st.write(f"- T° initiale: {params['product'].initial_temp}°C")
            st.write(f"- Débit: {params['product'].flow_rate} kg/s")

        with col2:
            st.markdown("**Eau Refroidissement**")
            st.write(f"- Débit: {params['water'].flow_rate} kg/s")
            st.write(f"- T° entrée: {params['water'].inlet_temp}°C")
            st.write(f"- T° sortie max: {params['water'].max_outlet_temp}°C")

        st.markdown("**Échangeur**")
        st.write(f"- Surface: {params['exchanger'].area} m²")
        st.write(f"- U: {params['exchanger'].U} W/(m²·K)")

        st.info("👈 Configurez les paramètres et cliquez sur 'Lancer Simulation'")


if __name__ == "__main__":
    main()