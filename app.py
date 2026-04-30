"""Streamlit application for heat exchanger simulation."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from simulation import Simulator, SimulationParameters, ControlMode
import time


st.set_page_config(
    page_title="Échangeur Thermique Tubulaire",
    page_icon="🌡️",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables."""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'params' not in st.session_state:
        st.session_state.params = None


def create_default_params() -> SimulationParameters:
    """Create default simulation parameters."""
    return SimulationParameters(
        # Hot fluid (product)
        m_hot=0.5,            # kg/s
        T_hot_inlet=85.0,      # °C
        Cp_hot=3800.0,        # J/(kg·K) - similar to glycol/water mixture

        # Cold fluid (refrigeration water)
        m_cold=0.8,           # kg/s
        T_cold_inlet=5.0,     # °C
        Cp_cold=4184.0,       # J/(kg·K) - water

        # Heat exchanger
        U=500.0,              # W/(m²·K)
        A=5.0,               # m²

        # Tank
        m_tank=200.0,        # kg
        Cp_tank=3800.0,      # J/(kg·K)
        T_tank_init=80.0,    # °C

        # Control
        control_mode=ControlMode.NO_CONTROL,
        setpoint=40.0,
        Kp=2.0,
        Ti=60.0,
        Td=0.0
    )


def render_sidebar():
    """Render sidebar with parameter inputs."""
    st.sidebar.header("🔧 Paramètres de Simulation")

    # Mode de contrôle
    st.sidebar.subheader("Mode de Contrôle")
    control_mode = st.sidebar.selectbox(
        "Sélectionner le mode",
        options=[ControlMode.NO_CONTROL, ControlMode.CONTROL_PRODUCT_TEMP, ControlMode.CONTROL_WATER_OUTLET],
        format_func=lambda x: {
            ControlMode.NO_CONTROL: "Sans contrôle",
            ControlMode.CONTROL_PRODUCT_TEMP: "Contrôle température produit (cuve)",
            ControlMode.CONTROL_WATER_OUTLET: "Contrôle température sortie eau"
        }[x],
        index=0
    )

    # Fluid parameters
    st.sidebar.subheader("🌡️ Fluide Chaud (Produit)")
    m_hot = st.sidebar.number_input("Débit massique [kg/s]", min_value=0.01, max_value=10.0, value=0.5, step=0.1)
    Cp_hot = st.sidebar.number_input("Cp [J/(kg·K)]", min_value=1000.0, max_value=5000.0, value=3800.0, step=100.0)
    T_hot_inlet = st.sidebar.number_input("Température initiale Cuve [°C]", min_value=0.0, max_value=150.0, value=80.0, step=5.0)

    st.sidebar.subheader("❄️ Fluide Froid (Eau Réfrigération)")
    m_cold = st.sidebar.number_input("Débit massique [kg/s]", min_value=0.01, max_value=10.0, value=0.8, step=0.1)
    Cp_cold = st.sidebar.number_input("Cp [J/(kg·K)]", min_value=1000.0, max_value=5000.0, value=4184.0, step=100.0)
    T_cold_inlet = st.sidebar.number_input("Température entrée eau [°C]", min_value=-10.0, max_value=50.0, value=5.0, step=1.0)

    st.sidebar.subheader("🔥 Échangeur Thermique")
    U = st.sidebar.number_input("Coefficient U [W/(m²·K)]", min_value=50.0, max_value=2000.0, value=500.0, step=50.0)
    A = st.sidebar.number_input("Surface d'échange [m²]", min_value=0.1, max_value=100.0, value=5.0, step=0.5)

    st.sidebar.subheader("📦 Cuve (Produit)")
    m_tank = st.sidebar.number_input("Masse produit [kg]", min_value=10.0, max_value=10000.0, value=200.0, step=10.0)
    Cp_tank = st.sidebar.number_input("Cp produit [J/(kg·K)]", min_value=1000.0, max_value=5000.0, value=3800.0, step=100.0)
    T_tank_init = st.sidebar.number_input("Température initiale [°C]", min_value=0.0, max_value=150.0, value=80.0, step=5.0)

    # Control parameters
    setpoint = 40.0
    Kp = 2.0
    Ti = 60.0

    if control_mode != ControlMode.NO_CONTROL:
        st.sidebar.subheader("🎯 Paramètres de Contrôle")
        setpoint = st.sidebar.number_input("Consigne [°C]", min_value=-20.0, max_value=150.0, value=40.0, step=5.0)
        Kp = st.sidebar.number_input("Gain proportionnel (Kp)", min_value=0.1, max_value=20.0, value=2.0, step=0.5)
        Ti = st.sidebar.number_input("Temps intégral (Ti) [s]", min_value=1.0, max_value=500.0, value=60.0, step=10.0)

    # Simulation parameters
    st.sidebar.subheader("⏱️ Simulation")
    duration = st.sidebar.number_input("Durée [s]", min_value=10.0, max_value=3600.0, value=600.0, step=10.0)
    dt_record = st.sidebar.number_input("Pas de temps enregistrement [s]", min_value=0.1, max_value=10.0, value=1.0, step=0.5)
    target_temp = st.sidebar.number_input("Température cible [°C]", min_value=-20.0, max_value=100.0, value=40.0, step=5.0)

    params = SimulationParameters(
        m_hot=m_hot,
        T_hot_inlet=T_hot_inlet,
        Cp_hot=Cp_hot,
        m_cold=m_cold,
        T_cold_inlet=T_cold_inlet,
        Cp_cold=Cp_cold,
        U=U,
        A=A,
        m_tank=m_tank,
        Cp_tank=Cp_tank,
        T_tank_init=T_tank_init,
        control_mode=control_mode,
        setpoint=setpoint,
        Kp=Kp,
        Ti=Ti,
        Td=0.0
    )

    return params, duration, dt_record, target_temp


def plot_temperature_evolution(results: dict):
    """Plot temperature evolution over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['T_tank'],
        name='Cuve (Produit)',
        line=dict(color='#e74c3c', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['T_hot_out'],
        name='Sortie échangeur → Cuve',
        line=dict(color='#f39c12', width=2, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['T_cold_out'],
        name='Sortie eau froide',
        line=dict(color='#3498db', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Évolution des Températures',
        xaxis_title='Temps [s]',
        yaxis_title='Température [°C]',
        hovermode='x unified',
        height=400
    )

    return fig


def plot_power_transfer(results: dict):
    """Plot heat transfer power and accumulated energy."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['time'],
        y=[q / 1000 for q in results['Q']],  # Convert to kW
        name='Puissance thermique [kW]',
        line=dict(color='#9b59b6', width=2),
        fill='tozeroy'
    ))

    fig.update_layout(
        title='Puissance Thermique Échangée',
        xaxis_title='Temps [s]',
        yaxis_title='Puissance [kW]',
        hovermode='x unified',
        height=300
    )

    return fig


def plot_energy(results: dict):
    """Plot accumulated energy."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['time'],
        y=[e / 1000 for e in results['energy']],  # Convert to kJ
        name='Énergie totale [kJ]',
        line=dict(color='#27ae60', width=2),
        fill='tozeroy'
    ))

    fig.update_layout(
        title='Énergie Thermique Totale Retirée',
        xaxis_title='Temps [s]',
        yaxis_title='Énergie [kJ]',
        hovermode='x unified',
        height=300
    )

    return fig


def plot_control_action(results: dict, control_mode: ControlMode):
    """Plot control action if applicable."""
    if control_mode == ControlMode.NO_CONTROL:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['time'],
        y=results['flow_multiplier'],
        name='Multiplicateur débit',
        line=dict(color='#1abc9c', width=2)
    ))

    fig.update_layout(
        title='Action de Contrôle',
        xaxis_title='Temps [s]',
        yaxis_title='Multiplicateur de débit',
        hovermode='x unified',
        height=250
    )

    return fig


def display_results_summary(results: dict, params: SimulationParameters, duration: float, target_temp: float):
    """Display numerical results summary."""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Température finale Cuve", f"{results['T_tank'][-1]:.1f} °C")

    with col2:
        st.metric("Température sortie eau", f"{results['T_cold_out'][-1]:.1f} °C")

    with col3:
        energy_kJ = results['energy'][-1] / 1000
        st.metric("Énergie retirée", f"{energy_kJ:.0f} kJ")

    with col4:
        # Calculate average power
        avg_power = results['energy'][-1] / results['time'][-1] if results['time'][-1] > 0 else 0
        st.metric("Puissance moyenne", f"{avg_power/1000:.1f} kW")

    # Time to reach target
    target_reached = False
    time_to_target = None

    for i, T in enumerate(results['T_tank']):
        if T <= target_temp:
            target_reached = True
            time_to_target = results['time'][i]
            break

    if target_reached:
        st.success(f"✓ Température cible de {target_temp}°C atteinte en {time_to_target:.0f} s")
    else:
        st.warning(f"○ Température cible de {target_temp}°C non atteinte dans la durée de simulation ({duration} s)")

    return time_to_target


def render_comparison_view():
    """Render comparison view for multiple scenarios."""
    st.subheader("📊 Comparaison de Scénarios")

    if not st.session_state.get('scenarios', []):
        st.info("Aucun scénario enregistré. Lancez des simulations et cliquez sur 'Sauvegarder ce scénario' pour les comparer.")
        return

    scenarios = st.session_state.scenarios

    fig = go.Figure()

    colors = ['#e74c3c', '#3498db', '#27ae60', '#9b59b6', '#f39c12']

    for i, scenario in enumerate(scenarios):
        fig.add_trace(go.Scatter(
            x=scenario['time'],
            y=scenario['T_tank'],
            name=f"{scenario['name']} (cuve)",
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title='Comparaison - Température Cuve',
        xaxis_title='Temps [s]',
        yaxis_title='Température [°C]',
        hovermode='x unified',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    comparison_data = []
    for scenario in scenarios:
        time_to_target = None
        for i, T in enumerate(scenario['T_tank']):
            if T <= 40:
                time_to_target = scenario['time'][i]
                break

        comparison_data.append({
            'Scénario': scenario['name'],
            'T finale Cuve [°C]': f"{scenario['T_tank'][-1]:.1f}",
            'Énergie [kJ]': f"{scenario['energy'][-1]/1000:.0f}",
            'Temps à 40°C [s]': f"{time_to_target:.0f}" if time_to_target else "Non atteint"
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)


def main():
    init_session_state()

    st.title("🌡️ Simulation Échangeur Thermique Tubulaire")
    st.markdown("""
    Simulation dynamique d'un échangeur thermique tubes et calandre avec了一圈 de la puissance de refroidissement
    sur un produit en recirculation à travers un échangeur refroidi par eau de réfrigération.
    """)

    # Render sidebar and get parameters
    params, duration, dt_record, target_temp = render_sidebar()

    # Simulation control buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        run_clicked = st.button("▶️ Lancer Simulation", type="primary", use_container_width=True)

    with col2:
        reset_clicked = st.button("🔄 Réinitialiser", use_container_width=True)

    with col3:
        if st.button("💾 Sauvegarder Scénario", use_container_width=True):
            if st.session_state.results:
                if 'scenarios' not in st.session_state:
                    st.session_state.scenarios = []

                name = f"Scénario {len(st.session_state.scenarios) + 1}"
                st.session_state.scenarios.append({
                    'name': name,
                    'params': params,
                    'time': st.session_state.results['time'].copy(),
                    'T_tank': st.session_state.results['T_tank'].copy(),
                    'T_hot_out': st.session_state.results['T_hot_out'].copy(),
                    'T_cold_out': st.session_state.results['T_cold_out'].copy(),
                    'Q': st.session_state.results['Q'].copy(),
                    'energy': st.session_state.results['energy'].copy()
                })
                st.success(f"{name} sauvegardé!")

    # Run simulation
    if run_clicked:
        with st.spinner("Simulation en cours..."):
            simulator = Simulator(params)
            results = simulator.run(duration=duration, dt=dt_record, target_temp=target_temp)
            st.session_state.simulator = simulator
            st.session_state.results = results
            st.session_state.params = params

    # Display results
    if st.session_state.results:
        results = st.session_state.results

        st.divider()
        st.subheader("📈 Résultats de Simulation")

        # Summary metrics
        display_results_summary(results, params, duration, target_temp)

        # Plots
        tab1, tab2, tab3, tab4 = st.tabs(["Températures", "Puissance", "Énergie", "Contrôle"])

        with tab1:
            fig = plot_temperature_evolution(results)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig_power = plot_power_transfer(results)
            st.plotly_chart(fig_power, use_container_width=True)

        with tab3:
            fig_energy = plot_energy(results)
            st.plotly_chart(fig_energy, use_container_width=True)

        with tab4:
            fig_control = plot_control_action(results, params.control_mode)
            if fig_control:
                st.plotly_chart(fig_control, use_container_width=True)
            else:
                st.info("Aucun contrôle actif dans ce scénario.")

        # Detailed results table
        st.divider()
        st.subheader("📋 Données Détaillées")

        df = pd.DataFrame({
            'Temps [s]': results['time'],
            'T Cuve [°C]': results['T_tank'],
            'T Sortie Échangeur [°C]': results['T_hot_out'],
            'T Sortie Eau [°C]': results['T_cold_out'],
            'Puissance [kW]': [q/1000 for q in results['Q']],
            'Énergie [kJ]': [e/1000 for e in results['energy']]
        })

        st.dataframe(
            df.style.format({
                'Temps [s]': '{:.1f}',
                'T Cuve [°C]': '{:.2f}',
                'T Sortie Échangeur [°C]': '{:.2f}',
                'T Sortie Eau [°C]': '{:.2f}',
                'Puissance [kW]': '{:.2f}',
                'Énergie [kJ]': '{:.1f}'
            }),
            use_container_width=True,
            height=300
        )

        # Comparison section
        st.divider()
        render_comparison_view()

    else:
        # Show default values and explanation
        st.divider()
        st.subheader("ℹ️ Configuration Actuelle")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fluide Chaud (Produit)**")
            st.write(f"- Débit: {params.m_hot} kg/s")
            st.write(f"- Cp: {params.Cp_hot} J/(kg·K)")
            st.write(f"- Température initiale: {params.T_tank_init} °C")

        with col2:
            st.markdown("**Fluide Froid (Eau Réfrigération)**")
            st.write(f"- Débit: {params.m_cold} kg/s")
            st.write(f"- Cp: {params.Cp_cold} J/(kg·K)")
            st.write(f"- Température entrée: {params.T_cold_inlet} °C")

        st.markdown("---")
        st.markdown("**Échangeur**")
        st.write(f"- Coefficient U: {params.U} W/(m²·K)")
        st.write(f"- Surface: {params.A} m²")

        st.markdown("**Cuve**")
        st.write(f"- Masse: {params.m_tank} kg")
        st.write(f"- Cp: {params.Cp_tank} J/(kg·K)")

        st.info("👈 Configurez les paramètres dans la barre latérale, puis cliquez sur 'Lancer Simulation' pour démarrer.")


if __name__ == "__main__":
    main()