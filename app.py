"""Streamlit application for industrial batch cooling simulation.

PFD-style interface with block-based parameter entry and industrial audio feedback.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

from model import ProductBatch, CoolingWater, HeatExchanger, Architecture
from simulation import BatchCoolingSimulator, SimulationConfig, run_all_modes, create_comparison_dataframe
from audio_manager import get_audio_manager, play_sound, toggle_audio, get_sound_base64

st.set_page_config(
    page_title="Industrial Batch Cooling Simulator",
    page_icon="🏭",
    layout="wide"
)


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


def industrial_preset():
    return {
        "batch_mass": 45000.0, "Cp_product": 2200.0, "density": 900.0,
        "T_initial": 150.0, "recirculation_flow": 50.0,
        "water_flow": 111.0, "water_inlet": 28.0, "water_max_outlet": 36.0,
        "exchanger_area": 100.0, "U_value": 600.0, "max_batch_time": 3600.0
    }


def inject_audio_manager():
    """Inject audio system using components.html (hidden)."""
    audio = get_audio_manager()
    sounds = audio.get_all_sounds()

    audio_elements = ""
    for name, data_uri in sounds.items():
        audio_elements += f'<audio id="sound_{name}" src="{data_uri}" preload="auto"></audio>'

    js = """
    <script>
    window.AudioManager = {
        muted: false,
        play: function(name) {
            if (this.muted) return;
            var a = document.getElementById('sound_' + name);
            if (a) { a.currentTime = 0; a.play().catch(() => {}); }
        },
        toggle: function() { this.muted = !this.muted; return this.muted; }
    };
    </script>
    """

    html = f'<div style="display:none">{audio_elements}{js}</div>'
    components.html(html, height=0, width=0)


def play_ui_sound(sound_name: str):
    """Play sound (works with injected JS audio elements)."""
    audio = get_audio_manager()
    if audio.is_muted():
        return
    data_uri = play_sound(sound_name)
    if data_uri:
        st.audio(data_uri, format="audio/wav")


def render_pfd_schematic():
    """Render improved PFD schematic with plotly shapes and arrows."""

    fig = go.Figure()

    # Tank (cuve)
    fig.add_shape(type="rect", x0=0.05, y0=0.2, x1=0.18, y1=0.75,
                  fillcolor="#ffcccc", line=dict(color="#c0392b", width=3))
    fig.add_annotation(x=0.115, y=0.85, text="<b>CUVE</b><br>Produit", showarrow=False,
                       font=dict(size=14, color="#c0392b"), align="center")

    # Pump
    fig.add_shape(type="circle", x0=0.28, y0=0.42, x1=0.33, y1=0.55,
                  fillcolor="#e74c3c", line=dict(color="#c0392b", width=2))
    fig.add_annotation(x=0.305, y=0.35, text="POMPE", showarrow=False,
                       font=dict(size=9, color="#666"))

    # Heat Exchanger
    fig.add_shape(type="rect", x0=0.45, y0=0.15, x1=0.7, y1=0.82,
                  fillcolor="#ffeaa7", line=dict(color="#d68910", width=3))
    fig.add_annotation(x=0.575, y=0.88, text="<b>ÉCHANGEUR</b><br>Tubes/Calandre", showarrow=False,
                       font=dict(size=12, color="#d68910"), align="center")

    # Water tank
    fig.add_shape(type="rect", x0=0.45, y0=0.02, x1=0.7, y1=0.12,
                  fillcolor="#aed6f1", line=dict(color="#2980b9", width=2))
    fig.add_annotation(x=0.575, y=0.15, text="EAU RÉFRIGÉRATION", showarrow=False,
                       font=dict(size=9, color="#2980b9"))

    # Product pipe (top)
    fig.add_shape(type="line", x0=0.18, y0=0.55, x1=0.45, y1=0.55,
                  line=dict(color="#c0392b", width=4))
    fig.add_annotation(x=0.315, y=0.62, text="Produit chaud", showarrow=False,
                       font=dict(size=10, color="#c0392b"))

    # Product pipe return (bottom)
    fig.add_shape(type="line", x0=0.18, y0=0.35, x1=0.45, y1=0.35,
                  line=dict(color="#27ae60", width=4))
    fig.add_annotation(x=0.315, y=0.28, text="Produit refroidi", showarrow=False,
                       font=dict(size=10, color="#27ae60"))

    # Water pipes (bottom)
    fig.add_shape(type="line", x0=0.35, y0=0.07, x1=0.45, y1=0.07,
                  line=dict(color="#2980b9", width=3))
    fig.add_annotation(x=0.28, y=0.02, text="Eau froide 28°C", showarrow=False,
                       font=dict(size=8, color="#2980b9"))

    fig.add_shape(type="line", x0=0.7, y0=0.07, x1=0.85, y1=0.07,
                  line=dict(color="#2980b9", width=3))
    fig.add_annotation(x=0.775, y=0.02, text="Eau chaude →", showarrow=False,
                       font=dict(size=8, color="#2980b9"))

    # Valves
    fig.add_shape(type="circle", x0=0.22, y0=0.5, x1=0.25, y1=0.53,
                  fillcolor="#fff", line=dict(color="#333", width=1))
    fig.add_shape(type="circle", x0=0.4, y0=0.5, x1=0.43, y1=0.53,
                  fillcolor="#fff", line=dict(color="#333", width=1))

    # Arrow
    fig.add_annotation(x=0.4, y=0.55, ax=0.3, ay=0.55,
                      text="→", showarrow=True, arrowhead=2, arrowsize=2,
                      arrowcolor="#c0392b", font=dict(size=16))

    # Return flow dashed
    fig.add_shape(type="line", x0=0.18, y0=0.35, x1=0.18, y1=0.2,
                  line=dict(color="#27ae60", width=3, dash="dash"))

    fig.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 1], showgrid=False, showticklabels=False, zeroline=False),
        height=280,
        margin=dict(l=10, r=10, t=30, b=20),
        showlegend=False,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white"
    )

    return fig


def plot_temperature_evolution(results_dict, target_temp):
    fig = go.Figure()
    for name, results in results_dict.items():
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_tank,
            name=MODE_NAMES.get(name, name),
            line=dict(color=MODE_COLORS.get(name, "#333"), width=2.5),
            hovertemplate="t=%{x:.0f}s<br>T=%{y:.1f}°C"
        ))
    fig.add_hline(y=target_temp, line_dash="dash", line_color="gray",
                  annotation_text=f"Cible: {target_temp}°C")
    fig.update_layout(title="Température Cuve vs Temps",
                      xaxis_title="Temps [s]", yaxis_title="T° [°C]",
                      hovermode="x unified", height=400)
    return fig


def plot_water_temperature(results_dict, max_allowed):
    fig = go.Figure()
    for name, results in results_dict.items():
        fig.add_trace(go.Scatter(
            x=results.t, y=results.T_water_out,
            name=MODE_NAMES.get(name, name),
            line=dict(color=MODE_COLORS.get(name, "#333"), width=2, dash="dot"),
            hovertemplate="t=%{x:.0f}s<br>T_eau=%{y:.1f}°C"
        ))
    fig.add_hline(y=max_allowed, line_dash="dash", line_color="red",
                  annotation_text=f"Max: {max_allowed}°C")
    fig.update_layout(title="Température Sortie Eau",
                      xaxis_title="Temps [s]", yaxis_title="T° eau [°C]",
                      hovermode="x unified", height=350)
    return fig


def plot_power_and_energy(results_dict):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Puissance [kW]", "Énergie [MJ]"])
    for name, results in results_dict.items():
        color = MODE_COLORS.get(name, "#333")
        fig.add_trace(go.Scatter(x=results.t, y=results.Q_W/1000, name=MODE_NAMES.get(name, name),
                                line=dict(color=color, width=2), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=results.t, y=results.energy_J/1e6, name=MODE_NAMES.get(name, name),
                                line=dict(color=color, width=2), showlegend=True), row=1, col=2)
    fig.update_layout(height=350, hovermode="x unified")
    fig.update_xaxes(title_text="Temps [s]", row=1, col=1)
    fig.update_xaxes(title_text="Temps [s]", row=1, col=2)
    return fig


def plot_overlay_all(results_dict, target_temp, max_water):
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=["T° Cuve [°C]", "T° Eau Sortie [°C]", "Puissance [kW]", "Énergie [MJ]"])
    for name, results in results_dict.items():
        color = MODE_COLORS.get(name, "#333")
        fig.add_trace(go.Scatter(x=results.t, y=results.T_tank, line=dict(color=color, width=2), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=results.t, y=results.T_water_out, line=dict(color=color, width=2, dash="dot"), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=results.t, y=results.Q_W/1000, line=dict(color=color, width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=results.t, y=results.energy_J/1e6, line=dict(color=color, width=2), showlegend=False), row=2, col=2)
    fig.add_hline(y=target_temp, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=max_water, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(height=600, showlegend=True, title_text="Vue Overlay - Tous les Scénarios")
    fig.update_xaxes(title_text="Temps [s]", row=2, col=1)
    fig.update_xaxes(title_text="Temps [s]", row=2, col=2)
    return fig


def display_comparison_table(comparison, max_time):
    data = []
    for name, metrics in comparison.items():
        data.append({
            "Mode": metrics["name"],
            "Temps à 60°C": metrics["time_to_60C"],
            "Contrainte eau": metrics["constraint_ok"],
            "Énergie [MJ]": metrics["total_energy_MJ"],
            "Puiss. moy. [kW]": metrics["avg_power_kW"],
            "Débit eau [t/h]": metrics["avg_water_flow_th"],
            "Score": f"{metrics['overall_score']:.0f}/100"
        })
    df = pd.DataFrame(data)

    def style_row(row):
        styles = [""] * len(row)
        if "❌" in str(row["Contrainte eau"]):
            styles[2] = "background-color: #ffe6e6"
        else:
            styles[2] = "background-color: #e6ffe6"
        return styles

    st.dataframe(df.style.apply(style_row, axis=1), use_container_width=True, height=180)
    return df


def display_recommendation(comparison):
    best = max(comparison.items(), key=lambda x: x[1]["overall_score"])
    respecting = [(k, v) for k, v in comparison.items() if v["constraint_violation_C"] == 0]
    fastest_robust = min(respecting, key=lambda x: x[1]["time_to_60C_s"]) if respecting else None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Meilleur score", best[1]["name"], f"{best[1]['overall_score']:.0f}/100")
    with col2:
        if fastest_robust:
            st.metric("⚡ Plus rapide (robuste)", fastest_robust[1]["name"], fastest_robust[1]["time_to_60C"])
        else:
            st.metric("⚡ Plus rapide", "Aucun", "Contrainte non respectée")
    with col3:
        violated = [(k, v) for k, v in comparison.items() if v["constraint_violation_C"] > 0]
        if violated:
            st.metric("⚠️ Violation", str(len(violated)), f"+{violated[0][1]['constraint_violation_C']:.1f}°C")
        else:
            st.metric("⚠️ Violation", "0", "Tous OK")

    st.divider()
    st.subheader("📋 Analyse Détaillée")

    for name, metrics in comparison.items():
        with st.expander(f"{metrics['name']}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Temps à 60°C:** {metrics['time_to_60C']}")
                st.write(f"**Énergie:** {metrics['total_energy_MJ']} MJ")
                st.write(f"**Puiss. moy.:** {metrics['avg_power_kW']} kW")
            with col_b:
                st.write(f"**Contrainte eau:** {metrics['constraint_ok']}")
                st.write(f"**Débit eau:** {metrics['avg_water_flow_th']} t/h")
                st.write(f"**Score:** {metrics['overall_score']:.0f}/100")
            st.progress(min(1.0, max(0.0, metrics['overall_score']/100)),
                        text=f"Perf: {metrics['overall_score']:.0f}%")

    st.divider()
    if fastest_robust:
        st.success(f"**Recommandation:** Mode *'{fastest_robust[1]['name']}'* - Atteint 60°C en {fastest_robust[1]['time_to_60C']}, contrainte respectée.")
    else:
        st.warning("**Attention:** Aucun mode ne respecte la contrainte T° eau. Considerer: augmenter débit eau ou surface'échange.")


def export_csv(results_dict):
    rows = []
    for name, results in results_dict.items():
        for i in range(len(results.t)):
            rows.append({"mode": name, "time_s": results.t[i], "T_tank_C": results.T_tank[i],
                         "T_water_out_C": results.T_water_out[i], "Q_W": results.Q_W[i], "energy_J": results.energy_J[i]})
    return pd.DataFrame(rows).to_csv(index=False)


def render_audio_control():
    audio = get_audio_manager()
    col1, col2 = st.columns([1, 2])
    with col1:
        icon = "🔊" if not audio.is_muted() else "🔇"
        if st.button(icon, key="audio_toggle", use_container_width=True):
            new_state = toggle_audio()
            st.toast("🔇 Son désactivé" if new_state else "🔊 Son activé")
            st.rerun()
    with col2:
        vol = st.selectbox("Volume", ["Off", "Low", "Medium"], index=2, key="vol")
        audio.set_volume({"Off": "off", "Low": "low", "Medium": "medium"}[vol])


def main():
    # Inject audio system (hidden)
    inject_audio_manager()

    st.title("🏭 Industrial Batch Cooling Simulator")
    st.markdown("**Contexte:** Refroidissement batch 45t de 150°C→60°C via échangeur tubes/calandre. Comparaison 4 architectures.")

    # Audio controls
    with st.sidebar:
        st.subheader("🔊 Audio")
        render_audio_control()

    # Preset button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🏭 Industrial Case (45t)", use_container_width=True):
            play_ui_sound("tick")
            st.session_state.preset = industrial_preset()
            st.rerun()

    # Initialize preset
    if 'preset' not in st.session_state:
        st.session_state.preset = {
            "batch_mass": 500.0, "Cp_product": 2200.0, "density": 900.0,
            "T_initial": 150.0, "recirculation_flow": 1.0,
            "water_flow": 10.0, "water_inlet": 28.0, "water_max_outlet": 36.0,
            "exchanger_area": 10.0, "U_value": 500.0, "max_batch_time": 3600.0
        }
    p = st.session_state.preset

    # === PFD Schematic ===
    st.subheader("📊 Schéma Procédé (PFD)")
    fig_pfd = render_pfd_schematic()
    st.plotly_chart(fig_pfd, use_container_width=True)

    st.divider()

    # === Parameter Blocks ===
    st.subheader("🔧 Configuration des Paramètres")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**📦 Cuve (Produit)**")
        batch_mass = st.number_input("Masse [kg]", 100.0, 100000.0, p["batch_mass"], 100.0)
        Cp = st.number_input("Cp [J/kg·K]", 1000.0, 4000.0, p["Cp_product"], 50.0)
        T_init = st.number_input("T° initiale [°C]", 50.0, 250.0, p["T_initial"], 5.0)
        recirculation = st.number_input("Débit recircul. [kg/s]", 1.0, 200.0, p["recirculation_flow"], 1.0)

    with col2:
        st.markdown("**🔥 Échangeur**")
        A = st.number_input("Surface [m²]", 1.0, 300.0, p["exchanger_area"], 1.0)
        U = st.number_input("U [W/m²·K]", 100.0, 2000.0, p["U_value"], 50.0)

    with col3:
        st.markdown("**💧 Eau Refroidissement**")
        water_flow = st.number_input("Débit [kg/s]", 1.0, 500.0, p["water_flow"], 1.0)
        T_water_in = st.number_input("T° entrée [°C]", 15.0, 40.0, p["water_inlet"], 1.0)
        T_water_max = st.number_input("T° sortie max [°C]", 30.0, 50.0, p["water_max_outlet"], 1.0)

    with col4:
        st.markdown("**⏱️ Contraintes**")
        target_temp = st.number_input("T° cible [°C]", 30.0, 120.0, 60.0, 5.0)
        max_time = st.number_input("Temps max [s]", 300.0, 7200.0, p["max_batch_time"], 60.0)
        st.write(f"**Max: {max_time/60:.0f} min**")

        mode_options = ["Tous les modes", "Mode 1 - Sans contrôle", "Mode 2 - Contrôle débit eau",
                       "Mode 3 - Bypass produit", "Mode 4 - Passage unique"]
        selected_mode = st.selectbox("Mode", mode_options)

    # Update preset
    st.session_state.preset = {
        "batch_mass": batch_mass, "Cp_product": Cp, "density": 900.0,
        "T_initial": T_init, "recirculation_flow": recirculation,
        "water_flow": water_flow, "water_inlet": T_water_in, "water_max_outlet": T_water_max,
        "exchanger_area": A, "U_value": U, "max_batch_time": max_time
    }

    st.divider()

    # === Run Simulation ===
    col_run, col_exp = st.columns([1, 4])
    with col_run:
        run_clicked = st.button("▶️ Lancer Simulation", type="primary", use_container_width=True)
    with col_exp:
        if st.session_state.get('results') and st.button("📊 Exporter CSV", use_container_width=True):
            csv = export_csv(st.session_state.results)
            st.download_button("Download CSV", csv, "batch_cooling.csv", "text/csv")

    if run_clicked:
        play_ui_sound("start")

        with st.spinner("Simulation en cours..."):
            batch = ProductBatch(mass=batch_mass, Cp=Cp, density=900.0,
                                initial_temp=T_init, recirculation_flow=recirculation)
            water = CoolingWater(flow_rate=water_flow, inlet_temp=T_water_in, max_outlet_temp=T_water_max)
            exchanger = HeatExchanger(area=A, U=U)
            config = SimulationConfig(batch=batch, water=water, exchanger=exchanger,
                                      max_batch_time=max_time, target_temp=target_temp)

            if selected_mode == "Tous les modes":
                results_dict = run_all_modes(config)
            else:
                arch_map = {
                    "Mode 1 - Sans contrôle": Architecture.RECIRC_NO_CONTROL,
                    "Mode 2 - Contrôle débit eau": Architecture.RECIRC_WATER_CONTROL,
                    "Mode 3 - Bypass produit": Architecture.RECIRC_BYPASS,
                    "Mode 4 - Passage unique": Architecture.SINGLE_PASS
                }
                cfg = SimulationConfig(batch=batch, water=water, exchanger=exchanger,
                                      max_batch_time=max_time, target_temp=target_temp,
                                      mode=arch_map[selected_mode])
                sim = BatchCoolingSimulator(cfg)
                results_dict = {selected_mode.split(" - ")[0].lower().replace(" ", "_").replace("mode_", "mode"): sim.run()}

            # Feedback sounds
            all_ok = all(r.constraint_satisfied for r in results_dict.values())
            any_reached = any(r.time_to_60C_s is not None for r in results_dict.values())

            if all_ok and any_reached:
                play_ui_sound("success")
                st.success("✅ Simulation terminée - Objectif atteint!")
            elif not all_ok:
                play_ui_sound("error")
                st.error("⚠️ Contrainte eau violée!")
            else:
                play_ui_sound("complete")

            st.session_state.results = results_dict
            st.session_state.comparison = create_comparison_dataframe(results_dict, max_time)

    # === Display Results ===
    if st.session_state.get('results'):
        results_dict = st.session_state.results
        comparison = st.session_state.comparison

        st.divider()
        st.subheader("📈 Métriques Clés")

        cols = st.columns(len(results_dict))
        for col, (name, results) in zip(cols, results_dict.items()):
            with col:
                t = f"{results.time_to_60C_s:.0f}s" if results.time_to_60C_s else "—"
                st.metric(MODE_NAMES.get(name, name), t, f"{results.avg_power_kW:.0f} kW moy")

        st.write("**Contrainte eau (≤36°C):**")
        ccols = st.columns(len(results_dict))
        for col, (name, results) in zip(ccols, results_dict.items()):
            with col:
                if results.constraint_satisfied:
                    st.success(f"✅ {MODE_NAMES.get(name, name)}")
                else:
                    play_ui_sound("warning")
                    st.error(f"❌ +{results.constraint_violation_max_C:.1f}°C")

        st.divider()
        st.subheader("📉 Graphiques")

        tabs = st.tabs(["Vue Overlay", "T° Cuve", "T° Eau", "Puissance"])

        with tabs[0]:
            fig = plot_overlay_all(results_dict, target_temp, T_water_max)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[1]:
            fig = plot_temperature_evolution(results_dict, target_temp)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[2]:
            fig = plot_water_temperature(results_dict, T_water_max)
            st.plotly_chart(fig, use_container_width=True)
        with tabs[3]:
            fig = plot_power_and_energy(results_dict)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("📊 Tableau Comparatif")
        display_comparison_table(comparison, max_time)
        display_recommendation(comparison)

        st.divider()
        st.subheader("📋 Données Détaillées")
        for name, results in results_dict.items():
            with st.expander(f"Données - {MODE_NAMES.get(name, name)}"):
                df = pd.DataFrame({
                    "t [s]": results.t, "T_cuve [°C]": results.T_tank,
                    "T_eau [°C]": results.T_water_out, "Q [kW]": results.Q_W/1000,
                    "Énergie [MJ]": results.energy_J/1e6
                })
                st.dataframe(df, use_container_width=True, height=200)

    else:
        st.divider()
        st.info("👆 Configurez les paramètres et cliquez sur 'Lancer Simulation'")
        st.markdown("**Mode d'emploi:** 1) Utilisez le preset Industrial Case  2) Lancez  3) Comparez les 4 modes")


if __name__ == "__main__":
    main()