# pmsm_plotter.py
import matplotlib.pyplot as plt
import pandas as pd


def plot_pmsm_results(df: pd.DataFrame):
    t = df["t"]

    plt.figure(figsize=(12, 8))

    # currents
    plt.subplot(4, 1, 1)
    plt.plot(t, df["i_d_meas"], label="i_d_meas")
    plt.plot(t, df["i_q_meas"], label="i_q_meas")
    if "i_q_ref" in df.columns:
        plt.plot(t, df["i_q_ref"], "--", label="i_q_ref")
    plt.ylabel("Current [A]")
    plt.legend()
    plt.grid(True)

    # speed
    plt.subplot(4, 1, 2)
    plt.plot(t, df["omega_m"], label="omega_m")
    plt.plot(t, df["omega_m_meas"], label="omega_m_meas")
    if "omega_ref" in df.columns and df["omega_ref"].notna().any():
        plt.plot(t, df["omega_ref"], "--", label="omega_ref")
    plt.ylabel("Speed [rad/s]")
    plt.legend()
    plt.grid(True)

    # torque & load
    plt.subplot(4, 1, 3)
    plt.plot(t, df["torque_e"], label="torque_e")
    plt.plot(t, df["torque_load"], "--", label="torque_load")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.legend()
    plt.grid(True)

    # Ud, Uq
    plt.subplot(4, 1, 4)
    plt.plot(t, df["v_d"], label="v_d")
    plt.plot(t, df["v_q"], label="v_q")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid(True)


    plt.tight_layout()
    plt.show()
