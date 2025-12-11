# pmsm_plotter.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_pmsm_results(df: pd.DataFrame):
    t = df["t"]

    fig = plt.figure(figsize=(14, 10))

    # currents
    ax1 = plt.subplot(6, 1, 1)
    #ax1.plot(t, df["i_d_meas"], label="i_d_meas")
    ax1.plot(t, df["i_d"], label="i_d")
    ax1.plot(t, df["i_q"], label="i_q")
    if "i_q_ref" in df.columns:
        ax1.plot(t, df["i_q_ref"], "--", label="i_q_ref")
    ax1.set_ylabel("Current [A]")
    ax1.legend()
    ax1.grid(True)

    # speed
    ax2 = plt.subplot(6, 1, 2)
    ax2.plot(t, df["omega_e"], label="omega_e (true)")
    #ax2.plot(t, df["omega_e_meas"], label="omega_e (meas)")
    ax2.plot(t, df["omega_ref"], "--", label="omega_ref")
    ax2.set_ylabel("Speed [rad/s]")
    ax2.legend()
    ax2.grid(True)

    # angle
    ax3 = plt.subplot(6, 1, 3)
    ax3.plot(t, df["theta_e"], label="θ_e (electrical)")
    ax3.set_ylabel("Angle [rad]")
    ax3.set_xlabel("Time [s]")
    ax3.legend()
    ax3.grid(True)

    ax3 = plt.subplot(6, 1, 4)
    ax3.plot(t, df["sin_theta_e"], label="sin theta (electrical) true")
    ax3.plot(t, df["cos_theta_e"], label="cos theta (electrical)")
    ax3.set_ylabel("Angle [rad]")
    ax3.set_xlabel("Time [s]")
    ax3.legend()
    ax3.grid(True)

    # torque & load
    ax4 = plt.subplot(6, 1, 5)
    ax4.plot(t, df["torque_e"], label="torque_e")
    ax4.plot(t, df["torque_load"], "--", label="torque_load")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Torque [Nm]")
    ax4.legend()
    ax4.grid(True)

    # Ud, Uq
    ax5 = plt.subplot(6, 1, 6)
    ax5.plot(t, df["v_d"], label="v_d")
    ax5.plot(t, df["v_q"], label="v_q")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Voltage [V]")
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    plt.show()