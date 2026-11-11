# ============================================================================
# cooling_scheduler_copy/live_visualizer.py
# Visualization for the real-data rolling mode (--live)
#
# Splits the 24-hour round-by-round run results into 5 separate figures,
# polished one by one so that non-experts can understand them quickly:
#   Fig 1  Electrolyte temperature (comfort zone / safety lines /
#          cooling periods / ambient comparison)
#   Fig 2  Cooling ON/OFF timeline (Gantt chart, colored by valley/peak tariff)
#   Fig 3  Time-of-use electricity price (color coded + value labels)
#   Fig 4  Electricity cost (hourly bars + cumulative curve)
#   Fig 5  SoC state of charge (background shading for charge/discharge periods)
#
# Depends on: logs produced by live_runner
# ============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------- Unified color scheme (consistent with project) -------------------------
TARIFF_BAR = {"谷": "#2ecc71", "平": "#95a5a6", "峰": "#e67e22", "尖峰": "#e74c3c"}
TARIFF_LIGHT = {"谷": "#a9dfbf", "平": "#d5dbdb", "峰": "#f5cba7", "尖峰": "#f1948a"}
# English display names for tariff tiers (keys stay in Chinese to match config labels)
TARIFF_EN = {"谷": "Valley", "平": "Flat", "峰": "Peak", "尖峰": "Critical Peak"}
TEMP_LINE = "#e74c3c"      # Electrolyte temperature
AMB_LINE = "#3498db"       # Ambient temperature
COMFORT = "#2ecc71"        # Comfort zone
COOL_ON = "#3498db"        # Cooling ON
SOC_LINE = "#16a085"       # SoC
COST_BAR = "#9b59b6"       # Hourly electricity cost
COST_CUM = "#e67e22"       # Cumulative electricity cost


def _setup():
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                           "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _extract(logs):
    n = len(logs)
    idx = np.arange(n)
    d = {
        "n": n, "idx": idx,
        "T": np.array([r["T_ely"] for r in logs], float),
        "action": np.array([r["action"] for r in logs], int),
        "price": np.array([r["price"] if r["price"] != "" else np.nan
                           for r in logs], float),
        "step_cost": np.array([r["step_cost"] for r in logs], float),
        "cum_cost": np.array([r["cum_cost"] for r in logs], float),
        "soc": np.array([r["SoC"] for r in logs], float) * 100,
        "day_h": np.array([r["day_h"] for r in logs], int),
        "amb": np.array([r.get("ambient") if r.get("ambient", "") != "" else np.nan
                         for r in logs], float),
        "labels": [r["abs_dt"][11:16] for r in logs],
    }
    return d


def _xaxis(ax, d, rotation=45):
    ax.set_xticks(d["idx"])
    ax.set_xticklabels(d["labels"], rotation=rotation, ha="right", fontsize=8)
    ax.set_xlim(-0.5, d["n"] - 0.5)
    ax.grid(alpha=0.22, axis="x")


def _tariff_of(config, hour):
    try:
        return config.get_tariff_label(int(hour))
    except Exception:
        return "平"


# ============================== Fig 1: Temperature ==============================
def _fig_temperature(d, config, out):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    T_low = T_high = None
    if config is not None:
        T_low = config.optimizer.get("T_target_low", 25)
        T_high = config.optimizer.get("T_target_high", 35)

    # Light shading for cooling periods
    for i in range(d["n"]):
        if d["action"][i] == 1:
            ax.axvspan(i - 0.5, i + 0.5, color=COOL_ON, alpha=0.10)

    # Comfort zone
    if T_low is not None and T_high is not None:
        ax.axhspan(T_low, T_high, color=COMFORT, alpha=0.13,
                   label=f"Comfort zone {T_low:.0f}~{T_high:.0f}°C")
    # Safety lines
    ax.axhline(38, color="#f85149", ls="--", lw=1.3, alpha=0.8,
               label="Over-temp forced ON 38°C")
    ax.axhline(18, color="#58a6ff", ls="--", lw=1.3, alpha=0.8,
               label="Low-temp forced OFF 18°C")

    # Ambient temperature
    if not np.all(np.isnan(d["amb"])):
        ax.plot(d["idx"], d["amb"], color=AMB_LINE, ls="--", lw=1.8,
                marker=".", ms=6, label="Ambient temp (forecast)")
    # Electrolyte temperature
    ax.plot(d["idx"], d["T"], color=TEMP_LINE, lw=2.6, marker="o", ms=6,
            label="Electrolyte temperature", zorder=5)

    imax = int(np.argmax(d["T"]))
    ax.annotate(f"Max {d['T'][imax]:.1f}°C\n{d['labels'][imax]}",
                xy=(imax, d["T"][imax]), xytext=(imax + 0.4, d["T"][imax] + 1.1),
                color=TEMP_LINE, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEMP_LINE, lw=1.3))

    n_on = int(d["action"].sum())
    ax.set_title(f"Electrolyte Temperature —— Cooling ON for {n_on} hours in total, "
                 f"temperature kept within {d['T'].min():.1f}~{d['T'].max():.1f}°C",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Temperature [°C]", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    _xaxis(ax, d)
    ax.set_xlabel("Time", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================== Fig 2: Cooling schedule ==============================
def _fig_schedule(d, config, out):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    on_idx = [i for i in range(d["n"]) if d["action"][i] == 1]

    # Tariff-tier color band at the top
    if config is not None:
        for i in range(d["n"]):
            tag = _tariff_of(config, d["day_h"][i])
            ax.axvspan(i - 0.5, i + 0.5, ymin=0.92, ymax=1.0,
                       color=TARIFF_BAR.get(tag, "#95a5a6"), alpha=0.9)

    # ON hours colored by tariff tier
    for i in on_idx:
        tag = _tariff_of(config, d["day_h"][i]) if config is not None else "平"
        ax.barh(0.45, 0.92, left=i - 0.46, height=0.62,
                color=TARIFF_BAR.get(tag, "#95a5a6"),
                edgecolor="white", lw=1.2)
        ax.text(i, 0.45, "ON", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")

    ax.set_yticks([0.45])
    ax.set_yticklabels(["Cooling"], fontsize=11)
    ax.set_ylim(0, 0.95)
    ax.set_xlim(-0.5, d["n"] - 0.5)

    valley_on = sum(1 for i in on_idx
                    if config is not None and _tariff_of(config, d["day_h"][i]) == "谷")
    peak_on = sum(1 for i in on_idx if config is not None
                  and _tariff_of(config, d["day_h"][i]) in ("峰", "尖峰"))
    ax.set_title(f"Cooling ON/OFF Schedule —— {len(on_idx)} hours ON in total "
                 f"({valley_on}h valley pre-cooling, {peak_on}h on-demand at peak)",
                 fontsize=14, fontweight="bold", pad=18)
    handles = [mpatches.Patch(color=TARIFF_BAR["谷"], label="ON in valley (cheap · pre-cooling)"),
               mpatches.Patch(color=TARIFF_BAR["平"], label="ON in flat period"),
               mpatches.Patch(color=TARIFF_BAR["峰"], label="ON at peak (expensive · on-demand)"),
               mpatches.Patch(color="#3a4048", label="OFF")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, -0.28),
              fontsize=9, ncol=4)
    ax.grid(alpha=0.22, axis="x")
    ax.set_xticks(d["idx"])
    ax.set_xticklabels(d["labels"], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================== Fig 3: Electricity price ==============================
def _fig_price(d, config, out):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(d["n"]):
        if np.isnan(d["price"][i]):
            continue
        tag = _tariff_of(config, d["day_h"][i]) if config is not None else "平"
        ax.bar(i, d["price"][i], color=TARIFF_BAR.get(tag, "#95a5a6"),
               edgecolor="none")
        ax.text(i, d["price"][i] + 0.02 * np.nanmax(d["price"]),
                f"{d['price'][i]:.2f}", ha="center", fontsize=7.5,
                color="#333")

    # ipk = int(np.nanargmax(d["price"]))
    # ax.annotate(f"Most expensive {d['price'][ipk]:.2f}", xy=(ipk, d["price"][ipk]),
    #             xytext=(ipk, d["price"][ipk] * 1.12), ha="center",
    #             color=TARIFF_BAR.get(_tariff_of(config, d["day_h"][ipk]), "#333"),
    #             fontsize=10.5, fontweight="bold")

    handles = [mpatches.Patch(color=c, label=TARIFF_EN.get(k, k))
               for k, c in TARIFF_BAR.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=10, ncol=4)
    ax.set_title("Time-of-Use Electricity Price —— cheap in the valley, "
                 "expensive at peak; shift cooling to cheap periods whenever possible",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Price [CNY/kWh]", fontsize=11)
    ax.grid(alpha=0.22, axis="y")
    _xaxis(ax, d)
    ax.set_xlabel("Time", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================== Fig 4: Cost ==============================
def _fig_cost(d, config, out):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i in range(d["n"]):
        tag = _tariff_of(config, d["day_h"][i]) if config is not None else "平"
        ax.bar(i, d["step_cost"][i],
               color=TARIFF_BAR.get(tag, COST_BAR), alpha=0.85)
    ax.set_ylabel("Hourly cost [CNY]", fontsize=11)
    ax.grid(alpha=0.22, axis="y")

    ax2 = ax.twinx()
    ax2.plot(d["idx"], d["cum_cost"], color=COST_CUM, lw=2.6,
             marker="s", ms=5, label="Cumulative cost")
    ax2.set_ylabel("Cumulative cost [CNY]", color=COST_CUM, fontsize=11)
    ax2.grid(False)

    total = d["cum_cost"][-1]
    ax2.annotate(f"Total cost {total:.2f} CNY", xy=(d["n"] - 1, total),
                 xytext=(d["n"] - 3.2, total * 1.06), color=COST_CUM,
                 fontsize=12, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COST_CUM, lw=1.3))

    n_charge = int((d["step_cost"] > 0).sum())
    ax.set_title(f"Cooling Electricity Cost —— {total:.2f} CNY in total over 24 hours "
                 f"(cost incurred in {n_charge} hours)",
                 fontsize=14, fontweight="bold", pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, ["Hourly cost (colored by tariff tier)"] + l2,
              loc="upper left", fontsize=10)
    _xaxis(ax, d)
    ax.set_xlabel("Time", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================== Fig 5: SoC ==============================
def _fig_soc(d, config, out):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    # Background shading for charge/discharge periods (10-17 charging / 18-06 discharging)
    for i in range(d["n"]):
        h = int(d["day_h"][i])
        if 10 <= h < 17:
            ax.axvspan(i - 0.5, i + 0.5, color="#2ecc71", alpha=0.10)
        elif h >= 18 or h < 6:
            ax.axvspan(i - 0.5, i + 0.5, color="#e67e22", alpha=0.10)

    ax.axhline(90, color="#888", ls=":", lw=1.4)
    ax.axhline(10, color="#888", ls=":", lw=1.4)
    ax.text(d["n"] - 0.4, 91.5, "SoC upper limit 90%", ha="right", fontsize=8.5, color="#666")
    ax.text(d["n"] - 0.4, 5.5, "SoC lower limit 10%", ha="right", fontsize=8.5, color="#666")

    ax.plot(d["idx"], d["soc"], color=SOC_LINE, lw=2.6, marker="o", ms=6)
    for i in range(d["n"]):
        if i % 3 == 0:
            ax.text(i, d["soc"][i] + 3, f"{d['soc'][i]:.0f}%",
                    ha="center", fontsize=8, color=SOC_LINE)

    handles = [mpatches.Patch(color="#2ecc71", alpha=0.25, label="Charging period (10-17)"),
               mpatches.Patch(color="#e67e22", alpha=0.25, label="Discharging period (18-06)")]
    ax.legend(handles=handles, loc="upper left", fontsize=10, ncol=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"SoC State of Charge —— from {d['soc'][0]:.0f}% to {d['soc'][-1]:.0f}%",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("SoC [%]", fontsize=11)
    ax.grid(alpha=0.22, axis="y")
    _xaxis(ax, d)
    ax.set_xlabel("Time", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================== Main entry ==============================
def plot_live_figures(logs, config=None, out_dir="output", tag=""):
    """Generate the 5 separate figures and return a dict of paths."""
    _setup()
    d = _extract(logs)
    os.makedirs(out_dir, exist_ok=True)
    date = logs[0]["abs_dt"][:10].replace("-", "")
    base = os.path.join(out_dir, f"live{tag}_{date}")

    paths = {}
    _fig_temperature(d, config, base + "_1_temperature.png")
    paths["temperature"] = base + "_1_temperature.png"
    _fig_schedule(d, config, base + "_2_schedule.png")
    paths["schedule"] = base + "_2_schedule.png"
    _fig_price(d, config, base + "_3_price.png")
    paths["price"] = base + "_3_price.png"
    _fig_cost(d, config, base + "_4_cost.png")
    paths["cost"] = base + "_4_cost.png"
    _fig_soc(d, config, base + "_5_soc.png")
    paths["soc"] = base + "_5_soc.png"
    return paths


# # Legacy interface (single composite figure), kept for backward compatibility
# def plot_live_summary(logs, config=None, start_label="", save_path=None):
#     _setup()
#     d = _extract(logs)
#     if save_path is None:
#         os.makedirs("output", exist_ok=True)
#         save_path = os.path.join(
#             "output",
#             f"live_rolling_summary_{logs[0]['abs_dt'][:10].replace('-', '')}.png")
#     fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True,
#                              gridspec_kw={"height_ratios": [3, 1.2, 1.8, 2, 1.5],
#                                           "hspace": 0.3})
#     ax = axes[0]
#     ax.plot(d["idx"], d["T"], color=TEMP_LINE, marker="o", lw=2)
#     ax.axhspan(25, 35, color=COMFORT, alpha=0.13)
#     ax.set_ylabel("Temp [°C]")
#     ax.set_title("Electrolyte Temperature (composite overview)")
#     ax.grid(alpha=0.25)
#     ax = axes[1]
#     for i in range(d["n"]):
#         if d["action"][i]:
#             ax.barh(0.5, 1.0, left=i - 0.5, height=0.7, color=COOL_ON)
#     ax.set_yticks([0.5]); ax.set_yticklabels(["Cooling"])
#     ax.set_ylim(0, 1); ax.grid(alpha=0.25, axis="x")
#     ax = axes[2]
#     ax.bar(d["idx"], d["price"], color="#95a5a6")
#     ax.set_ylabel("Price")
#     ax.grid(alpha=0.25, axis="y")
#     ax = axes[3]
#     ax.bar(d["idx"], d["step_cost"], color=COST_BAR, alpha=0.75)
#     ax2 = ax.twinx()
#     ax2.plot(d["idx"], d["cum_cost"], color=COST_CUM, marker="s", lw=2)
#     ax.set_ylabel("Cost")
#     ax = axes[4]
#     ax.plot(d["idx"], d["soc"], color=SOC_LINE, marker="o", lw=2)
#     ax.set_ylabel("SoC [%]"); ax.set_ylim(0, 100); ax.grid(alpha=0.25)
#     axes[-1].set_xticks(d["idx"])
#     axes[-1].set_xticklabels(d["labels"], rotation=45, ha="right", fontsize=8)
#     axes[-1].set_xlim(-0.5, d["n"] - 0.5)
#     fig.savefig(save_path, dpi=130, bbox_inches="tight")
#     plt.close(fig)
#     return save_path
