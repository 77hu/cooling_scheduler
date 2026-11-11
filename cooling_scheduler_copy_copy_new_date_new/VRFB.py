#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRFB THERMAL MANAGEMENT SIMULATION v5.2
=========================================
Complete thermal management model for a containerized Vanadium Redox Flow
Battery (VRFB), extended to run and compare TWO independent scenarios in a
single execution.

Structure of this version (v5.2), vs. v5.1 / v5.0 / v4.1:
  - All physics (thermal + electrochemical) is now wrapped in a single
    reusable function `run_simulation(scenario)` so that any number of
    scenarios can be evaluated with identical, verified logic.
  - Each scenario bundles its own environment/solar profile, container
    insulation state, and battery/stack parameters, so scenarios can differ
    in climate, insulation, battery sizing, or any combination thereof
    (e.g. "same battery, Shenzhen summer vs. Shenyang winter", or
    "same climate, full insulation vs. partial insulation").
  - Container insulation is now a simple selectable OPTION per scenario:
    INSULATION_FULL or INSULATION_PARTIAL (one long side left as bare,
    uninsulated steel plate).
  - The original v4.1 single-scenario report (console summary, plots, CSV
    export) is preserved unchanged in content and is produced once per
    scenario (i.e. twice, if two scenarios are defined).
  - A new SECTION 8 adds a side-by-side comparison of both scenarios
    (energy balance, cooling/heating demand, cost, round-trip efficiency).

Physics Models (retained from v4.1, incl. all v4.0 -> v4.1 corrections):
  - Arrhenius temperature-dependent viscosity and stack resistance
  - Ohmic + reversible (entropic) stack heat generation (v3.1 sign convention)
  - SoC integration via Faraday's law
  - SoC-proportional shunt / crossover losses
  - T_stack: independent thermal model (Newton cooling vs. inner-air node)
  - T_innen: algebraic inner-container climate model, properly coupled into
    the electrolyte energy balance (two-resistor path: ambient -> inner air
    -> electrolyte)
  - Cooling control: TOU-based state machine (State A: peak / State B:
    off-peak), re-evaluates tariff state on every state transition
  - Agitator: 10% PWM duty cycle during active thermal or charge/discharge
    operations
  - Stack flushing: asynchronous warm-keeping sub-routine
  - Optional BMS-controlled ventilation fan (reduces sol-air heat ingress at
    the container skin)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import os
import math

try:
    _BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _BASE = os.path.abspath("output")
os.makedirs(_BASE, exist_ok=True)


# ================================================================================
# SECTION 1 - INSULATION OPTIONS
# Two selectable container envelope configurations. "PARTIAL" leaves one long
# side of a 20 ft container (approx. 15.7 m^2 out of 60 m^2 total envelope)
# as bare, uninsulated steel plate, raising the effective wall U-value for
# that fraction of the surface area from 0.5 to 6.5 W/m^2K.
# ================================================================================

INSULATION_FULL = "full"
INSULATION_PARTIAL = "partial"

A_TOTAL_ENVELOPE = 60.0     # [m^2] total container envelope area (roof + 4 walls)
A_BARE_LONG_SIDE = 15.7     # [m^2] one long side of a 20 ft container
U_INSULATED = 0.5           # [W/m^2K] effective U-value of insulated envelope
U_BARE_STEEL = 6.5          # [W/m^2K] effective U-value of bare steel plate

def get_wall_U(insulation_option):
    """Return the effective container-wall U-value [W/m^2K] for the chosen
    insulation option, applied uniformly across the whole envelope area."""
    if insulation_option == INSULATION_FULL:
        return U_INSULATED
    elif insulation_option == INSULATION_PARTIAL:
        UA_full = U_INSULATED * A_TOTAL_ENVELOPE
        UA_partial = (U_INSULATED * (A_TOTAL_ENVELOPE - A_BARE_LONG_SIDE)
                      + U_BARE_STEEL * A_BARE_LONG_SIDE)
        return UA_partial / A_TOTAL_ENVELOPE
    else:
        raise ValueError(f"Unknown insulation option: {insulation_option}")


# ================================================================================
# SECTION 2 - SCENARIO DEFINITIONS
# Each scenario is a self-contained dictionary of environment, solar, and
# battery/system parameters, plus an insulation option and TOU tariff table.
# Two example scenarios are defined below for direct comparison; swap in any
# other climate/insulation/battery combination as needed.
# ================================================================================

def make_scenario(name, T_max, T_min, I_max, sunrise, sunset, insulation,
                   tech_id=1, ekes_scale=1.0, ekes_power_model="COP",
                   days=14, dt=60, vent_active=True):
    """Assemble one complete, self-consistent scenario configuration.

    tech_id: 1 = Plate heat exchanger + external chiller (indirect loop,
             heat exchange works without main pump circulation).
             2 = EKES tube-bundle heat exchanger, installed directly in
             the primary electrolyte loop (main pumps MUST run whenever
             cooling is active, since heat exchange only occurs while
             electrolyte is circulating).
    ekes_scale: linear capacity scaling factor applied to the EKES
             reference unit (see hex.ekes_*_ref for example values of
             the EKES60/100/150 product family).
    """
    return {
        "name": name,
        "insulation": insulation,
        "tech_id": tech_id,
        "ekes_scale": ekes_scale,
        "ekes_power_model": ekes_power_model,
        "env": {"T_max": T_max, "T_min": T_min},
        "solar": {"I_max": I_max, "alpha": 0.6, "h_ext": 35,
                  "sunrise": sunrise, "sunset": sunset},
        "sim": {"days": days, "dt": dt},
        "vent_active": vent_active,
        # --- Physics & container geometry (identical battery in both scenarios
        #     by default; override per-scenario if a different battery size is
        #     to be compared) ---
        "phys": {"V_total": 14000, "rho": 1.35, "cp": 3200},
        "container": {"A_tank_surface": 40, "U_tank_inner": 1.5,
                      "A_wall_inner": A_TOTAL_ENVELOPE},
        "stack": {"active": True, "P_nom": 45000, "Cap_nom": 270000,
                  "SoC_min": 0.10, "SoC_max": 0.90, "SoC_start": 0.50,
                  "start_charge_h": 10.0, "start_disch_h": 18.0,
                  "R_ref": 0.015, "Ea_R": 2500, "U_OCV_ref": 55.0,
                  "dS_coeff": 0.05, "P_shunt_W": 450.0},
        "pump": {"ratio_rated": 0.049, "ratio_idle": 0.007, "Ea_visc": 1800},
        "hex": {"active": True, "plate_UA": 544, "plate_T_in": 7.0,
                "plate_P_max": 14000, "plate_COP": 2.9, "LMTD_corr": 0.9,
                "COP_deg": 0.08,
                # --- EKES tube-bundle direct cooling (primary electrolyte loop) ---
                # Consolidated single reference unit, scaled via ekes_scale.
                # Example nameplate values for the EKES product family
                # (for reference only, no longer modeled as discrete cases):
                #   EKES60:  Q_nom=6000  W, P_elec=1300 W, COP=4.61 (scale=0.40)
                #   EKES100: Q_nom=10000 W, P_elec=2200 W, COP=4.55 (scale=0.67)
                #   EKES150: Q_nom=15000 W, P_elec=4100 W, COP=3.65 (scale=1.00)
                "ekes_Q_nom_ref": 15000.0, "ekes_P_elec_ref": 4100.0,
                "ekes_COP_ref": 3.65},
        "cool": {"T_A_on": 35.0, "T_A_off": 32.0, "T_B_on": 28.0,
                 "T_B_off": 27.0, "min_on_s": 300, "min_off_s": 180},
        "heat": {"active": True, "P_heat_W": 5000.0, "COP_heat": 1.0,
                 "T_on": 12.0, "T_off": 15.0},
        "agitator": {"active": True, "P_agit_W": 300, "period_s": 600,
                     "pulse_s": 60},
        "stack_thermal": {"m_metal_kg": 150.0, "cp_metal": 500.0,
                       "m_plastic_kg": 350.0, "cp_plastic": 1200.0,
                       "V_ely_L": 150.0, "dim_m": [0.877, 0.589, 0.820],
                       "U_eff_stack": 3.5, "T_flush_trigger": 10.0},
        "flush_cfg": {"duration_s": 120},
        "vent": {"Flow_m3h": 2000, "P_ref": 150, "Flow_ref": 2000,
                 "dT_trigger_on": 3.0, "dT_trigger_off": 1.0},
        # --- TOU electricity tariffs: Shenzhen large-industrial/commercial
        #     (10 kV) rates, valley:flat:peak = 0.3142:1:1.5445, critical =
        #     peak x 1.25 ---
        "price": {"valley": 0.23, "flat": 0.72, "peak": 1.11, "critical": 1.39,
                  "default": 0.72,
                  "windows": np.array([
                      [0.0, 8.0, 1], [8.0, 10.0, 2], [10.0, 12.0, 3],
                      [12.0, 14.0, 2], [14.0, 19.0, 3], [19.0, 24.0, 2],
                  ])},
    }


# --- Example scenario A: Shenzhen summer, fully insulated container ---
SCENARIO_A = make_scenario(
    name="Shenzhen Summer - Plate-HEX (Indirect, tech_id=1)",
    T_max=35, T_min=27, I_max=850, sunrise=5.75, sunset=19.17,
    insulation=INSULATION_FULL, tech_id=1, vent_active=True
)

# --- Example scenario B: identical climate, insulation and battery as
#     Scenario A, but EKES tube-bundle direct cooling (tech_id=2) instead
#     of the plate heat exchanger -- isolates the cooling-technology effect
#     on main-pump runtime and energy allocation. ---
SCENARIO_B = make_scenario(
    name="Shenzhen Summer - Plate-HEX (Indirect, tech_id=1)",
    T_max=35, T_min=27, I_max=850, sunrise=5.75, sunset=19.17,
    insulation=INSULATION_FULL, tech_id=1, vent_active=True
)

# NOTE: to reproduce a "same battery, different climate" comparison instead
# (e.g. Shenzhen summer vs. Shenyang winter), simply redefine SCENARIO_B with
# a different T_max/T_min/I_max/sunrise/sunset combination while keeping
# insulation and battery parameters identical to SCENARIO_A.


# ---> OVERRIDE PARAMETERS HERE <---
# Overwrites the battery size for Scenario A only
# SCENARIO_B["vent"]["V_total"] = 20000
SCENARIO_B["stack"]["P_nom"] = 40000

# ================================================================================
# SECTION 3 - CORE SIMULATION ENGINE
# Runs the full thermal + electrochemical model for one scenario and returns
# all time series plus derived summary quantities. Physics identical to v4.1
# (all corrections from the v4.0 -> v4.1 review are retained).
# ================================================================================

def get_price_and_tariff(day_h, price):
    levels = {1: price["valley"], 2: price["flat"],
              3: price["peak"], 4: price["critical"]}
    for row in price["windows"]:
        h0, h1, lid = row[0], row[1], int(row[2])
        if h0 <= day_h < h1:
            p = levels.get(lid, price["default"])
            ts = "peak" if lid == 3 else "off_peak"
            return p, ts
    return price["default"], "off_peak"


def run_simulation(scenario):
    env = scenario["env"]
    solar = dict(scenario["solar"])
    solar["day_len_h"] = solar["sunset"] - solar["sunrise"]
    phys = dict(scenario["phys"])
    phys["A_surface"] = A_TOTAL_ENVELOPE
    phys["U_wall"] = get_wall_U(scenario["insulation"])
    phys["C_th"] = phys["V_total"] * phys["rho"] * phys["cp"]
    container = dict(scenario["container"])
    container["U_wall_inner"] = get_wall_U(scenario["insulation"])
    stack = scenario["stack"]
    pump = dict(scenario["pump"])
    pump["P_rated"] = stack["P_nom"] * pump["ratio_rated"]
    pump["P_idle"] = stack["P_nom"] * pump["ratio_idle"]
    hex_ = scenario["hex"]
    tech_id = scenario["tech_id"]
    ekes_power_model = scenario["ekes_power_model"]
    ekes_Q_nom = hex_["ekes_Q_nom_ref"] * scenario["ekes_scale"]
    ekes_P_elec_nom = hex_["ekes_P_elec_ref"] * scenario["ekes_scale"]
    ekes_COP_nom = hex_["ekes_COP_ref"]
    cool = scenario["cool"]
    heat = scenario["heat"]
    agitator = scenario["agitator"]
    stack_thermal = scenario["stack_thermal"]
    L_dim, W_dim, H_dim = stack_thermal["dim_m"]
    A_stack = 2.0 * (L_dim * W_dim + L_dim * H_dim + W_dim * H_dim)
    UA_stack = stack_thermal["U_eff_stack"] * A_stack
    C_th_metal = stack_thermal["m_metal_kg"] * stack_thermal["cp_metal"]
    C_th_plastic = stack_thermal["m_plastic_kg"] * stack_thermal["cp_plastic"]
    C_th_ely_stk = stack_thermal["V_ely_L"] * phys["rho"] * phys["cp"]
    C_th_stack = C_th_metal + C_th_plastic + C_th_ely_stk
    flush_cfg = scenario["flush_cfg"]
    price = scenario["price"]
    sim = dict(scenario["sim"])
    sim["steps"] = int(sim["days"] * 24 * 3600 / sim["dt"])
    vent = dict(scenario["vent"])
    vent["active"] = scenario["vent_active"]
    vent["P_fan"] = vent["P_ref"] * (vent["Flow_m3h"] / vent["Flow_ref"]) ** 3
    vent["residual_factor"] = 220 / (220 + vent["Flow_m3h"])

    t = np.arange(0, sim["days"] * 24 * 3600, sim["dt"], dtype=float)
    steps = len(t)

    T_ely = np.zeros(steps); T_stack = np.zeros(steps)
    T_innen_arr = np.zeros(steps); T_air = np.zeros(steps); T_cont = np.zeros(steps)
    SoC = np.zeros(steps)
    Q_gen = np.zeros(steps); Q_cool = np.zeros(steps); Q_heat_vec = np.zeros(steps)
    Q_hex_cap = np.zeros(steps); Q_shunt_vec = np.zeros(steps); Q_wall_vec = np.zeros(steps)
    P_cool_elec_vec = np.zeros(steps); P_pump_elec_vec = np.zeros(steps)
    P_flush_elec_vec = np.zeros(steps); P_heat_elec_vec = np.zeros(steps)
    P_agit_elec_vec = np.zeros(steps); P_fan_elec_vec = np.zeros(steps)
    Cool_Run = np.zeros(steps, dtype=int); Cool_Mode = np.zeros(steps, dtype=int)
    Heat_Run = np.zeros(steps, dtype=int); Agit_Run = np.zeros(steps, dtype=int)
    Flush_Run = np.zeros(steps, dtype=int); State_Mode = np.zeros(steps, dtype=int)
    Fan_State = np.zeros(steps, dtype=int)

    cool_state_mem = 0; cool_timer_s = 1e9; heat_is_running = 0
    Current_State = 0; agit_timer_s = 0.0
    flush_is_running = 0; flush_timer_s = 0.0; vent_state_mem = 0

    E_cool_elec_kWh = 0.0; E_pump_elec_kWh = 0.0; E_flush_elec_kWh = 0.0
    E_heat_elec_kWh = 0.0; E_agit_elec_kWh = 0.0; E_fan_elec_kWh = 0.0
    E_pump_cool_kWh = 0.0; E_pump_not_cool_kWh = 0.0; Cost_cool_CNY = 0.0; Cost_fan_CNY = 0.0
    E_charge_grid_kWh = 0.0; E_discharge_grid_kWh = 0.0

    T_ely[0] = 25.0; T_stack[0] = 25.0; SoC[0] = stack["SoC_start"]

    UA_tank_inner = container["U_tank_inner"] * container["A_tank_surface"]
    UA_wall_inner = container["U_wall_inner"] * container["A_wall_inner"]
    UA_total = UA_tank_inner + UA_wall_inner

    for i in range(steps - 1):
        day_h = (t[i] / 3600.0) % 24
        T_air[i] = ((env["T_max"] + env["T_min"]) / 2.0
                    - (env["T_max"] - env["T_min"]) / 2.0
                    * np.cos((day_h - 3.0) * 2 * np.pi / 24))
        dT_solar = 0.0
        if solar["sunrise"] <= day_h <= solar["sunset"]:
            I_now = solar["I_max"] * np.sin(np.pi * (day_h - solar["sunrise"]) / solar["day_len_h"])
            dT_solar = solar["alpha"] * I_now / solar["h_ext"]
        T_sol_air = T_air[i] + dT_solar

        p_now, tariff_state = get_price_and_tariff(day_h, price)

        if vent["active"]:
            dT_wa = T_sol_air - T_air[i]
            if vent_state_mem == 1 and dT_wa < vent["dT_trigger_off"]: vent_state_mem = 0
            elif vent_state_mem == 0 and dT_wa > vent["dT_trigger_on"]: vent_state_mem = 1
            T_cont[i] = T_air[i] + vent["residual_factor"] * dT_solar if vent_state_mem else T_sol_air
        else:
            T_cont[i] = T_sol_air
        Fan_State[i] = vent_state_mem
        P_fan_now = vent["P_fan"] if vent_state_mem else 0.0
        P_fan_elec_vec[i] = P_fan_now
        E_fan_elec_kWh += P_fan_now * sim["dt"] / 3.6e6
        Cost_fan_CNY += P_fan_now * sim["dt"] / 3.6e6 * p_now

        T_innen = (UA_tank_inner * T_ely[i] + UA_wall_inner * T_cont[i]) / UA_total
        T_innen_arr[i] = T_innen

        T_heat_off_eff = heat["T_off"]; T_cool_B_off_eff = cool["T_B_off"]

        req_cool_mode = 0
        if hex_["active"]:
            if cool_state_mem == 0:
                if tariff_state == "peak" and T_ely[i] >= cool["T_A_on"]: req_cool_mode = 1
                elif tariff_state == "off_peak" and T_ely[i] >= cool["T_B_on"]: req_cool_mode = 2
            elif cool_state_mem == 1:
                # Re-check tariff state on transition out of State A so the
                # controller cannot stay locked into peak-mode logic once the
                # tariff window has actually changed (v4.1 fix, retained).
                if tariff_state != "peak":
                    req_cool_mode = 2 if T_ely[i] >= cool["T_B_on"] else 0
                else:
                    req_cool_mode = 1 if T_ely[i] > cool["T_A_off"] else 0
            elif cool_state_mem == 2:
                req_cool_mode = 2 if T_ely[i] > T_cool_B_off_eff else 0

        cool_timer_s += sim["dt"]
        prev_cool_running = Cool_Run[i - 1] if i > 0 else 0
        desired_cool_running = 1 if req_cool_mode > 0 else 0
        if desired_cool_running != prev_cool_running:
            if desired_cool_running == 0 and cool_timer_s < cool["min_on_s"]:
                desired_cool_running = 1; req_cool_mode = cool_state_mem
            elif desired_cool_running == 1 and cool_timer_s < cool["min_off_s"]:
                desired_cool_running = 0; req_cool_mode = 0
        if desired_cool_running != prev_cool_running:
            cool_timer_s = 0.0
        cool_state_mem = req_cool_mode; cool_is_running = desired_cool_running
        Cool_Run[i] = cool_is_running; Cool_Mode[i] = req_cool_mode

        if heat["active"]:
            if heat_is_running == 0 and T_ely[i] < heat["T_on"]: heat_is_running = 1
            elif heat_is_running == 1 and T_ely[i] >= T_heat_off_eff: heat_is_running = 0
        else:
            heat_is_running = 0
        if cool_is_running: heat_is_running = 0
        Heat_Run[i] = heat_is_running

        agit_active_cond = (Current_State != 0) or heat_is_running or cool_is_running
        if agitator["active"] and agit_active_cond:
            agit_timer_s += sim["dt"]
            if agit_timer_s >= agitator["period_s"]: agit_timer_s = 0.0
            agit_is_running = 1 if agit_timer_s < agitator["pulse_s"] else 0
        else:
            agit_timer_s = 0.0; agit_is_running = 0
        Agit_Run[i] = agit_is_running
        P_agit_now = agitator["P_agit_W"] * agit_is_running
        P_agit_elec_vec[i] = P_agit_now
        E_agit_elec_kWh += P_agit_now * sim["dt"] / 3.6e6

        if flush_is_running:
            flush_timer_s += sim["dt"]
            if flush_timer_s >= flush_cfg["duration_s"]:
                flush_is_running = 0; flush_timer_s = 0.0
        else:
            if (T_stack[i] <= stack_thermal["T_flush_trigger"]
                    and Current_State == 0 and not cool_is_running):
                flush_is_running = 1; flush_timer_s = 0.0
        Flush_Run[i] = flush_is_running

        if stack["active"]:
            ns = Current_State
            if Current_State == 0:
                if (stack["start_charge_h"] <= day_h < 17) and SoC[i] < stack["SoC_max"]: ns = 1
                elif ((day_h >= stack["start_disch_h"]) or (day_h < 6)) and SoC[i] > stack["SoC_min"]: ns = 2
            elif Current_State == 1:
                if SoC[i] >= stack["SoC_max"] or (day_h > 17 and day_h < stack["start_charge_h"]): ns = 0
            elif Current_State == 2:
                if SoC[i] <= stack["SoC_min"] or (day_h > 6 and day_h < stack["start_disch_h"]): ns = 0
            Current_State = ns
        else:
            Current_State = 0
        State_Mode[i] = Current_State

        T_K = T_ely[i] + 273.15
        visc_factor = float(np.clip(np.exp(pump["Ea_visc"] * (1.0 / T_K - 1.0 / 298.15)), 0.5, 3.0))
        if Current_State != 0:
            pump_load = pump["P_rated"]
        elif tech_id == 1:
            # Plate HEX: heat exchange works via convection alone,
            # main pumps are NOT required while idle-cooling.
            pump_load = 0.0
        else:
            # EKES (tech_id == 2): tube-bundle HEX sits in the primary
            # electrolyte loop -> main pumps MUST run at rated flow
            # whenever cooling is active, even in idle (no charge/discharge).
            pump_load = pump["P_rated"] if cool_is_running else 0.0
        q_pumps = pump_load * visc_factor
        P_pump_elec_vec[i] = q_pumps
        E_pump_elec_kWh += q_pumps * sim["dt"] / 3.6e6
        if tech_id == 2 and cool_is_running and Current_State == 0:
            pump_cool_power_W = q_pumps
        else:
            pump_cool_power_W = 0.0

        P_flush_now = (pump["P_idle"] * visc_factor
                       if (flush_is_running and Current_State == 0 and not cool_is_running) else 0.0)
        P_flush_elec_vec[i] = P_flush_now
        E_flush_elec_kWh += P_flush_now * sim["dt"] / 3.6e6

        R_stack = stack["R_ref"] * np.exp(stack["Ea_R"] * (1.0 / T_K - 1.0 / 298.15))
        SoC_safe = float(np.clip(SoC[i], 0.05, 0.95))
        U_OCV = stack["U_OCV_ref"] + 1.5 * math.log(SoC_safe / (1.0 - SoC_safe))
        Q_stack_total = 0.0; SoC_next_raw = SoC[i]
        P_elec_charge = stack["P_nom"]
        P_max_disch = U_OCV**2 / (4.0 * R_stack)
        P_elec_disch = min(stack["P_nom"], P_max_disch * 0.95)

        if stack["active"]:
            if Current_State == 1:  # charging
                I_A = (-U_OCV + math.sqrt(max(0.0, U_OCV**2 + 4 * R_stack * P_elec_charge))) / (2 * R_stack)
                Q_stack_total = I_A**2 * R_stack - I_A * T_K * stack["dS_coeff"]
                E_charge_grid_kWh += P_elec_charge * sim["dt"] / 3.6e6
                SoC_next_raw = SoC[i] + (I_A * stack["U_OCV_ref"] * sim["dt"] / 3600.0) / stack["Cap_nom"]
            elif Current_State == 2:  # discharging
                disc = max(0.0, U_OCV**2 - 4 * R_stack * P_elec_disch)
                I_A = (U_OCV - math.sqrt(disc)) / (2 * R_stack)
                Q_stack_total = I_A**2 * R_stack + I_A * T_K * stack["dS_coeff"]
                E_discharge_grid_kWh += P_elec_disch * sim["dt"] / 3.6e6
                SoC_next_raw = SoC[i] - (I_A * stack["U_OCV_ref"] * sim["dt"] / 3600.0) / stack["Cap_nom"]

        Q_shunt = stack["P_shunt_W"] * SoC[i]; Q_shunt_vec[i] = Q_shunt
        SoC_loss = Q_shunt * sim["dt"] / 3600.0 / stack["Cap_nom"]
        SoC[i + 1] = float(np.clip(SoC_next_raw - SoC_loss, 0.0, 1.0))

        Q_gen[i] = Q_stack_total + q_pumps + P_flush_now + Q_shunt

        Q_real = 0.0; P_elec_cool = 0.0; Q_physics = 0.0
        if cool_is_running:
            dT_amb = T_air[i] - 25.0
            if tech_id == 1:
                # Plate HEX + external chiller (indirect loop)
                curr_COP = max(hex_["plate_COP"] - hex_["COP_deg"] * dT_amb, 1.5)
                dT_hex = T_ely[i] - hex_["plate_T_in"]
                if dT_hex > 0:
                    Q_physics = hex_["plate_UA"] * dT_hex * hex_["LMTD_corr"]
                Q_real = min(Q_physics, hex_["plate_P_max"])
                P_elec_cool = Q_real / curr_COP
            else:
                # EKES tube-bundle direct cooling (primary electrolyte loop)
                cap_f = min(1.0, max(0.0, T_ely[i] - 22.0) / 10.0)
                derating = max(1.0 - 0.005 * dT_amb, 0.0) if dT_amb > 0 else 1.0
                Q_real = float(np.clip(ekes_Q_nom * derating * cap_f, 0.0, ekes_Q_nom))
                curr_COP = max(ekes_COP_nom - 0.05 * dT_amb, 2.0)
                if Q_real > 0:
                    P_elec_cool = (Q_real / curr_COP if ekes_power_model == "COP"
                                   else ekes_P_elec_nom * (0.2 + 0.8 * Q_real / ekes_Q_nom))
                Q_physics = Q_real
        Q_cool[i] = Q_real; Q_hex_cap[i] = Q_physics
        P_cool_elec_vec[i] = P_elec_cool
        E_cool_elec_kWh += P_elec_cool * sim["dt"] / 3.6e6
        E_pump_cool_kWh += pump_cool_power_W * sim["dt"] / 3.6e6
        E_pump_not_cool_kWh += (q_pumps - pump_cool_power_W) * sim["dt"] / 3.6e6
        Cost_cool_CNY += (P_elec_cool + pump_cool_power_W) * sim["dt"] / 3.6e6 * p_now

        Q_heat_now = heat["P_heat_W"] if heat_is_running else 0.0
        P_heat_now = Q_heat_now / heat["COP_heat"] if heat_is_running else 0.0
        Q_heat_vec[i] = Q_heat_now; P_heat_elec_vec[i] = P_heat_now
        E_heat_elec_kWh += P_heat_now * sim["dt"] / 3.6e6

        pumps_running = (Current_State != 0) or (flush_is_running == 1) or cool_is_running
        if pumps_running:
            T_stack[i + 1] = T_ely[i]
        else:
            dT = (UA_stack * (T_innen - T_stack[i]) / C_th_stack) * sim["dt"]
            T_stack[i + 1] = T_stack[i] + dT

        # Wall heat flow, physically coupled: ambient -> inner air (T_innen)
        # -> electrolyte tank, via the same UA network used for T_innen.
        Q_wall = UA_tank_inner * (T_innen - T_ely[i])
        Q_wall_vec[i] = Q_wall
        T_ely[i + 1] = T_ely[i] + ((Q_gen[i] + Q_wall - Q_cool[i] + Q_heat_vec[i]) * sim["dt"] / phys["C_th"])

    for arr in [T_air, T_cont, T_ely, T_stack, T_innen_arr, Fan_State, Q_gen, Q_cool,
                Q_heat_vec, P_cool_elec_vec, P_pump_elec_vec, P_flush_elec_vec,
                P_heat_elec_vec, P_agit_elec_vec, P_fan_elec_vec, Cool_Run, Cool_Mode,
                Heat_Run, Agit_Run, Flush_Run, Q_hex_cap, Q_shunt_vec, Q_wall_vec, State_Mode]:
        arr[-1] = arr[-2]

    E_internal_kWh = np.sum(Q_gen) * sim["dt"] / 3.6e6
    E_shunt_kWh = np.sum(Q_shunt_vec) * sim["dt"] / 3.6e6
    E_cool_th_kWh = np.sum(Q_cool) * sim["dt"] / 3.6e6
    E_heat_th_kWh = np.sum(Q_heat_vec) * sim["dt"] / 3.6e6
    E_wall_kWh = np.sum(Q_wall_vec) * sim["dt"] / 3.6e6
    E_parasitic_total_kWh = (E_cool_elec_kWh + E_pump_elec_kWh + E_flush_elec_kWh
                              + E_heat_elec_kWh + E_agit_elec_kWh + E_fan_elec_kWh)
    RTE_stack = (E_discharge_grid_kWh / E_charge_grid_kWh * 100) if E_charge_grid_kWh > 0 else float("nan")
    RTE_system = (E_discharge_grid_kWh / (E_charge_grid_kWh + E_parasitic_total_kWh) * 100
                  ) if (E_charge_grid_kWh + E_parasitic_total_kWh) > 0 else float("nan")

    series = dict(t=t, T_ely=T_ely, T_stack=T_stack, T_innen=T_innen_arr, T_air=T_air,
                  T_cont=T_cont, SoC=SoC, Q_gen=Q_gen, Q_cool=Q_cool, Q_heat=Q_heat_vec,
                  Q_shunt=Q_shunt_vec, Q_wall=Q_wall_vec, P_cool_elec=P_cool_elec_vec,
                  P_pump_elec=P_pump_elec_vec, P_flush_elec=P_flush_elec_vec,
                  P_heat_elec=P_heat_elec_vec, P_agit_elec=P_agit_elec_vec,
                  P_fan_elec=P_fan_elec_vec, Cool_Run=Cool_Run, Cool_Mode=Cool_Mode,
                  Heat_Run=Heat_Run, Agit_Run=Agit_Run, Flush_Run=Flush_Run, State_Mode=State_Mode)

    tech_name = "Plate-HEX (Indirect)" if tech_id == 1 else "EKES (Direct, primary loop)"
    summary = dict(
        name=scenario["name"], sim_days=sim["days"],
        tech_id=tech_id, tech_name=tech_name,
        ekes_Q_nom=ekes_Q_nom if tech_id == 2 else float("nan"),
        E_pump_cool_kWh=E_pump_cool_kWh, E_pump_not_cool_kWh=E_pump_not_cool_kWh,
        T_ely_min=np.min(T_ely), T_ely_max=np.max(T_ely),
        T_stack_min=np.min(T_stack), T_stack_max=np.max(T_stack),
        E_internal_kWh=E_internal_kWh, E_shunt_kWh=E_shunt_kWh,
        E_wall_kWh=E_wall_kWh, E_cool_th_kWh=E_cool_th_kWh, E_heat_th_kWh=E_heat_th_kWh,
        E_cool_elec_kWh=E_cool_elec_kWh, E_pump_elec_kWh=E_pump_elec_kWh,
        E_flush_elec_kWh=E_flush_elec_kWh, E_heat_elec_kWh=E_heat_elec_kWh,
        E_agit_elec_kWh=E_agit_elec_kWh, E_fan_elec_kWh=E_fan_elec_kWh,
        E_parasitic_total_kWh=E_parasitic_total_kWh,
        Cost_cool_CNY=Cost_cool_CNY, Cost_fan_CNY=Cost_fan_CNY,
        Cost_total_CNY=Cost_cool_CNY + Cost_fan_CNY,
        Cool_h=np.sum(Cool_Run) * sim["dt"] / 3600, Heat_h=np.sum(Heat_Run) * sim["dt"] / 3600,
        Agit_h=np.sum(Agit_Run) * sim["dt"] / 3600, Flush_h=np.sum(Flush_Run) * sim["dt"] / 3600,
        Fan_h=np.sum(Fan_State) * sim["dt"] / 3600,
        RTE_stack=RTE_stack, RTE_system=RTE_system,
        U_wall=phys["U_wall"],
    )
    return series, summary


# ================================================================================
# SECTION 4 - PLOTTING (unchanged v4.1 5-panel dashboard, generated per scenario)
# ================================================================================

def plot_scenario(series, summary, filename_tag):
    t_h = series["t"] / 3600.0
    fig = plt.figure(figsize=(15, 24))
    fig.suptitle(f"VRFB Thermal Management Simulation v5.2 -- {summary['name']} ({summary['tech_name']})",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_h, series["T_ely"], color="#2980b9", lw=2.5, label="Electrolyte (T_ely)")
    ax1.plot(t_h, series["T_stack"], color="#c0392b", lw=1.5, label="Stack (T_stack)")
    ax1.plot(t_h, series["T_innen"], color="#8e44ad", lw=1, ls="--", label="Inner Air (T_innen)")
    ax1.plot(t_h, series["T_air"], color="#7f8c8d", lw=1, ls=":", label="Ambient (T_air)")
    ax1.plot(t_h, series["T_cont"], color="#34495e", lw=1, ls="--", alpha=0.5, label="Boundary (T_cont)")
    ax1.fill_between(t_h, series["T_air"].min(), series["T_ely"].max() + 5,
                      where=(series["Heat_Run"] == 1), alpha=0.15, color="#e67e22")
    ax1.fill_between(t_h, series["T_air"].min(), series["T_ely"].max() + 5,
                      where=(series["Cool_Run"] == 1), alpha=0.1, color="#2980b9")
    ax1.axhline(38.0, color="#c0392b", ls=":", lw=1.5, label="WARN: Overheat (38 C)")
    ax1.axhline(8.0, color="#2980b9", ls=":", lw=1.5, label="WARN: Undercool (8 C)")
    ax1.set_ylabel("Temperature [C]"); ax1.set_title("Temperatures & Thermal State")
    ax1.legend(loc="upper right", ncol=3, fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_h, series["Q_gen"] / 1000, color="#c0392b", lw=1.5, label="Total Internal Heat")
    ax2.plot(t_h, series["Q_shunt"] / 1000, color="#8e44ad", lw=1, ls="--", label="Shunt / Crossover")
    ax2.plot(t_h, series["Q_wall"] / 1000, color="#16a085", lw=1, ls="-.", label="Wall Conduction")
    ax2.plot(t_h, series["Q_cool"] / 1000, color="#2980b9", lw=2, label="Active Cooling")
    ax2.plot(t_h, series["Q_heat"] / 1000, color="#e67e22", lw=2, label="Active Heating")
    ax2.set_ylabel("Power [kW]"); ax2.set_title("Thermal Power Balance (Electrolyte)")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t_h, series["P_cool_elec"] / 1000, color="#2980b9", lw=2, label="Chiller")
    ax3.plot(t_h, series["P_pump_elec"] / 1000, color="#8e44ad", lw=1.5, label="Main Pumps")
    ax3.plot(t_h, series["P_heat_elec"] / 1000, color="#e67e22", lw=2, label="Heater")
    ax3.plot(t_h, series["P_agit_elec"] / 1000, color="#27ae60", lw=1.5, label="Agitator (PWM)")
    ax3.plot(t_h, series["P_fan_elec"] / 1000, color="#f39c12", lw=1.5, label="Ventilation Fan")
    ax3.plot(t_h, series["P_flush_elec"] / 1000, color="#d35400", lw=1.5, ls="--", label="Flush Pump")
    ax3.set_ylabel("Power [kW]"); ax3.set_title("Parasitic Electrical Power")
    ax3.legend(loc="upper right", fontsize=8, ncol=2); ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(t_h, 0, 100, where=(series["State_Mode"] == 1), alpha=0.15, color="#f1c40f", label="Charging")
    ax4.fill_between(t_h, 0, 100, where=(series["State_Mode"] == 2), alpha=0.15, color="#3498db", label="Discharging")
    ax4.plot(t_h, series["SoC"] * 100, color="#27ae60", lw=2, label="SoC [%]")
    ax4.set_ylabel("SoC [%]"); ax4.set_ylim(0, 100); ax4.set_title("State of Charge")
    ax4.legend(loc="upper right", fontsize=8); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[4])
    ax5.fill_between(t_h, 3, 4, where=(series["Cool_Mode"] == 1), step="post", alpha=0.7, color="#2980b9", label="Cooling State A (Peak)")
    ax5.fill_between(t_h, 3, 4, where=(series["Cool_Mode"] == 2), step="post", alpha=0.7, color="#5dade2", label="Cooling State B (Off-Peak)")
    ax5.fill_between(t_h, 2, 3, where=(series["Heat_Run"] == 1), step="post", alpha=0.7, color="#e67e22", label="Heating")
    ax5.fill_between(t_h, 1, 2, where=(series["Agit_Run"] == 1), step="post", alpha=0.7, color="#27ae60", label="Agitator (PWM)")
    ax5.fill_between(t_h, 0, 1, where=(series["Flush_Run"] == 1), step="post", alpha=0.7, color="#c0392b", label="Stack Flush")
    ax5.set_yticks([0.5, 1.5, 2.5, 3.5]); ax5.set_yticklabels(["Flush", "Agitator", "Heating", "Cooling"])
    ax5.set_xlabel("Time [h]"); ax5.set_title("Discrete Operating States")
    ax5.legend(loc="upper right", fontsize=8); ax5.grid(True, alpha=0.3)

    plot_path = os.path.join(_BASE, f"simulation_v52_{filename_tag}_plot.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    return plot_path


# ================================================================================
# SECTION 5 - CONSOLE SUMMARY (unchanged v4.1 report format, per scenario)
# ================================================================================

def print_summary(summary):
    print("\n" + "=" * 60)
    print(f" THERMAL & ENERGY BALANCE v5.2 -- {summary['name']} -- {summary['sim_days']} days")
    print("=" * 60)
    print(f"Cooling technology: {summary['tech_name']}")
    if summary['tech_id'] == 2:
        print(f"EKES nominal cooling capacity: {summary['ekes_Q_nom']:.0f} W")
    print(f"T_ely: min={summary['T_ely_min']:.1f} C | max={summary['T_ely_max']:.1f} C")
    print(f"T_stack: min={summary['T_stack_min']:.1f} C | max={summary['T_stack_max']:.1f} C")
    print(f"Internal Heat (excl. agitator): {summary['E_internal_kWh']:.1f} kWh")
    print(f"Wall conduction (net, ambient->electrolyte): {summary['E_wall_kWh']:.1f} kWh_th")
    print("-" * 40)
    print("Parasitic Power:")
    print(f"  Chiller:      {summary['E_cool_elec_kWh']:.1f} kWh")
    print(f"  Main Pumps:   {summary['E_pump_elec_kWh']:.1f} kWh")
    print(f"  Flush Pumps:  {summary['E_flush_elec_kWh']:.1f} kWh")
    print(f"  Heater:       {summary['E_heat_elec_kWh']:.1f} kWh")
    print(f"  Agitator:     {summary['E_agit_elec_kWh']:.1f} kWh")
    print(f"  Vent. Fan:    {summary['E_fan_elec_kWh']:.1f} kWh")
    print(f"  TOTAL AUX:    {summary['E_parasitic_total_kWh']:.1f} kWh")
    print("-" * 40)
    print("Pump Energy Allocation (cooling-technology dependent):")
    print(f"  Pump energy allocated to cooling:     {summary['E_pump_cool_kWh']:.1f} kWh")
    print(f"  Pump energy NOT allocated to cooling: {summary['E_pump_not_cool_kWh']:.1f} kWh")
    print("-" * 40)
    print(f"TOU Cooling Costs: {summary['Cost_cool_CNY']:.2f} CNY")
    print(f"TOU Fan Costs:     {summary['Cost_fan_CNY']:.2f} CNY")
    print(f"TOU Total Costs:   {summary['Cost_total_CNY']:.2f} CNY")
    print(f"Cooling runtime: {summary['Cool_h']:.1f} h")
    print(f"Heating runtime: {summary['Heat_h']:.1f} h")
    print(f"Agitator runtime: {summary['Agit_h']:.1f} h")
    print(f"Flush runtime: {summary['Flush_h']:.1f} h")
    print(f"Fan runtime: {summary['Fan_h']:.1f} h")
    print(f"Round-Trip Efficiency (stack, AC-AC): {summary['RTE_stack']:.2f} %")
    print(f"Round-Trip Efficiency (system, incl. aux): {summary['RTE_system']:.2f} %")
    print("=" * 60)


def export_csv(summary, filename_tag):
    KV = [
        ("Scenario name", "name", summary["name"]),
        ("Model version", "version", "v5.2"),
        ("Simulation days", "sim.days", summary["sim_days"]),
        ("Cooling technology", "tech_name", summary["tech_name"]),
        ("EKES nominal cooling capacity", "ekes_Q_nom", summary["ekes_Q_nom"]),
        ("Pump energy allocated to cooling", "E_pump_cool", summary["E_pump_cool_kWh"]),
        ("Pump energy NOT allocated to cooling", "E_pump_not_cool", summary["E_pump_not_cool_kWh"]),
        ("Container wall U-value", "U_wall", summary["U_wall"]),
        ("T_ely max", "T_ely_max", summary["T_ely_max"]),
        ("T_ely min", "T_ely_min", summary["T_ely_min"]),
        ("T_stack min", "T_stack_min", summary["T_stack_min"]),
        ("Internal heat", "E_internal", summary["E_internal_kWh"]),
        ("Wall conduction net", "E_wall", summary["E_wall_kWh"]),
        ("Chiller cooling (thermal)", "E_cool_th", summary["E_cool_th_kWh"]),
        ("Heating added (thermal)", "E_heat_th", summary["E_heat_th_kWh"]),
        ("Chiller elec", "E_chiller_elec", summary["E_cool_elec_kWh"]),
        ("Main pump elec", "E_pump_elec", summary["E_pump_elec_kWh"]),
        ("Flush pump elec", "E_flush_elec", summary["E_flush_elec_kWh"]),
        ("Agitator elec", "E_agit_elec", summary["E_agit_elec_kWh"]),
        ("Heater elec", "E_heat_elec", summary["E_heat_elec_kWh"]),
        ("Fan elec", "E_fan_elec", summary["E_fan_elec_kWh"]),
        ("Total parasitic", "E_parasitic_total", summary["E_parasitic_total_kWh"]),
        ("Cooling costs CNY", "Cost_cool_CNY", summary["Cost_cool_CNY"]),
        ("Fan costs CNY", "Cost_fan_CNY", summary["Cost_fan_CNY"]),
        ("Total costs CNY", "Cost_total_CNY", summary["Cost_total_CNY"]),
        ("RTE stack [%]", "RTE_stack", summary["RTE_stack"]),
        ("RTE system [%]", "RTE_system", summary["RTE_system"]),
    ]
    csv_path = os.path.join(_BASE, f"simulation_v52_{filename_tag}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Description", "Parameter", "Value"])
        for row in KV:
            val = row[2]
            if isinstance(val, (float, np.floating)):
                val = f"{val:.3f}".replace(".", ",")
            w.writerow([row[0], row[1], val])
    return csv_path


# ================================================================================
# SECTION 6 - RUN SCENARIO A (original v4.1-style report, unchanged content)
# ================================================================================

series_A, summary_A = run_simulation(SCENARIO_A)
print_summary(summary_A)
plot_path_A = plot_scenario(series_A, summary_A, "scenario_A")
csv_path_A = export_csv(summary_A, "scenario_A")
print(f"CSV exported to {csv_path_A}")
print(f"Plot exported to {plot_path_A}")


# ================================================================================
# SECTION 7 - RUN SCENARIO B (identical report structure, second scenario)
# ================================================================================

series_B, summary_B = run_simulation(SCENARIO_B)
print_summary(summary_B)
plot_path_B = plot_scenario(series_B, summary_B, "scenario_B")
csv_path_B = export_csv(summary_B, "scenario_B")
print(f"CSV exported to {csv_path_B}")
print(f"Plot exported to {plot_path_B}")


# ================================================================================
# SECTION 8 - SCENARIO COMPARISON (introduced in v5.0, retained in v5.1)
# Side-by-side comparison of both scenarios: energy balance, cooling/heating
# demand, cost, and round-trip efficiency, plus absolute and relative deltas.
# ================================================================================

def compare_scenarios(sA, sB):
    keys = [
        ("T_ely_max", "Max electrolyte temperature [C]"),
        ("E_wall_kWh", "Wall heat ingress (net) [kWh_th]"),
        ("E_cool_th_kWh", "Cooling delivered (thermal) [kWh_th]"),
        ("E_cool_elec_kWh", "Chiller/EKES electrical energy [kWh_el]"),
        ("E_pump_elec_kWh", "Main pump electrical energy [kWh_el]"),
        ("E_pump_cool_kWh", "Pump energy allocated to cooling [kWh_el]"),
        ("E_pump_not_cool_kWh", "Pump energy NOT allocated to cooling [kWh_el]"),
        ("E_fan_elec_kWh", "Ventilation fan energy [kWh_el]"),
        ("E_heat_elec_kWh", "Heater electrical energy [kWh_el]"),
        ("E_parasitic_total_kWh", "Total parasitic energy [kWh_el]"),
        ("Cost_total_CNY", "Total energy cost [CNY]"),
        ("RTE_stack", "Round-trip efficiency, stack [%]"),
        ("RTE_system", "Round-trip efficiency, system [%]"),
    ]
    rows = []
    for key, label in keys:
        vA = sA[key]; vB = sB[key]
        delta_abs = vB - vA
        delta_pct = (delta_abs / vA * 100) if vA not in (0, None) and not np.isnan(vA) and vA != 0 else float("nan")
        rows.append((label, vA, vB, delta_abs, delta_pct))
    return rows


comparison_rows = compare_scenarios(summary_A, summary_B)

print("\n" + "=" * 90)
print(f" SCENARIO COMPARISON: A = '{summary_A['name']}'  vs.  B = '{summary_B['name']}'")
print("=" * 90)
header = f"{'Metric':<42}{'Scenario A':>14}{'Scenario B':>14}{'Delta (B-A)':>14}{'Delta [%]':>10}"
print(header)
print("-" * 90)
for label, vA, vB, dabs, dpct in comparison_rows:
    dpct_str = f"{dpct:+.1f}%" if not np.isnan(dpct) else "n/a"
    print(f"{label:<42}{vA:>14.2f}{vB:>14.2f}{dabs:>+14.2f}{dpct_str:>10}")
print("=" * 90)

comparison_csv_path = os.path.join(_BASE, "simulation_v52_comparison.csv")
with open(comparison_csv_path, "w", newline="") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["Metric", "Scenario A", "Scenario B", "Delta (B-A)", "Delta [%]"])
    for label, vA, vB, dabs, dpct in comparison_rows:
        w.writerow([label,
                    f"{vA:.3f}".replace(".", ","),
                    f"{vB:.3f}".replace(".", ","),
                    f"{dabs:.3f}".replace(".", ","),
                    f"{dpct:.2f}".replace(".", ",") if not np.isnan(dpct) else "n/a"])
print(f"Comparison CSV exported to {comparison_csv_path}")
