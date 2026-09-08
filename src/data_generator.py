"""
Industrial Maintenance Narrative Dataset Generator
Synthesizes a realistic industrial maintenance log benchmark with temporal visit sequences,
authentic technician narratives, telemetry parameters, domain acronyms, and multi-task failure labels.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EQUIPMENT_TYPES = {
    "PUMP": {
        "name": "Centrifugal Pump",
        "components": ["Drive End Bearing", "Non-Drive End Bearing", "Mechanical Seal", "Impeller", "Shaft Coupling", "Discharge Flange"],
        "nominal_temp": (40, 65),
        "nominal_vib": (0.8, 2.2),
        "nominal_press": (45, 90),
        "nominal_rpm": (1450, 2950),
        "failure_modes": ["Bearing_Degradation", "Hydraulic_Leak", "Normal_Operation"]
    },
    "TURB": {
        "name": "Gas Turbine",
        "components": ["Turbine Blades", "Burner Nozzle", "Thrust Bearing", "Exhaust Diffuser", "Rotor Shaft"],
        "nominal_temp": (70, 95),
        "nominal_vib": (1.0, 2.8),
        "nominal_press": (120, 250),
        "nominal_rpm": (3000, 6000),
        "failure_modes": ["Motor_Overheating", "Bearing_Degradation", "Normal_Operation"]
    },
    "COMP": {
        "name": "Air Compressor",
        "components": ["Suction Valve", "Discharge Valve", "Piston Rings", "Intercooler", "Main Journal Bearing"],
        "nominal_temp": (45, 75),
        "nominal_vib": (1.2, 3.0),
        "nominal_press": (90, 160),
        "nominal_rpm": (1200, 1800),
        "failure_modes": ["Hydraulic_Leak", "Motor_Overheating", "Normal_Operation"]
    },
    "GEAR": {
        "name": "Industrial Gearbox",
        "components": ["Helical Pinion", "Bull Gear", "Input Shaft Bearing", "Output Shaft Bearing", "Oil Sump Filter"],
        "nominal_temp": (50, 70),
        "nominal_vib": (0.9, 2.5),
        "nominal_press": (20, 40),
        "nominal_rpm": (750, 1500),
        "failure_modes": ["Gearbox_Tooth_Wear", "Bearing_Degradation", "Normal_Operation"]
    },
    "MTR": {
        "name": "Induction Motor",
        "components": ["Stator Windings", "Rotor Bars", "DE Bearing", "NDE Bearing", "Cooling Fan Cowl", "Terminal Box"],
        "nominal_temp": (45, 70),
        "nominal_vib": (0.7, 2.0),
        "nominal_press": (0, 0),
        "nominal_rpm": (1480, 2980),
        "failure_modes": ["Motor_Overheating", "Electrical_Fault", "Bearing_Degradation", "Normal_Operation"]
    },
    "CONV": {
        "name": "Heavy Conveyor Drive",
        "components": ["Drive Pulley", "Idler Roller", "Tail Drum", "Belt Tensioner", "Drive Sprocket"],
        "nominal_temp": (35, 60),
        "nominal_vib": (1.0, 2.6),
        "nominal_press": (0, 0),
        "nominal_rpm": (300, 900),
        "failure_modes": ["Gearbox_Tooth_Wear", "Bearing_Degradation", "Normal_Operation"]
    }
}

FAILURE_MODES = [
    "Normal_Operation",
    "Bearing_Degradation",
    "Motor_Overheating",
    "Hydraulic_Leak",
    "Gearbox_Tooth_Wear",
    "Electrical_Fault"
]

RISK_LEVELS = ["Normal", "Low_Risk", "Medium_Risk", "Critical"]

# Narrative templates and domain vocabulary
NARRATIVE_CORPUS = {
    "Normal_Operation": [
        "Routine PM inspection completed. Equipment running smoothly within nominal tolerances. VIB overall {vib:.2f} mm/s, casing temp {temp:.1f}C. Oil levels checked and topped off. No abnormal acoustics or thermal anomalies observed.",
        "Monthly condition monitoring inspection on {component}. Acoustic levels normal, vibration spectra show clean baseline at {vib:.2f} mm/s. Fasteners torqued, zero leakage detected. Unit signed off for full production duty.",
        "Periodic lubrication and visual check completed on {equipment_id}. DE and NDE bearing temperatures steady at {temp:.1f}C. Operational RPM at {rpm}. No evidence of wear, contamination, or mechanical looseness.",
        "Semi-annual overhaul inspection. Inspected {component}, clearances verified against OEM specs. Operating pressure {press} PSI. Dynamic balancing intact, zero cavitation or abnormal resonance."
    ],
    "Bearing_Degradation": {
        "early": [
            "Routine vibration monitoring detected slight elevation on {component}. Overall VIB increased to {vib:.2f} mm/s with faint 1X/2X harmonics. Casing temp slightly elevated at {temp:.1f}C. Recommend monitoring next shift.",
            "Technician noted subtle metallic whining from {component} during high RPM cycle. High-frequency acoustic peak present. Lubrication grease sample shows mild particulate darkening. Placed on watchlist.",
            "Minor acoustic rattle observed at DE bearing housing. Vibration at {vib:.2f} mm/s, slightly above baseline. Thermal imaging shows 8C delta across bearing cap. Flushed and re-greased.",
            "Ultrasonic inspection revealed early micro-impacts in outer raceway of {component}. Vibration level {vib:.2f} mm/s. Running temperature {temp:.1f}C. Early indicator of sub-surface spalling."
        ],
        "advanced": [
            "Severe high-pitch whining and grinding noise emanating from {component}. High vibration velocity of {vib:.2f} mm/s exceeding ISO class alert limit. Housing temperature reached {temp:.1f}C. Flaking and spalling confirmed.",
            "Urgent maintenance alert: {component} exhibiting heavy cage rattling and loose raceway play. Vibration spike at {vib:.2f} mm/s. Metal debris and bronze flakes found in lube filter. Bearing approaching catastrophic seizure.",
            "Emergency inspection triggered by high vibration trip. {component} raceway severely pitted and brinelled. Temp spiked to {temp:.1f}C. Shaft runout detected. Immediate replacement required before full line restart."
        ]
    },
    "Motor_Overheating": {
        "early": [
            "Stator RTD sensors indicate slight thermal drift up to {temp:.1f}C on {component}. Cooling airflow partially restricted by dust accumulation on fan cowl. Cleaned cowl, monitoring winding temp.",
            "Thermal camera inspection identified warm hotspot on motor frame near {component}. Temp recorded at {temp:.1f}C, approx 12C above ambient. Current draw slightly asymmetrical. Scheduled for thermal re-check.",
            "Technician noticed mild burning varnish smell near terminal enclosure after continuous 8-hour run. Stator casing temp {temp:.1f}C. Insulation resistance test recommended at next downtime window."
        ],
        "advanced": [
            "Critical high-temperature alarm tripped on {component}. Stator core temperature reached {temp:.1f}C, far exceeding Class F insulation limit. Heavy burning odor and smoking reported from cooling cowl.",
            "Motor casing thermal runaway observed: surface temp {temp:.1f}C. Stator windings severely discolored and brittle from excessive thermal cycling. Rotor bar overheating confirmed with dynamic torque ripple.",
            "Emergency shutdown: Motor overheated to {temp:.1f}C causing thermal overload trip. Cooling fan blades cracked and clogged. Immediate motor rewinding or replacement necessary."
        ]
    },
    "Hydraulic_Leak": {
        "early": [
            "Minor hydraulic fluid seepage detected around {component} seal lip. System pressure slightly fluctuating at {press} PSI. Wiper seal shows minor glazing. Topped up hydraulic reservoir by 0.5L.",
            "Visual inspection noted oil weeping and wet residue near {component} fitting. Pressure drop of 4 PSI noted across the manifold. Tightened hose crimp fitting and cleaned surrounding pan.",
            "Technician flagged faint hydraulic misting and pressure instability during load cycling. {component} showing micro-cracks on O-ring seal. Added to planned maintenance replacement list."
        ],
        "advanced": [
            "Major hydraulic line rupture and heavy fluid spraying from {component}. System pressure collapsed to {press} PSI. Emergency containment berm deployed to capture oil spill.",
            "Critical pressure failure: {component} seal blown out completely under load. High-pressure hydraulic jetting caused sudden system depressurization and safety trip. Overhaul of hydraulic manifold mandatory.",
            "Severe fluid loss and cavitation noise in hydraulic pump due to dry suction line from leaking {component}. Pressure dropped drastically to {press} PSI. Immediate seal kit overhaul initiated."
        ]
    },
    "Gearbox_Tooth_Wear": {
        "early": [
            "Vibration spectra revealed emerging gear mesh harmonics (GMF) on {component}. Minor surface scuffing noted during borescope inspection. Gearbox lube oil iron count is 45 ppm. Monitoring gear backlash.",
            "Technician reported subtle rhythmic knocking sound under heavy torque on {component}. VIB at {vib:.2f} mm/s. Backlash slightly wider than OEM specification. Oil condition scheduled for ferrography.",
            "Borescope inspection of {component} shows initial micro-pitting along pitch line of helical teeth. Casing temperature steady at {temp:.1f}C. Added EP gear lubricant additive."
        ],
        "advanced": [
            "Severe cyclical clunking and loud grinding from gearbox housing. High vibration {vib:.2f} mm/s with dominant gear mesh sidebands. Large chipped gear tooth fragments recovered from magnetic drain plug.",
            "Critical gear failure: {component} suffered catastrophic tooth root fracture and severe macropitting. Gear backlash out of spec (>1.8mm). Gearbox seized under 80% rated load.",
            "Heavy metal shedding and broken tooth debris found in lube sump. Severe scoring on gear face of {component}. Vibration spike of {vib:.2f} mm/s. Complete gearbox overhaul and gear set replacement ordered."
        ]
    },
    "Electrical_Fault": {
        "early": [
            "Megger insulation resistance test on {component} showed drop from 100 MOhm to 18 MOhm. Phase impedance unbalance at 3.2%. Terminal lugs show slight discoloration from micro-arcing.",
            "Technician noted intermittent current spike and humming buzz from {component}. Harmonic distortion THD measured at 6.8%. Inspecting cable gland and grounding strap for corrosion.",
            "Partial discharge detected during high-voltage test on {component}. Infrared scan shows elevated 14C delta on Phase B lug. Tightened terminal connections and scheduled dielectric testing."
        ],
        "advanced": [
            "Phase-to-ground flashover and arc fault tripped upstream breaker on {component}. Stator insulation completely carbonized and punctured. Heavy acrid smoke in electrical enclosure.",
            "Catastrophic short circuit in {component}. Inter-turn winding insulation burned through, creating heavy localized melting. Earth fault relay tripped instantly on startup.",
            "Critical electrical breakdown: {component} phase balance destroyed by broken rotor end-ring and shorted coils. Sparks observed from terminal box. Total motor isolation and rewiring required."
        ]
    }
}

RECOMMENDATIONS = {
    "Normal_Operation": "Continue standard routine PM schedule; record baseline vibration and temperature at next shift.",
    "Bearing_Degradation": "Perform vibration spectral analysis, inspect bearing lubrication, and schedule replacement of bearing assembly.",
    "Motor_Overheating": "Inspect cooling fan and air ducts, verify load balance, check winding RTD sensors, and test insulation resistance.",
    "Hydraulic_Leak": "Depressurize hydraulic circuit, replace damaged seals/O-rings, tighten fittings, and replenish hydraulic oil.",
    "Gearbox_Tooth_Wear": "Conduct ferrography oil analysis, measure gear backlash and tooth pitting, and schedule gear set replacement.",
    "Electrical_Fault": "Lockout/Tagout equipment, perform Megger insulation and Hi-Pot tests, replace damaged leads, and re-torque terminals."
}


def generate_equipment_trajectory(equipment_id, eq_type, target_failure, num_visits=5, start_date=None):
    """
    Generates a realistic temporal degradation trajectory for a specific equipment instance.
    Visits evolve from Normal -> Early Warning / Anomaly -> Developing Fault -> Critical / Failure.
    """
    if start_date is None:
        start_date = datetime(2025, 1, 10) + timedelta(days=random.randint(0, 180))
    
    eq_meta = EQUIPMENT_TYPES[eq_type]
    records = []
    
    # Choose component associated with this trajectory
    component = random.choice(eq_meta["components"])
    
    # Days between maintenance inspections (approx 14-30 days)
    current_date = start_date
    
    # Failure trajectory progression steps:
    # If target_failure is Normal_Operation, all visits remain Normal.
    # Otherwise, progression goes: Normal -> Early -> Developing -> Critical.
    
    if target_failure == "Normal_Operation":
        stages = ["normal"] * num_visits
        risk_levels = ["Normal"] * num_visits
        early_warning_flags = [0] * num_visits
        ttf_values = [random.randint(180, 365) for _ in range(num_visits)]
    else:
        # Determine transition points based on num_visits
        if num_visits == 3:
            stages = ["normal", "early", "critical"]
            risk_levels = ["Normal", "Low_Risk", "Critical"]
            early_warning_flags = [0, 1, 1]
            ttf_values = [90, 30, 0]
        elif num_visits == 4:
            stages = ["normal", "early", "medium", "critical"]
            risk_levels = ["Normal", "Low_Risk", "Medium_Risk", "Critical"]
            early_warning_flags = [0, 1, 1, 1]
            ttf_values = [120, 45, 14, 0]
        else: # 5 or more visits
            stages = ["normal", "normal", "early", "medium", "critical"]
            if num_visits > 5:
                stages = ["normal"] * (num_visits - 4) + ["early", "medium", "critical"]
            risk_levels = []
            early_warning_flags = []
            ttf_values = []
            for idx, stage in enumerate(stages):
                if stage == "normal":
                    risk_levels.append("Normal")
                    early_warning_flags.append(0)
                    ttf_values.append(120 - idx * 20)
                elif stage == "early":
                    risk_levels.append("Low_Risk")
                    early_warning_flags.append(1)
                    ttf_values.append(40)
                elif stage == "medium":
                    risk_levels.append("Medium_Risk")
                    early_warning_flags.append(1)
                    ttf_values.append(15)
                elif stage == "critical":
                    risk_levels.append("Critical")
                    early_warning_flags.append(1)
                    ttf_values.append(0)

    for visit_idx, stage in enumerate(stages):
        current_date += timedelta(days=random.randint(12, 28))
        risk = risk_levels[visit_idx]
        is_early_warn = early_warning_flags[visit_idx]
        ttf = ttf_values[visit_idx]
        
        # Telemetry generation based on stage
        base_temp_min, base_temp_max = eq_meta["nominal_temp"]
        base_vib_min, base_vib_max = eq_meta["nominal_vib"]
        base_press_min, base_press_max = eq_meta["nominal_press"]
        base_rpm_min, base_rpm_max = eq_meta["nominal_rpm"]
        
        rpm = random.randint(base_rpm_min, base_rpm_max)
        
        if stage == "normal":
            temp = round(random.uniform(base_temp_min, base_temp_max), 1)
            vib = round(random.uniform(base_vib_min, base_vib_max), 2)
            press = random.randint(base_press_min, base_press_max) if base_press_max > 0 else 0
            current_failure_mode = "Normal_Operation"
            template = random.choice(NARRATIVE_CORPUS["Normal_Operation"])
            narrative = template.format(
                vib=vib, temp=temp, press=press, rpm=rpm,
                component=component, equipment_id=equipment_id
            )
        elif stage == "early":
            temp = round(random.uniform(base_temp_max + 2, base_temp_max + 12), 1)
            vib = round(random.uniform(base_vib_max * 1.1, base_vib_max * 1.8), 2)
            press = round(base_press_min * 0.85) if base_press_max > 0 else 0
            current_failure_mode = target_failure
            template = random.choice(NARRATIVE_CORPUS[target_failure]["early"])
            narrative = template.format(
                vib=vib, temp=temp, press=press, rpm=rpm,
                component=component, equipment_id=equipment_id
            )
        elif stage == "medium":
            temp = round(random.uniform(base_temp_max + 12, base_temp_max + 25), 1)
            vib = round(random.uniform(base_vib_max * 1.8, base_vib_max * 2.8), 2)
            press = round(base_press_min * 0.65) if base_press_max > 0 else 0
            current_failure_mode = target_failure
            # Mix early and advanced
            template = random.choice(NARRATIVE_CORPUS[target_failure]["early"] + NARRATIVE_CORPUS[target_failure]["advanced"])
            narrative = f"[Follow-up Alert] Previous inspection flagged anomaly. " + template.format(
                vib=vib, temp=temp, press=press, rpm=rpm,
                component=component, equipment_id=equipment_id
            )
        else: # critical
            temp = round(random.uniform(base_temp_max + 25, base_temp_max + 50), 1)
            vib = round(random.uniform(base_vib_max * 2.8, base_vib_max * 4.5), 2)
            press = round(base_press_min * 0.3) if base_press_max > 0 else 0
            current_failure_mode = target_failure
            template = random.choice(NARRATIVE_CORPUS[target_failure]["advanced"])
            narrative = f"[CRITICAL WORK ORDER] Equipment tripped during operation. " + template.format(
                vib=vib, temp=temp, press=press, rpm=rpm,
                component=component, equipment_id=equipment_id
            )
        
        telemetry_str = f"VIB: {vib:.2f} mm/s | TEMP: {temp:.1f} C | RPM: {rpm}" + (f" | PRESS: {press} PSI" if press > 0 else "")
        recommendation = RECOMMENDATIONS[current_failure_mode]
        
        record = {
            "equipment_id": equipment_id,
            "equipment_type": eq_meta["name"],
            "component": component,
            "log_date": current_date.strftime("%Y-%m-%d"),
            "visit_sequence": visit_idx + 1,
            "total_visits": num_visits,
            "narrative": narrative,
            "vibration_mms": vib,
            "temperature_c": temp,
            "pressure_psi": press,
            "rpm": rpm,
            "telemetry_summary": telemetry_str,
            "failure_mode": current_failure_mode,
            "risk_level": risk,
            "early_warning_flag": is_early_warn,
            "time_to_failure_days": ttf,
            "recommendation": recommendation
        }
        records.append(record)
        
    return records


def generate_full_dataset(num_equipments=750, output_dir="data"):
    """
    Generates dataset with thousands of maintenance records and temporal sequences.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_records = []
    equipment_trajectories = {}
    
    eq_type_keys = list(EQUIPMENT_TYPES.keys())
    
    for i in range(1, num_equipments + 1):
        eq_type = eq_type_keys[i % len(eq_type_keys)]
        eq_id = f"{eq_type}-{100 + i}"
        
        # Decide if this equipment experiences a failure trajectory or stays normal
        # 30% normal, 70% developing one of the valid failure modes for this equipment type
        if random.random() < 0.28:
            target_failure = "Normal_Operation"
        else:
            valid_failures = [f for f in EQUIPMENT_TYPES[eq_type]["failure_modes"] if f != "Normal_Operation"]
            target_failure = random.choice(valid_failures) if valid_failures else "Bearing_Degradation"
        
        num_visits = random.randint(4, 6)
        trajectory = generate_equipment_trajectory(eq_id, eq_type, target_failure, num_visits=num_visits)
        all_records.extend(trajectory)
        equipment_trajectories[eq_id] = trajectory
        
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "industrial_maintenance_dataset.csv")
    json_path = os.path.join(output_dir, "equipment_temporal_sequences.json")
    
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(equipment_trajectories, f, indent=2)
        
    print(f"Generated {len(df)} maintenance records across {num_equipments} industrial machines.")
    print(f"Saved dataset CSV to: {csv_path}")
    print(f"Saved temporal sequences JSON to: {json_path}")
    print("\nClass distribution for failure_mode:")
    print(df["failure_mode"].value_counts())
    print("\nClass distribution for risk_level:")
    print(df["risk_level"].value_counts())
    
    return df, equipment_trajectories


if __name__ == "__main__":
    generate_full_dataset(num_equipments=800)
