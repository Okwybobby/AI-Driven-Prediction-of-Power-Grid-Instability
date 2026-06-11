import csv, math, random
random.seed(42)

NOMINAL_KV = 330.0
Q_P_RATIO = 0.6012

FIELDNAMES = [
    'Record_ID','Scenario','Source','Sim_Time_s','Bus_ID','Bus_Name',
    'Frequency_Hz','Voltage_pu','Voltage_kV','Angle_deg',
    'Active_Power_MW','Reactive_Power_MVAr','ROCOF_Hz_per_s',
    'Ambient_Temp_C','Grid_State','Stability_Label'
]

def classify(f, v):
    if f <= 0.5 or v <= 0.05:        return 2, 'Collapse'
    if f < 48.5 or v < 0.80:         return 2, 'Collapse'
    if f < 49.0 or f > 51.5 or v < 0.90: return 1, 'Unstable'
    if f < 49.5 or f > 50.5 or v < 0.95: return 1, 'Warning'
    return 0, 'Stable'

def make_row(rid, t, f, v, angle, p, rocof, temp, scenario):
    f = max(0.0, f); v = max(0.0, v); p = max(0.0, p)
    lbl, state = classify(f, v)
    return [
        rid, scenario, 'Synthetic_PhysicsInformed',
        round(t, 4), 'Synth_NigGrid_Bus', 'Nigeria 330kV Grid (Synthetic)',
        round(f, 4), round(v, 4), round(v * NOMINAL_KV, 1), round(angle, 2),
        round(p, 3), round(p * Q_P_RATIO, 3), round(rocof, 4),
        temp, state, lbl
    ]

rows = []
rid = 1

# ============================================================
# SCENARIO 1: Normal Operation (4000 rows)
# Nominal grid: f~50Hz, V~1.0pu, slow mean-reverting oscillations
# ============================================================
t = 0.0; f = 50.0; v = 1.0; angle = 20.0; p = 430.0; prev_f = 50.0
for i in range(4000):
    temp = round(30 + 5 * math.sin(2 * math.pi * i / 500), 1)
    df = random.gauss(0, 0.08) + 0.05 * (50.0 - f)
    dv = random.gauss(0, 0.007) + 0.05 * (1.0 - v)
    f  = max(49.5, min(50.5, f + df))
    v  = max(0.97, min(1.05, v + dv))
    angle = 20.0 + 25.0 * math.sin(2 * math.pi * i / 40) + random.gauss(0, 2)
    p  = max(300, min(550, p + random.gauss(0, 20) + 0.05 * (430 - p)))
    rocof = (f - prev_f) / 0.1; prev_f = f
    rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Normal_Operation'))
    rid += 1; t += 0.1

# ============================================================
# SCENARIO 2: Light Load Night (1000 rows)
# Off-peak hours: f slightly above nominal, V high, P low
# ============================================================
t = 0.0; f = 50.15; v = 1.02; p = 270.0; prev_f = 50.15
for i in range(1000):
    temp = round(24 + 3 * math.sin(2 * math.pi * i / 200), 1)
    df = random.gauss(0, 0.05) + 0.04 * (50.1 - f)
    dv = random.gauss(0, 0.004) + 0.04 * (1.02 - v)
    f  = max(49.85, min(50.45, f + df))
    v  = max(0.99, min(1.05, v + dv))
    angle = 14.0 + 12.0 * math.sin(2 * math.pi * i / 35) + random.gauss(0, 1.5)
    p  = max(180, min(370, p + random.gauss(0, 14) + 0.05 * (270 - p)))
    rocof = (f - prev_f) / 0.1; prev_f = f
    rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'LightLoad_Night'))
    rid += 1; t += 0.1

# ============================================================
# SCENARIO 3: Heavy Load Peak (1200 rows)
# Peak demand: f dips to ~49.5Hz, V sags to ~0.97pu
# ============================================================
t = 0.0; f = 49.55; v = 0.972; p = 505.0; prev_f = 49.55
for i in range(1200):
    temp = round(36 + 3 * math.sin(2 * math.pi * i / 300), 1)
    df = random.gauss(0, 0.12) + 0.04 * (49.5 - f)
    dv = random.gauss(0, 0.008) + 0.04 * (0.97 - v)
    f  = max(49.1, min(50.1, f + df))
    v  = max(0.945, min(1.00, v + dv))
    angle = 36.0 + 20.0 * math.sin(2 * math.pi * i / 40) + random.gauss(0, 3)
    p  = max(380, min(630, p + random.gauss(0, 25) + 0.04 * (500 - p)))
    rocof = (f - prev_f) / 0.1; prev_f = f
    rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'HeavyLoad_Peak'))
    rid += 1; t += 0.1

# ============================================================
# SCENARIO 4: Line Fault Events (20 events)
# 8 collapse, 12 recover after fault clearing
# Each: 20 pre-fault + 8 fault onset + 25 unstable + 25 collapse OR 50 recovery
# ============================================================
for event in range(20):
    t = 0.0
    f = 50.0 + random.uniform(-0.15, 0.15)
    v = 1.0  + random.uniform(-0.01, 0.01)
    angle = 20.0 + random.uniform(-3, 3)
    p = 440.0 + random.uniform(-40, 40)
    temp = round(31 + random.uniform(0, 9), 1)
    prev_f = f
    will_collapse = (event < 8)

    # Pre-fault steady state
    for j in range(20):
        df = random.gauss(0, 0.02); dv = random.gauss(0, 0.003)
        f  = max(49.8, min(50.2, f + df + 0.02 * (50 - f)))
        v  = max(0.98, min(1.02, v + dv + 0.02 * (1  - v)))
        angle += random.gauss(0, 1); p += random.gauss(0, 10)
        rocof = (f - prev_f) / 0.1; prev_f = f
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Fault_PreFault'))
        rid += 1; t += 0.1

    # Fault onset — sudden V drop, f deviates
    for j in range(8):
        v    -= random.uniform(0.025, 0.055)
        f    -= random.uniform(0.04, 0.12)
        rocof = -random.uniform(0.4, 1.5); prev_f = f
        angle += random.uniform(5, 20)
        p    += random.uniform(0, 50)
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Fault_Onset'))
        rid += 1; t += 0.1

    # Unstable oscillations
    for j in range(25):
        if will_collapse:
            df = -random.uniform(0.08, 0.5); dv = -random.uniform(0.01, 0.04)
            rocof = -random.uniform(0.2, 2.0)
        else:
            df = random.gauss(-0.04, 0.3); dv = random.gauss(0, 0.018)
            rocof = random.gauss(-0.08, 0.3)
        f = max(0, f + df); v = max(0, v + dv)
        angle += random.gauss(0, 18)
        p = max(0, p + random.gauss(0, 55)); prev_f = f
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Fault_Unstable'))
        rid += 1; t += 0.1

    if will_collapse:
        for j in range(25):
            f = max(0, f - random.uniform(0.8, 3.5))
            v = max(0, v - random.uniform(0.04, 0.14))
            rocof = -random.uniform(1.0, 5.0)
            p = max(0, p * 0.75)
            if f < 1.0 and v < 0.1:
                f = 0.0; v = 0.0; p = 0.0
            rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Fault_Collapse'))
            rid += 1; t += 0.1
    else:
        for j in range(50):
            f     += 0.18 * (50  - f) + random.gauss(0, 0.05)
            v     += 0.12 * (1.0 - v) + random.gauss(0, 0.005)
            angle += 0.10 * (22  - angle) + random.gauss(0, 2)
            p     += 0.10 * (430 - p) + random.gauss(0, 15)
            rocof  = random.gauss(0, 0.04)
            rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Fault_Recovery'))
            rid += 1; t += 0.1

# ============================================================
# SCENARIO 5: Voltage Instability (10 events × 85 steps)
# Reactive demand surge: V collapses while f stays near 50 Hz
# ============================================================
for event in range(10):
    t = 0.0
    f = 50.0 + random.gauss(0, 0.06)
    v = 1.0; p = 370.0 + random.uniform(-40, 40)
    temp = round(33 + random.uniform(0, 6), 1)
    prev_f = f

    for j in range(65):
        df = random.gauss(0, 0.04) + 0.02 * (50.0 - f)
        f  = max(49.6, min(50.4, f + df))
        v -= random.uniform(0.004, 0.01)
        p += random.uniform(0, 18)
        rocof = (f - prev_f) / 0.1; prev_f = f
        angle = 24 + random.gauss(0, 6)
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Voltage_Instability'))
        rid += 1; t += 0.1

    for j in range(20):
        v = max(0, v - random.uniform(0.04, 0.12))
        f = max(0, f - random.uniform(0.05, 0.60))
        p = max(0, p * 0.82)
        rocof = -random.uniform(0.3, 2.5)
        if v < 0.05:
            f = 0.0; v = 0.0; p = 0.0
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'Voltage_Collapse'))
        rid += 1; t += 0.1

# ============================================================
# SCENARIO 6: Generation Loss — Frequency Drop (12 events × 50 steps)
# Sudden loss of a generating unit; f plummets; V sags secondary
# ============================================================
for event in range(12):
    t = 0.0
    f = 50.0 + random.gauss(0, 0.06)
    v = 1.0 + random.gauss(0, 0.01)
    p = 445.0 + random.uniform(-25, 25)
    temp = round(34 + random.uniform(0, 7), 1)
    prev_f = f

    for j in range(50):
        rocof = -random.uniform(0.08, 0.55)
        f = max(0, f + rocof * 0.1 + random.gauss(0, 0.04))
        v_adj = -0.003 * max(0, (50.0 - f))
        v = max(0.5, min(1.1, v + v_adj + random.gauss(0, 0.008)))
        p = max(0, p - random.uniform(3, 14))
        angle = 26 + random.gauss(0, 12)
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'GenLoss_FreqDrop'))
        rid += 1; t += 0.1; prev_f = f

# ============================================================
# SCENARIO 7: Daily Load Cycle (1500 rows)
# Simulates natural 24-hour demand variation; f/V follow load profile
# ============================================================
t = 0.0; f = 50.0; v = 1.0; p = 350.0; prev_f = 50.0
for i in range(1500):
    hour_equiv = (i / 1500) * 24
    if 6 <= hour_equiv <= 22:
        load_factor = 0.70 + 0.30 * math.sin(math.pi * (hour_equiv - 6) / 16)
    else:
        load_factor = 0.60
    p_target = 430.0 * load_factor
    f_target = 50.0 - 0.15 * (load_factor - 0.85)

    df = random.gauss(0, 0.10) + 0.06 * (f_target - f)
    dv = random.gauss(0, 0.007) + 0.05 * (1.0 - v)
    f  = max(49.3, min(50.5, f + df))
    v  = max(0.95, min(1.03, v + dv))
    angle = 18.0 + 22.0 * math.sin(2 * math.pi * i / 40) + random.gauss(0, 2.5)
    p  = max(250, min(580, p + random.gauss(0, 20) + 0.06 * (p_target - p)))
    temp = round(28 + 8 * math.sin(math.pi * max(0, hour_equiv - 6) / 16), 1)
    rocof = (f - prev_f) / 0.1; prev_f = f
    rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'DailyLoadCycle'))
    rid += 1; t += 60.0  # 1-minute intervals

# ============================================================
# SCENARIO 8: Under-Frequency Load Shedding Response (500 rows)
# f drops, load shed restores it — common Nigerian grid event
# ============================================================
for event in range(5):
    t = 0.0
    f = 49.8 + random.uniform(-0.1, 0.1)
    v = 0.985 + random.uniform(-0.01, 0.01)
    p = 480.0 + random.uniform(-20, 20)
    temp = round(32 + random.uniform(0, 6), 1)
    prev_f = f

    # Frequency declining phase (40 steps)
    for j in range(40):
        rocof = -random.uniform(0.02, 0.12)
        f = max(48.5, f + rocof * 0.1 + random.gauss(0, 0.03))
        v = max(0.93, v - random.uniform(0, 0.003))
        p += random.uniform(0, 8)
        angle = 30 + random.gauss(0, 8)
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'UFLS_Declining'))
        rid += 1; t += 0.1; prev_f = f

    # Load shed event — f bounces back (20 steps)
    p *= 0.75  # 25% load shed
    for j in range(20):
        rocof = random.uniform(0.05, 0.25)
        f = min(50.3, f + rocof * 0.1 + random.gauss(0, 0.04))
        v = min(1.0,  v + random.uniform(0.003, 0.01))
        angle = 25 + random.gauss(0, 5)
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'UFLS_Shedding'))
        rid += 1; t += 0.1; prev_f = f

    # Recovery stabilisation (40 steps)
    for j in range(40):
        f += 0.12 * (50.0 - f) + random.gauss(0, 0.04)
        v += 0.10 * (0.99 - v) + random.gauss(0, 0.004)
        angle += 0.08 * (22 - angle) + random.gauss(0, 2)
        p += 0.08 * (430 - p) + random.gauss(0, 12)
        rocof = random.gauss(0, 0.03)
        rows.append(make_row(rid, t, f, v, angle, p, rocof, temp, 'UFLS_Recovery'))
        rid += 1; t += 0.1

# ============================================================
# Write CSV
# ============================================================
OUTPUT = r'G:\My Drive\Afribary\nigeria_grid_stability_augmented.csv'
with open(OUTPUT, 'w', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(FIELDNAMES)
    writer.writerows(rows)

labels   = [r[-1] for r in rows]
scenarios = [r[1] for r in rows]
total    = len(rows)
print(f"Total rows written : {total}")
print(f"Label 0 (Stable)   : {labels.count(0):6d}  ({100*labels.count(0)/total:.1f}%)")
print(f"Label 1 (Unstable) : {labels.count(1):6d}  ({100*labels.count(1)/total:.1f}%)")
print(f"Label 2 (Collapse) : {labels.count(2):6d}  ({100*labels.count(2)/total:.1f}%)")
print()
seen = {}
for s in scenarios:
    seen[s] = seen.get(s, 0) + 1
for k, v2 in sorted(seen.items()):
    print(f"  {k:30s}: {v2}")
