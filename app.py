import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import io
import warnings

warnings.filterwarnings('ignore')

# ================== DAFTAR HARI LIBUR NASIONAL INDONESIA 2025-2026 ==================
INDONESIA_HOLIDAYS = {
    # 2025 Holidays
    '2025-01-01', '2025-03-03', '2025-03-31', '2025-04-01', '2025-04-18',
    '2025-05-01', '2025-05-12', '2025-05-29', '2025-06-01', '2025-06-07',
    '2025-08-17', '2025-09-05', '2025-12-25', '2025-12-26',

    # 2026 Holidays
    '2026-01-01', '2026-02-17', '2026-03-28', '2026-03-29', '2026-04-03',
    '2026-05-01', '2026-05-02', '2026-05-21', '2026-06-07', '2026-08-17',
    '2026-09-05', '2026-12-25', '2026-12-26'
}

# ================== UTILITY FUNCTIONS DENGAN HOLIDAYS ==================
WORK_HOURS_PER_DAY = 12.5


def is_holiday(date):
    """Cek apakah tanggal termasuk hari libur nasional."""
    date_str = date.strftime('%Y-%m-%d')
    return date_str in INDONESIA_HOLIDAYS


def is_workday(date):
    """Cek apakah tanggal adalah hari kerja (bukan weekend dan bukan libur nasional)."""
    return date.weekday() < 5 and not is_holiday(date)


def next_workday(date):
    """Geser ke hari kerja berikutnya jika weekend atau libur."""
    d = pd.to_datetime(date)
    while not is_workday(d):
        d += timedelta(days=1)
    return d


def adjust_to_weekday(date):
    """Adjust date to next workday (considering holidays)"""
    d = pd.to_datetime(date)
    while not is_workday(d):
        d += timedelta(days=1)
    return d


def calculate_end_date(start_date, duration):
    """Calculate end date considering workdays and holidays"""
    if pd.isna(start_date):
        return None
    end_date = pd.to_datetime(start_date)
    days_added = 0
    while days_added < duration - 1:
        end_date += timedelta(days=1)
        if is_workday(end_date):
            days_added += 1
    return end_date


def calculate_working_days(start_str, end_str):
    """Calculate working days between two dates (excluding weekends and holidays)"""
    if start_str is None or end_str is None:
        return None

    s = pd.to_datetime(start_str, errors='coerce')
    e = pd.to_datetime(end_str, errors='coerce')

    if pd.isna(s) or pd.isna(e):
        return None

    # Use numpy busday_count with custom weekmask and holidays
    start_date = s.date()
    end_date = e.date()

    # Convert holidays to datetime64
    holidays_list = [np.datetime64(holiday) for holiday in INDONESIA_HOLIDAYS]

    # Count business days (Monday=0 to Friday=4)
    days = np.busday_count(
        start_date,
        end_date + timedelta(days=1),
        weekmask='1111100',  # Monday-Friday are working days
        holidays=holidays_list
    )
    return int(days)


def calculate_calendar_days(start_dt, end_dt):
    """Calculate total calendar days"""
    if pd.isna(start_dt) or pd.isna(end_dt):
        return None
    s = pd.to_datetime(start_dt)
    e = pd.to_datetime(end_dt)
    return max((e.date() - s.date()).days + 1, 1)


def manhour_to_days(manhour):
    """Convert manhours to working days"""
    return int(manhour / 12.5) + (0 if manhour % 12.5 == 0 else 1)


def add_working_days(start_date, work_days):
    """Add working days considering weekends and holidays"""
    current = pd.to_datetime(start_date)
    added = 0
    while added < work_days:
        current += timedelta(days=1)
        if is_workday(current):
            added += 1
    return current


# ================== CALCULATE LEAD TIME (MORE ROBUST) ==================
def calculate_lead_time(df_final_schedule):
    """
    Calculate Lead Time dari Create PRO sampai Max Process (untuk PO1 saja)
    Handle case dimana PRO bisa None untuk proses QFD
    """
    if df_final_schedule.empty:
        return 0

    # Buat copy dan convert dates
    df = df_final_schedule.copy()
    df['Start_dt'] = pd.to_datetime(df['Start'], errors='coerce')
    df['End_dt'] = pd.to_datetime(df['End'], errors='coerce')
    df = df[df['Start_dt'].notna() & df['End_dt'].notna()]

    if df.empty:
        return 0

    # STRATEGI 1: Cari Create PrO untuk PO1 secara eksplisit
    create_pro_po1 = df[(df['PRO'] == 'PO1') & (df['Process'] == 'Create PrO')]

    # STRATEGI 2: Jika tidak ketemu, cari Create PrO dengan PRO None (asumsi untuk PO1)
    if create_pro_po1.empty:
        create_pro_po1 = df[(df['PRO'].isna()) & (df['Process'] == 'Create PrO')]

    if create_pro_po1.empty:
        return 0

    create_pro_date = create_pro_po1['Start_dt'].iloc[0]

    # Cari max process date untuk PO1
    po1_processes = df[df['PRO'] == 'PO1']

    if po1_processes.empty:
        return 0

    max_process_date = po1_processes['End_dt'].max()

    # Hitung working days
    lead_time = calculate_working_days(
        create_pro_date.strftime('%Y-%m-%d'),
        max_process_date.strftime('%Y-%m-%d')
    )

    return lead_time if lead_time else 0


# ================== GET CRITICAL MATERIAL INFO ==================
def get_critical_material_info(df_po):
    """
    Get information about critical material (material dengan lead time terlama)
    """
    if df_po.empty or df_po['Adjusted_LeadTime'].notna().sum() == 0:
        return "No data", 0, "No data"

    critical_material_row = df_po.loc[df_po['Adjusted_LeadTime'].idxmax()]
    critical_material = critical_material_row['Material']
    critical_lt = critical_material_row['Adjusted_LeadTime']
    critical_process = critical_material_row['Process']

    return critical_material, critical_lt, critical_process


# ================== IMPROVED GANTT CHART FUNCTION ==================
def create_gantt_chart(df_final_schedule):
    """Create Gantt chart with proper data validation"""
    if df_final_schedule.empty:
        return None

    # Prepare data for Gantt chart
    gantt_data = df_final_schedule.copy()

    # Check for required columns
    required_cols = ['PRO', 'Process', 'Start', 'End']
    missing_cols = [col for col in required_cols if col not in gantt_data.columns]
    if missing_cols:
        st.error(f"Missing columns for Gantt chart: {missing_cols}")
        return None

    # Filter out rows with missing PRO (like QFD processes)
    gantt_data = gantt_data[gantt_data['PRO'].notna()]

    # Convert dates
    gantt_data['Start_dt'] = pd.to_datetime(gantt_data['Start'], errors='coerce')
    gantt_data['End_dt'] = pd.to_datetime(gantt_data['End'], errors='coerce')

    # Remove rows with invalid dates
    gantt_data = gantt_data[gantt_data['Start_dt'].notna() & gantt_data['End_dt'].notna()]

    # Ensure End_dt is after Start_dt
    gantt_data = gantt_data[gantt_data['End_dt'] >= gantt_data['Start_dt']]

    if gantt_data.empty:
        st.warning("No valid data available for Gantt chart after filtering")
        return None

    # Create Gantt chart
    try:
        fig = px.timeline(
            gantt_data,
            x_start="Start_dt",
            x_end="End_dt",
            y="PRO",
            color="Process",
            title="Production Schedule Gantt Chart",
            hover_data=["Process", "Lead Time", "Keterangan"],
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Update layout for better appearance
        fig.update_layout(
            xaxis_title="Timeline",
            yaxis_title="Production Order (PRO)",
            height=max(400, len(gantt_data['PRO'].unique()) * 50 + 200),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat="%Y-%m-%d")

        return fig

    except Exception as e:
        st.error(f"Error creating Gantt chart: {e}")
        return None


# ================== SUMMARY PROCESS FUNCTION ==================
def create_summary_process(df_final_schedule, MasterProcess, pn_input):
    """
    Create summary process per PRO
    """
    if df_final_schedule.empty:
        return pd.DataFrame()

    df_summary = df_final_schedule.copy()
    df_summary['PN'] = pn_input

    # Join dengan MasterProcess untuk dapetin ProcessGroup
    df_summary = df_summary.merge(
        MasterProcess[['PN', 'Process', 'ProcessGroup']],
        on=['PN', 'Process'],
        how='left'
    )

    # Pastikan tanggal dalam format datetime
    df_summary['Start'] = pd.to_datetime(df_summary['Start'])
    df_summary['End'] = pd.to_datetime(df_summary['End'])

    # Fungsi untuk hitung hari unik (tidak dobel di tanggal yang sama)
    def unique_days(group):
        days = []
        for _, row in group.iterrows():
            day_range = pd.date_range(row['Start'], row['End'])
            days.extend(day_range)
        return len(set(days))

    # Hitung Lead Time unik per kombinasi PRO + ProcessGroup
    summary_process = (
        df_summary.groupby(['PRO', 'ProcessGroup'])
        .apply(unique_days)
        .reset_index(name='Lead Time')
    )

    # Urutkan biar rapi
    summary_process = summary_process.sort_values(by=['PRO', 'ProcessGroup']).reset_index(drop=True)

    return summary_process


# ================== SCHEDULER LOGIC ==================
@st.cache_data
def build_schedule(
        Bom, LT_Material, MMBE, Subcont_Capacity, MasterProcess, SFS,
        pn_input, qty_unit, start_qfd, repeat_pn
):
    today = pd.to_datetime("today").normalize()

    # Update MMBE dengan Unrest_Stock
    MMBE = MMBE.copy()
    MMBE['Unrest_Stock'] = pd.to_numeric(MMBE['Unrest_Stock'], errors='coerce').fillna(0)
    MMBE = (
        MMBE.sort_values('Unrest_Stock', ascending=False)
        .drop_duplicates(subset='Material', keep='first')
        .reset_index(drop=True)
    )

    # Baseline / PO Allocation
    df_bom = (
        Bom[Bom['PN'] == pn_input]
        .merge(LT_Material, how='left', on=['Material', 'PN'])
        .merge(MMBE[['Material', 'Free Stock']], how='left', on='Material')
        .drop_duplicates(subset=['Material', 'Process'], keep='first').fillna(0)
    )

    po_list = []
    stock_available = df_bom.set_index("Material")["Free Stock"].to_dict()

    for i in range(1, qty_unit + 1):
        tmp = df_bom.copy()
        tmp["PO"] = f"PO{i}"
        tmp["PN"] = pn_input

        allocated_list = []
        lt_list = []

        for idx, row in tmp.iterrows():
            material = row["Material"]
            need = row.get("Qty", 0)
            try:
                need = float(need)
            except Exception:
                need = 0
            available = stock_available.get(material, 0)

            if available >= need:
                allocated = need
                lt = 0
                stock_available[material] = available - need
            else:
                allocated = available
                lt = row.get("Lead Time", 0)
                stock_available[material] = 0

            allocated_list.append(allocated)
            lt_list.append(lt)

        tmp["Allocated"] = allocated_list
        tmp["Lead Time_Final"] = lt_list
        po_list.append(tmp)

    df_po = pd.concat(po_list, ignore_index=True)

    # Adjustment Subcont_Capacity
    df_po = df_po.merge(
        Subcont_Capacity[['Material', 'Capacity', 'Shifting Day']],
        on="Material", how="left"
    )
    df_po["Adjusted_LeadTime"] = df_po["Lead Time_Final"]

    for material, group in df_po[df_po["Lead Time_Final"] != 0].groupby("Material"):
        if group["Capacity"].notna().all() and len(group) > 0:
            cap = int(group["Capacity"].iloc[0])
            shift_day = int(group["Shifting Day"].iloc[0])
            base_lt = int(group["Lead Time_Final"].iloc[0])
            order_idx = group.reset_index().index
            adjusted_lt = [
                base_lt + (i // cap) * shift_day
                for i in order_idx
            ]
            df_po.loc[group.index, "Adjusted_LeadTime"] = adjusted_lt

    df_po['Adjusted_LeadTime'] = pd.to_numeric(df_po['Adjusted_LeadTime'], errors='coerce')

    df_max_lt = df_po.groupby(["PO", "PN"])['Adjusted_LeadTime'].max().reset_index()
    df_max_lt.rename(columns={"Adjusted_LeadTime": "Max_Adjusted_LT"}, inplace=True)

    df_max_lt["Material_Available_Date"] = df_max_lt["Max_Adjusted_LT"].apply(
        lambda x: (today + timedelta(days=int(x))) if pd.notna(x) else today
    )

    # Ambil max enddate dari SFS
    if 'End Date' in SFS.columns:
        SFS = SFS.copy()
        SFS['End Date'] = pd.to_datetime(SFS['End Date'], errors='coerce')
        df_end = SFS.groupby('PN')['End Date'].max().reset_index().rename(columns={'End Date': 'Max_EndDate'})
    else:
        df_end = pd.DataFrame(columns=['PN', 'Max_EndDate'])

    df_compare = df_max_lt.merge(df_end, on='PN', how='left')
    df_compare['Material_Available_Date'] = pd.to_datetime(df_compare['Material_Available_Date'], errors='coerce')
    df_compare['Max_EndDate'] = pd.to_datetime(df_compare['Max_EndDate'], errors='coerce')

    def decide_status(row):
        max_end = row.get('Max_EndDate', pd.NaT)
        mat_avail = row.get('Material_Available_Date', pd.NaT)
        if pd.isna(max_end) and pd.isna(mat_avail):
            return 'No Data'
        if pd.isna(max_end):
            return 'Material_Available Lebih Lama'
        if pd.isna(mat_avail):
            return 'Max_EndDate Lebih Lama'
        if max_end > mat_avail:
            return 'Max_EndDate Lebih Lama'
        else:
            return 'Material_Available Lebih Lama'

    if not df_compare.empty:
        df_compare['Status'] = df_compare.apply(decide_status, axis=1)
    else:
        df_compare['Status'] = []

    df_compare_baseline = df_compare.copy()

    # QFD / SLA Schedule
    def create_qfd_schedule(start_date, repeat_pn):
        qfd_processes = [
            ("Design", 5), ("Workflow & Validasi", 3), ("LPPB", 2),
            ("BOM & Routing", 1), ("Upload BOM to SAP", 1), ("Create PrO", 1),
            ("Create PR", 1), ("Sourching Material", 3), ("Release PR", 2),
            ("Create PO", 1), ("Release PO", 1)
        ]

        start_idx = 0
        if repeat_pn == 'Y':
            for idx, (proc, dur) in enumerate(qfd_processes):
                if proc == 'Create PrO':
                    start_idx = idx
                    break

        schedule_qfd = []
        current_date = adjust_to_weekday(start_date)

        for process, duration in qfd_processes[start_idx:]:
            end_date = calculate_end_date(current_date, duration) if duration > 0 else current_date
            lead_time_days = max(1, calculate_working_days(current_date.strftime('%Y-%m-%d'),
                                                           end_date.strftime('%Y-%m-%d')))
            schedule_qfd.append({
                'Process': process,
                'Start': current_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d'),
                'Lead Time': lead_time_days,
                'Note': None if duration > 0 else 'Repeat PN - No Design Needed',
                'PRO': None,
                'Keterangan': ''
            })
            current_date = adjust_to_weekday(end_date + timedelta(days=1))

        return schedule_qfd

    schedule_qfd = create_qfd_schedule(start_qfd, repeat_pn)

    # Incoming Component per PO
    release_po_rows = [row for row in schedule_qfd if row['Process'] == 'Release PO']
    if len(release_po_rows) == 0:
        raise ValueError("Tidak menemukan process 'Release PO' di schedule_qfd.")
    release_po_end = pd.to_datetime(release_po_rows[0]['End'], errors='coerce')

    incoming_schedule = []
    for _, row in df_compare_baseline.iterrows():
        po = row['PO']
        incoming_start = release_po_end
        lt_val = None
        if pd.notna(row.get('Max_Adjusted_LT', np.nan)):
            try:
                lt_val = int(row.get('Max_Adjusted_LT'))
            except Exception:
                lt_val = None

        if lt_val is not None and lt_val >= 0:
            incoming_end = incoming_start + timedelta(days=lt_val)
            lead_time = lt_val
            note = f"Based on release_po_end + Max_Adjusted_LT ({lt_val} hari)"
        else:
            max_end = pd.to_datetime(row.get('Max_EndDate', pd.NaT), errors='coerce')
            mat_avail = pd.to_datetime(row.get('Material_Available_Date', pd.NaT), errors='coerce')
            if pd.isna(max_end) and pd.isna(mat_avail):
                incoming_end = incoming_start
                lead_time = 1
                note = 'No data'
            else:
                chosen = max_end if (pd.notna(max_end) and (pd.isna(mat_avail) or max_end >= mat_avail)) else mat_avail
                if pd.isna(chosen) or chosen < incoming_start:
                    incoming_end = incoming_start
                    lead_time = 1
                else:
                    incoming_end = chosen
                    lead_time = calculate_calendar_days(incoming_start, incoming_end) or 1
                note = 'Fallback based on Max_EndDate/Material_Available_Date'

        mat_avail_dt = pd.to_datetime(row.get('Material_Available_Date', pd.NaT), errors='coerce')
        max_end_dt = pd.to_datetime(row.get('Max_EndDate', pd.NaT), errors='coerce')

        if pd.notna(mat_avail_dt) and pd.notna(max_end_dt):
            if mat_avail_dt > max_end_dt:
                keterangan = 'Waiting Material Complete'
            elif max_end_dt > mat_avail_dt:
                keterangan = 'Waiting Capacity Available'
            else:
                keterangan = 'On Time'
        elif pd.notna(mat_avail_dt):
            keterangan = 'Waiting Material Complete'
        elif pd.notna(max_end_dt):
            keterangan = 'Waiting Capacity Available'
        else:
            keterangan = 'No Data'

        incoming_schedule.append({
            'Process': 'Incoming Component',
            'Start': incoming_start.strftime('%Y-%m-%d'),
            'End': incoming_end.strftime('%Y-%m-%d'),
            'Lead Time': int(lead_time),
            'Note': note,
            'PRO': po,
            'Material_Available_Date': mat_avail_dt,
            'Max_EndDate': max_end_dt,
            'Keterangan': keterangan
        })

    incoming_df = pd.DataFrame(incoming_schedule)
    incoming_df['End'] = pd.to_datetime(incoming_df['End'], errors='coerce')
    incoming_end_map = incoming_df.set_index('PRO')['End'].to_dict() if not incoming_df.empty else {}

    # Reorder PROs berdasarkan incoming_end
    incoming_end_sorted = incoming_df.groupby('PRO')['End'].max().sort_values(
        na_position='last').reset_index() if not incoming_df.empty else pd.DataFrame()
    pro_mapping = {old_pro: f"PO{i + 1}" for i, old_pro in
                   enumerate(incoming_end_sorted['PRO'])} if not incoming_end_sorted.empty else {}

    available_pos = list(df_compare_baseline['PO'].unique()) if 'PO' in df_compare_baseline.columns else []
    ordered_pos = [p for p in incoming_end_sorted['PRO'] if p in available_pos] if not incoming_end_sorted.empty else []
    remaining_pos = [p for p in available_pos if p not in ordered_pos]
    remaining_pos.sort()
    ordered_pos.extend(remaining_pos)

    # PRODUCTION SCHEDULING
    MasterProcess_Copy = MasterProcess.copy()
    MasterProcess_Copy['ManHour'] = pd.to_numeric(MasterProcess_Copy['ManHour'], errors='coerce') / pd.to_numeric(
        MasterProcess_Copy['ManPower'], errors='coerce')
    scheduler_filtered = MasterProcess_Copy[MasterProcess_Copy['PN'] == pn_input].copy()
    scheduler_filtered['ManHour'] = pd.to_numeric(scheduler_filtered.get('ManHour', 0), errors='coerce').fillna(0)
    scheduler_filtered['Maksimal Produksi per-Base'] = pd.to_numeric(
        scheduler_filtered.get('Maksimal Produksi per-Base', 1), errors='coerce'
    ).fillna(1).astype(int)

    production_schedule = []
    daily_capacity = {}
    FORWARD_DAYS = 14

    for po in ordered_pos:
        row = df_compare_baseline.loc[df_compare_baseline['PO'] == po].iloc[0]

        incoming_end_dt = incoming_end_map.get(po, pd.NaT)
        max_end_dt = pd.to_datetime(row.get('Max_EndDate', pd.NaT), errors='coerce')

        dates_ready = [d for d in [incoming_end_dt, max_end_dt] if pd.notna(d)]

        if dates_ready:
            max_date = max(dates_ready)
            anchor_start_date = adjust_to_weekday(max_date + timedelta(days=1))
        else:
            anchor_start_date = adjust_to_weekday(today + timedelta(days=1))

        processes_prod = scheduler_filtered.copy()
        dependency_graph = dict(zip(processes_prod['Process'], processes_prod['Dependency'].fillna('')))
        manhour_map = dict(zip(processes_prod['Process'], processes_prod['ManHour']))
        capacity_map = dict(zip(processes_prod['Process'], processes_prod['Maksimal Produksi per-Base']))

        scheduled_processes = {}
        remaining = list(dependency_graph.keys())

        # Cari anchor process dari material dengan lead time terlama
        anchor_process = "Assembly"  # default
        if df_po['Adjusted_LeadTime'].notna().any():
            critical_material_row = df_po.loc[df_po['Adjusted_LeadTime'].idxmax()]
            anchor_process = critical_material_row['Process']

        def find_pre_anchor_processes(anchor_proc, dep_graph):
            pre_processes = set()

            def find_dependencies(process):
                deps = dep_graph.get(process, '')
                if deps and deps != '':
                    for dep in deps.split(', '):
                        dep = dep.strip()
                        if dep and dep not in pre_processes:
                            pre_processes.add(dep)
                            find_dependencies(dep)

            find_dependencies(anchor_proc)
            return pre_processes

        pre_anchor_processes = set()
        if anchor_process and anchor_process in processes_prod['Process'].values:
            pre_anchor_processes = find_pre_anchor_processes(anchor_process, dependency_graph)
        else:
            pre_anchor_processes = set()

        forward_only = anchor_process not in processes_prod['Process'].values

        if forward_only:
            incoming_anchor_base = incoming_end_dt if pd.notna(incoming_end_dt) else today
            pre_anchor_chain_start = adjust_to_weekday(
                pd.to_datetime(incoming_anchor_base) + timedelta(days=FORWARD_DAYS))
            anchor_start_date = pre_anchor_chain_start
        else:
            pre_anchor_chain_start = None

        # STEP 1: SCHEDULE PRE-ANCHOR PROCESSES
        if not forward_only and anchor_process and pre_anchor_processes:
            total_pre_anchor_duration = 0
            for proc in pre_anchor_processes:
                if proc in manhour_map:
                    manhour = manhour_map[proc] or 0
                    total_pre_anchor_duration += manhour_to_days(manhour)

            pre_anchor_start = anchor_start_date
            days_to_subtract = total_pre_anchor_duration

            temp_date = pre_anchor_start
            while days_to_subtract > 0:
                temp_date -= timedelta(days=1)
                if is_workday(temp_date):
                    days_to_subtract -= 1

            pre_anchor_chain_start = temp_date

            temp_remaining = list(pre_anchor_processes)
            temp_scheduled = {}

            while temp_remaining:
                for process in temp_remaining[:]:
                    dep = dependency_graph.get(process, '')
                    manhour = manhour_map.get(process, 0) or 0
                    duration = manhour_to_days(manhour)

                    deps_met = True
                    if dep and dep != '':
                        for single_dep in dep.split(', '):
                            if single_dep.strip() not in temp_scheduled:
                                deps_met = False
                                break

                    if deps_met:
                        if dep == "":
                            s_date = pre_anchor_chain_start
                        else:
                            dep_end_dates = [temp_scheduled[d]['end'] for d in dep.split(', ') if
                                             d.strip() in temp_scheduled]
                            s_date = adjust_to_weekday(
                                max(dep_end_dates) + timedelta(days=1)) if dep_end_dates else pre_anchor_chain_start

                        max_per_day = capacity_map.get(process, 1)
                        while True:
                            key_check = (process, s_date.strftime("%Y-%m-%d"))
                            current_count = daily_capacity.get(key_check, 0)
                            if current_count < max_per_day:
                                break
                            s_date = adjust_to_weekday(s_date + timedelta(days=1))

                        current_date2 = s_date
                        days_counted = 0
                        while days_counted < duration:
                            if is_workday(current_date2):
                                key = (process, current_date2.strftime("%Y-%m-%d"))
                                daily_capacity[key] = daily_capacity.get(key, 0) + 1
                                days_counted += 1
                            current_date2 += timedelta(days=1)

                        e_date = calculate_end_date(s_date, duration) if duration > 0 else s_date
                        temp_scheduled[process] = {'start': s_date, 'end': e_date}

                        production_schedule.append({
                            "Process": process,
                            "Start": s_date.strftime("%Y-%m-%d"),
                            "End": e_date.strftime("%Y-%m-%d"),
                            "Lead Time": duration,
                            "Note": "",
                            "PRO": po,
                            "Keterangan": "Pre-Anchor"
                        })

                        scheduled_processes[process] = {'start': s_date, 'end': e_date}
                        temp_remaining.remove(process)
                        remaining.remove(process)

        # STEP 2: SCHEDULE ANCHOR PROCESS
        if not forward_only and anchor_process and anchor_process in remaining:
            dep = dependency_graph.get(anchor_process, '')
            anchor_s_date = anchor_start_date

            if dep and dep != '':
                dep_end_dates = [scheduled_processes[d]['end'] for d in dep.split(', ') if
                                 d.strip() in scheduled_processes]
                if dep_end_dates:
                    last_dep_end = max(dep_end_dates)
                    if last_dep_end >= anchor_start_date:
                        anchor_s_date = adjust_to_weekday(last_dep_end + timedelta(days=1))

            max_per_day = capacity_map.get(anchor_process, 1)
            while True:
                key_check = (anchor_process, anchor_s_date.strftime("%Y-%m-%d"))
                current_count = daily_capacity.get(key_check, 0)
                if current_count < max_per_day:
                    break
                anchor_s_date = adjust_to_weekday(anchor_s_date + timedelta(days=1))

            manhour = manhour_map.get(anchor_process, 0) or 0
            duration = manhour_to_days(manhour)

            current_date2 = anchor_s_date
            days_counted = 0
            while days_counted < duration:
                if is_workday(current_date2):
                    key = (anchor_process, current_date2.strftime("%Y-%m-%d"))
                    daily_capacity[key] = daily_capacity.get(key, 0) + 1
                    days_counted += 1
                current_date2 += timedelta(days=1)

            anchor_e_date = calculate_end_date(anchor_s_date, duration) if duration > 0 else anchor_s_date

            production_schedule.append({
                "Process": anchor_process,
                "Start": anchor_s_date.strftime("%Y-%m-%d"),
                "End": anchor_e_date.strftime("%Y-%m-%d"),
                "Lead Time": duration,
                "Note": f"Anchor Process (menunggu material terlama)",
                "PRO": po,
                "Keterangan": "Anchor Process"
            })

            scheduled_processes[anchor_process] = {'start': anchor_s_date, 'end': anchor_e_date}
            remaining.remove(anchor_process)

        # STEP 3: SCHEDULE REMAINING PROCESSES
        iteration_count = 0
        max_iterations = len(remaining) * 2

            iteration_count += 1
            for process in remaining[:]:
                dep = dependency_graph.get(process, '')
                manhour = manhour_map.get(process, 0) or 0
                duration = manhour_to_days(manhour)

                dependencies_met = True
                dep_end_dates = []
                if dep and dep != '':
                    for single_dep in dep.split(', '):
                        single_dep = single_dep.strip()
                        if single_dep and single_dep in scheduled_processes:
                            dep_end_dates.append(scheduled_processes[single_dep]['end'])
                        elif single_dep and single_dep not in scheduled_processes:
                            dependencies_met = False
                            break

                if dependencies_met:
                    if dep == "":
                        s_date = anchor_start_date if forward_only else pre_anchor_chain_start
                    else:
                        if dep_end_dates:
                            last_dep_end = max(dep_end_dates)
                            s_date = adjust_to_weekday(last_dep_end + timedelta(days=1))
                        else:
                            s_date = adjust_to_weekday(anchor_start_date + timedelta(days=1))

                    max_per_day = capacity_map.get(process, 1)
                    while True:
                        key_check = (process, s_date.strftime("%Y-%m-%d"))
                        current_count = daily_capacity.get(key_check, 0)
                        if current_count < max_per_day:
                            break
                        s_date = adjust_to_weekday(s_date + timedelta(days=1))

                    current_date2 = s_date
                    days_counted = 0
                    while days_counted < duration:
                        if is_workday(current_date2):
                            key = (process, current_date2.strftime("%Y-%m-%d"))
                            daily_capacity[key] = daily_capacity.get(key, 0) + 1
                            days_counted += 1
                        current_date2 += timedelta(days=1)

                    e_date = calculate_end_date(s_date, duration) if duration > 0 else s_date

                    production_schedule.append({
                        "Process": process,
                        "Start": s_date.strftime("%Y-%m-%d"),
                        "End": e_date.strftime("%Y-%m-%d"),
                        "Lead Time": duration,
                        "Note": "" if not forward_only else f"Forward Scheduling (Anchor '{anchor_process}' tidak ditemukan)",
                        "PRO": po,
                        "Keterangan": "Forward Process" if forward_only else "Post-Anchor"
                    })

                    scheduled_processes[process] = {'start': s_date, 'end': e_date}
                    remaining.remove(process)

        # HANDLE UNSCHEDULED PROCESSES
        if remaining:
            for process in remaining:
                production_schedule.append({
                    "Process": process,
                    "Start": anchor_start_date.strftime("%Y-%m-%d"),
                    "End": anchor_start_date.strftime("%Y-%m-%d"),
                    "Lead Time": 1,
                    "Note": "Unscheduled",
                    "PRO": po,
                    "Keterangan": "Error: Cannot schedule"
                })

    # Final merge
    df_final_schedule = pd.DataFrame(schedule_qfd + incoming_schedule + production_schedule)

    # normalisasi kolom Start/End jadi datetime jika ada, lalu format jadi YYYY-MM-DD
    if 'Start' in df_final_schedule.columns:
        df_final_schedule['Start'] = pd.to_datetime(df_final_schedule['Start'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        df_final_schedule['Start'] = pd.NaT

    if 'End' in df_final_schedule.columns:
        df_final_schedule['End'] = pd.to_datetime(df_final_schedule['End'], errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        df_final_schedule['End'] = pd.NaT

    # Pastikan PRO kolom ada dan mapping jika diperlukan
    if "PRO" in df_final_schedule.columns:
        df_final_schedule["PRO"] = df_final_schedule["PRO"].map(
            lambda x: pro_mapping.get(x, x) if 'pro_mapping' in globals() else x)

    # Pastikan kolom Lead Time ada dengan nama 'Lead Time'
    if 'Lead Time' not in df_final_schedule.columns:
        df_final_schedule['Lead Time'] = 0

    # Pastikan kolom Keterangan ada
    if 'Keterangan' not in df_final_schedule.columns:
        df_final_schedule['Keterangan'] = ""

    # Pilih hanya kolom yang diminta dan urutkan
    cols_wanted = ["PRO", "Process", "Start", "End", "Lead Time", "Keterangan"]
    for c in cols_wanted:
        if c not in df_final_schedule.columns:
            # isi default supaya tidak error
            df_final_schedule[c] = "" if c == "Keterangan" else pd.NaT if c in ["Start", "End"] else 0

    df_final_schedule = df_final_schedule[cols_wanted].copy()

    # Material delivery
    df_material_delivery = df_po.copy() if not df_po.empty else pd.DataFrame()
    df_material_delivery['DeliveryDate'] = pd.NaT
    if not df_material_delivery.empty:
        df_material_delivery['DeliveryDate'] = df_material_delivery['Adjusted_LeadTime'].apply(
            lambda x: release_po_end + timedelta(days=int(x)) if pd.notna(x) else release_po_end
        )
        df_material_delivery_output = df_material_delivery[[
            'PO', 'Material', 'Component_Desc', 'Adjusted_LeadTime', 'DeliveryDate'
        ]].rename(columns={'PO': 'PRO', 'Adjusted_LeadTime': 'LeadTime'})
        df_material_delivery_output['DeliveryDate'] = pd.to_datetime(df_material_delivery_output['DeliveryDate'],
                                                                     errors='coerce').dt.strftime('%Y-%m-%d')
    else:
        df_material_delivery_output = pd.DataFrame(
            columns=['PRO', 'Material', 'Component_Desc', 'LeadTime', 'DeliveryDate'])

    # Estimasi delivery per PRO
    pro_delivery = {}
    for pro in df_final_schedule['PRO'].dropna().unique():
        max_date = pd.to_datetime(df_final_schedule[df_final_schedule['PRO'] == pro]['End'], errors='coerce').max()
        pro_delivery[pro] = max_date

    df_delivery = pd.DataFrame(list(pro_delivery.items()), columns=['PO', 'Estimated Delivery Date'])
    df_delivery['Estimated Delivery Date'] = pd.to_datetime(df_delivery['Estimated Delivery Date'],
                                                            errors='coerce').dt.strftime('%Y-%m-%d')
    df_delivery = df_delivery.sort_values(by='Estimated Delivery Date', ascending=True, na_position='last').reset_index(
        drop=True)

    return df_final_schedule, df_material_delivery_output, df_delivery, df_po


# ================== Load Local Files ==================
@st.cache_data
def load_local_files():
    try:
        Bom = pd.read_excel('Bom.xlsx')
        LT_Material = pd.read_excel('LT_Material.xlsx')
        MMBE = pd.read_excel('MMBE.xlsx')
        Subcont_Capacity = pd.read_excel('Subcont_Capacity.xlsx')
        MasterProcess = pd.read_excel('MasterProcess.xlsx')
        SFS = pd.read_excel('SFS.xlsx')
        return Bom, LT_Material, MMBE, Subcont_Capacity, MasterProcess, SFS
    except Exception as e:
        st.error(f"Error loading files: {e}")
        # Return empty DataFrames if files not found
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# ================== Streamlit UI ==================
st.set_page_config(page_title='Automatic Delivery Date Estimation', layout='wide', page_icon='📊')

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📦 Automatic Delivery Date Estimation</h1>', unsafe_allow_html=True)

# Load files
Bom, LT_Material, MMBE, Subcont_Capacity, MasterProcess, SFS = load_local_files()

# Input Section
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("📋 Input Data")

col1, col2, col3, col4 = st.columns(4)
with col1:
    pn_input = st.text_input('Part Number (PN):', placeholder='Enter PN...')
with col2:
    qty_unit = st.number_input('Quantity Unit:', min_value=1, value=1, step=1)
with col3:
    start_qfd = st.date_input('PO Interco Date:', value=datetime.today())
with col4:
    repeat_pn = st.selectbox('Repeat PN?', options=['N', 'Y'], help='Select Y if this is a repeated part number')

run_btn = st.button('🚀 Build Production Schedule', type='primary', use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    if not pn_input:
        st.error("❌ Please enter a Part Number (PN)")
    elif Bom.empty or MasterProcess.empty:
        st.error(
            "❌ Required files (Bom.xlsx or MasterProcess.xlsx) not found. Please ensure these files are in the same directory.")
    else:
        with st.spinner('🔄 Building production schedule...'):
            try:
                df_final_schedule, df_material_delivery_output, df_delivery, df_po = build_schedule(
                    Bom, LT_Material, MMBE, Subcont_Capacity, MasterProcess, SFS,
                    pn_input, int(qty_unit), pd.to_datetime(start_qfd), repeat_pn
                )

                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success('✅ Production schedule successfully created!')
                st.markdown('</div>', unsafe_allow_html=True)

                # Calculate metrics
                lead_time = calculate_lead_time(df_final_schedule)
                critical_material, critical_lt, critical_process = get_critical_material_info(df_po)

                # Display only the required information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lead Time (working days)", lead_time)
                with col2:
                    st.metric("Material Critical", critical_material)
                with col3:
                    st.metric("Critical Lead Time (days)", critical_lt)

                # Display Supply Process
                st.info(f"**Supply Process:** {critical_process}")

                # Create summary process
                summary_process = create_summary_process(df_final_schedule, MasterProcess, pn_input)

                # Display Results in Tabs
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["📅 Production Schedule", "📊 Gantt Chart & Summary", "📦 Material Delivery", "🚚 Delivery Estimates"])

                with tab1:
                    st.dataframe(df_final_schedule, use_container_width=True)

                with tab2:
                    st.markdown('<div class="sub-header">Production Gantt Chart</div>', unsafe_allow_html=True)

                    # Create and display Gantt chart
                    fig = create_gantt_chart(df_final_schedule)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Unable to create Gantt chart. Check the data format above.")

                    st.markdown("---")
                    st.markdown('<div class="sub-header">Summary Process per PRO</div>', unsafe_allow_html=True)

                    if not summary_process.empty:
                        st.dataframe(summary_process, use_container_width=True)
                    else:
                        st.info("No summary process data available.")

                with tab3:
                    if not df_material_delivery_output.empty:
                        st.dataframe(df_material_delivery_output, use_container_width=True)
                    else:
                        st.info("No material delivery data available.")

                with tab4:
                    if not df_delivery.empty:
                        st.dataframe(df_delivery, use_container_width=True)
                    else:
                        st.info("No delivery estimates available.")

                # Download Section
                st.markdown("---")
                st.markdown('<div class="sub-header">📥 Download Results</div>', unsafe_allow_html=True)

                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                    df_final_schedule.to_excel(writer, sheet_name='Production_Schedule', index=False)
                    if not df_material_delivery_output.empty:
                        df_material_delivery_output.to_excel(writer, sheet_name='Material_Delivery', index=False)
                    if not df_delivery.empty:
                        df_delivery.to_excel(writer, sheet_name='Delivery_Estimates', index=False)
                    if not summary_process.empty:
                        summary_process.to_excel(writer, sheet_name='Summary_Process', index=False)
                towrite.seek(0)

                st.download_button(
                    '📄 Download All Results (Excel)',
                    data=towrite,
                    file_name=f'production_schedule_{pn_input}_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )

            except Exception as e:
                st.error(f'❌ Failed to build schedule: {str(e)}')
                st.info(
                    "Please check your input parameters and ensure all required data is available in the local files.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d;'>
        <p>Automatic Delivery Date Estimation • Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)