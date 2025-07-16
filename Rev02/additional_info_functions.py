import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

def find_cycle_breakpoints(raw_data):
    cycle_count_data = raw_data["Cycle Count"]
    total_cycle_count = int(cycle_count_data.max())

    cycle_ranges = []

    for i in range(1, total_cycle_count + 1):
        matching = cycle_count_data[cycle_count_data == i]
        if not matching.empty:
            start_idx = matching.index[0]
            end_idx = matching.index[-1]
            n_points = len(matching)

            # Calculate fractional index positions
            one_third_idx = matching.index[n_points // 3]
            middle_idx    = matching.index[n_points // 2]
            two_third_idx = matching.index[(2 * n_points) // 3]

            cycle_ranges.append((
                i,
                start_idx,
                one_third_idx,
                middle_idx,
                two_third_idx,
                end_idx
            ))

    # Create DataFrame with expanded columns
    df_ranges = pd.DataFrame(
        cycle_ranges,
        columns=[
            "Cycle",
            "Start Index",
            "One-Third Index",
            "Middle Index",
            "Two-Third Index",
            "End Index"
        ]
    )

    return df_ranges, total_cycle_count

def signed_distances_to_baseline(y: pd.Series) -> np.ndarray:
    x = np.arange(len(y))
    m = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
    b = y.iloc[0]
    # distance of each (x, y) to line y = m x + b
    return ((m * x - y + b) / np.hypot(m, -1))

def find_knee(data):
    sd = signed_distances_to_baseline(data)
    idx = np.argmax(sd)
    return data.iloc[idx], idx

def find_elbow(data):
    sd = signed_distances_to_baseline(data)
    idx = np.argmin(sd)
    return data.iloc[idx], idx

def locate_key_time_rows(cleaned_data, additional_info):
    """Return indices of key time points closest to provided timestamps."""
    if additional_info.iloc[1, 1] == "" or "NaN":
        holds_indices = additional_info.copy()
        holds_values = additional_info.copy()
        date_time_index = cleaned_data.set_index('Datetime')

        for row in range(1, len(holds_values)):
            holds_values.at[row, 1] = pd.to_datetime(
                holds_values.at[row, 1],
                format="%d/%m/%Y %H:%M:%S.%f",
                errors="coerce",
                dayfirst=True,
            )
            holds_indices.at[row, 1] = date_time_index.index.get_indexer(
                holds_values.iloc[row, 1],
                method="nearest",
            )[0]
            holds_values.at[row, 2] = cleaned_data[holds_indices.iloc[row, 1]][holds_values.at[0, 2]]
            holds_values.at[row, 3] = cleaned_data[holds_indices.iloc[row, 1]]["Body Temperature"]
        return holds_indices, holds_values
    else:
        return None, None

def locate_bto_btc_rows(raw_data, additional_info, channels_to_record):
    if additional_info.iloc[1, 1] == "" or "NaN" and channels_to_record.loc["Torque"].iloc[0]:
        breakout_values: List[Dict[str, Any]] = []
        breakout_indices: List[Dict[str, Any]] = []

        torque_series = raw_data["Torque"]

        # Get precomputed cycle boundaries
        indices_ranges, _ = find_cycle_breakpoints(raw_data)

        for cycle, start_idx, one_third_idx, middle_idx, two_third_idx, end_idx in indices_ranges.itertuples(index=False, name=None):
            # 1st third for BTO
            torque_first = torque_series.loc[start_idx:one_third_idx]
            bto = torque_first.min().round(1)
            bto_index = torque_first.idxmin()

            # 3rd third for BTC
            torque_third = torque_series.loc[middle_idx:two_third_idx]
            btc = torque_third.max().round(1)
            btc_index = torque_third.idxmax()

            # Store values
            breakout_values.append({
                "Cycle": cycle,
                "BTO": bto,
                "BTC": btc,
            })
            breakout_indices.append({
                "Cycle": cycle,
                "BTO_Index": bto_index,
                "BTC_Index": btc_index,
            })
        return (
            pd.DataFrame.from_records(breakout_values), 
            pd.DataFrame.from_records(breakout_indices),
        )
    else:
        return None, None

def locate_signature_key_points(
    channels_to_record: pd.DataFrame,
    raw_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Processes raw_data to find signature key points for each cycle.
    Returns a DataFrame with one row per cycle and columns for each key point and its index.
    """
    def find_a1() -> Tuple[Optional[float], Optional[int]]:
        """Finds A1 (Backseat Elbow)."""
        if channels_to_record.at["Backseat", 1]:
            bs_slice = backseat_data.loc[:middle_idx]
            _, idx = find_elbow(bs_slice)
            abs_idx = bs_slice.index[idx]
            return float(actuator_data.loc[abs_idx]), abs_idx
        return None, None


    def find_a2(end_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds A2 (Actuator Elbow before end_idx)."""
        ac_slice = actuator_data.loc[:end_idx]
        _, idx = find_elbow(ac_slice)
        abs_idx = ac_slice.index[idx]
        return float(actuator_data.loc[abs_idx]), abs_idx


    def find_a3() -> Tuple[Optional[float], Optional[int]]:
        """Finds A3 (Downstream Elbow)."""
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            ds_slice = downstream_data.loc[:middle_idx]
            _, idx = find_elbow(ds_slice)
            abs_idx = ds_slice.index[idx]
            return float(actuator_data.loc[abs_idx]), abs_idx
        return None, None


    def find_a4() -> Tuple[Optional[float], Optional[int]]:
        """Finds A4 (Downstream Knee)."""
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            ds_slice = downstream_data.loc[:middle_idx]
            _, idx = find_knee(ds_slice)
            abs_idx = ds_slice.index[idx]
            return float(actuator_data.loc[abs_idx]), abs_idx
        return None, None


    def find_a5(start_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds A5 (Actuator Elbow after start_idx)."""
        ac_slice = actuator_data.loc[start_idx:middle_idx]
        _, idx = find_elbow(ac_slice)
        abs_idx = ac_slice.index[idx]
        return float(actuator_data.loc[abs_idx]), abs_idx


    def find_r1(end_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds R1 (Actuator Elbow in return stroke)."""
        ac_slice = actuator_data.loc[middle_idx:end_idx]
        _, idx = find_elbow(ac_slice)
        abs_idx = ac_slice.index[idx]
        return float(actuator_data.loc[abs_idx]), abs_idx


    def find_r2() -> Tuple[Optional[float], Optional[int]]:
        """Finds R2 (Downstream Knee in return stroke)."""
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            ds_slice = downstream_data.loc[middle_idx:]
            _, idx = find_knee(ds_slice)
            abs_idx = ds_slice.index[idx]
            return float(actuator_data.loc[abs_idx]), abs_idx
        return None, None


    def find_r3() -> Tuple[Optional[float], Optional[int]]:
        """Finds R3 (Downstream Elbow in return stroke)."""
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            ds_slice = downstream_data.loc[middle_idx:]
            _, idx = find_elbow(ds_slice)
            abs_idx = ds_slice.index[idx]
            return float(actuator_data.loc[abs_idx]), abs_idx
        return None, None


    def find_r4(start_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds R4 (Actuator Knee after start_idx)."""
        ac_slice = actuator_data.loc[start_idx:]
        _, idx = find_knee(ac_slice)
        abs_idx = ac_slice.index[idx]
        return float(actuator_data.loc[abs_idx]), abs_idx

    def find_bto() -> Tuple[float, int]:
        mask_array = close_data.astype(bool).to_numpy()
        end_rel_idx = np.where(~mask_array)[0][0] if (~mask_array).any() else len(mask_array)
        tq_slice = torque_data.iloc[:end_rel_idx]
        val = float(tq_slice.min().round(1))
        abs_idx = tq_slice.idxmin()
        return val, abs_idx

    def find_rpo(start_idx: int) -> Tuple[Optional[float], Optional[int]]:
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            ds_slice = downstream_data.iloc[:middle_idx]
            _, rel_end_idx = find_elbow(ds_slice)
            end_abs_idx = ds_slice.index[rel_end_idx]
            tq_slice1 = torque_data.loc[start_idx:end_abs_idx]
            peak_abs_idx = tq_slice1.idxmax()
            tq_slice2 = torque_data.loc[start_idx:peak_abs_idx]
            _, rel_start_idx = find_knee(tq_slice2)
            start_abs_idx = tq_slice2.index[rel_start_idx]
            final_slice = torque_data.loc[start_abs_idx:end_abs_idx]
            val = float(final_slice.min().round(1))
            abs_idx = final_slice.idxmin()
            return val, abs_idx
        return None, None

    def find_rno() -> Tuple[float, int]:
        ds_slice = downstream_data.iloc[:middle_idx]
        _, rel_start_idx = find_knee(ds_slice)
        start_abs_idx = ds_slice.index[rel_start_idx]
        tq_slice1 = torque_data.loc[start_abs_idx:end_idx]
        _, rel_end_idx = find_knee(tq_slice1)
        end_abs_idx = tq_slice1.index[rel_end_idx]
        if not (channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]):
            ds_slice_alt = downstream_data.iloc[:middle_idx]
            _, rel_alt_idx = find_elbow(ds_slice_alt)
            alt_abs_idx = ds_slice_alt.index[rel_alt_idx]
            tq_slice2 = torque_data.loc[start_abs_idx:alt_abs_idx]
            peak_abs_idx = tq_slice2.idxmax()
            tq_slice3 = torque_data.loc[start_abs_idx:peak_abs_idx]
            _, rel_new_idx = find_knee(tq_slice3)
            start_abs_idx = tq_slice3.index[rel_new_idx]
        final_slice = torque_data.loc[start_abs_idx:end_abs_idx]
        val = float(final_slice.min().round(1))
        abs_idx = final_slice.idxmin()
        return val, abs_idx

    def find_jto() -> Tuple[float, int]:
        val = float(torque_data.min().round(1))
        abs_idx = torque_data.idxmin()
        return val, abs_idx

    def find_btc() -> Tuple[float, int]:
        mask_array = open_data.astype(bool).to_numpy()
        start_rel_idx = np.where(mask_array)[0][0] if mask_array.any() else 0
        end_rel_idx = np.where(~mask_array)[0][0] if (~mask_array).any() else len(mask_array)
        tq_slice = torque_data.iloc[start_rel_idx:end_rel_idx]
        val = float(tq_slice.max().round(1))
        abs_idx = tq_slice.idxmax()
        return val, abs_idx

    def find_rnc(start_idx: int, end_idx: int) -> Tuple[float, int]:
        ds_slice = downstream_data.iloc[middle_idx:]
        _, rel_mid_idx = find_knee(ds_slice)
        mid_abs_idx = ds_slice.index[rel_mid_idx]
        tq_slice1 = torque_data.loc[start_idx:mid_abs_idx]
        _, rel_start_idx = find_elbow(tq_slice1)
        start_abs_idx = tq_slice1.index[rel_start_idx]
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            _, rel_end_idx = find_knee(ds_slice)
            end_abs_idx = ds_slice.index[rel_end_idx]
        else:
            _, rel_alt_idx = find_elbow(ds_slice)
            alt_abs_idx = ds_slice.index[rel_alt_idx]
            tq_slice2 = torque_data.loc[alt_abs_idx:]
            min_idx_idx = tq_slice2.idxmin()
            tq_slice3 = torque_data.loc[min_idx_idx:end_idx]
            _, rel_new_idx = find_elbow(tq_slice3)
            end_abs_idx = tq_slice3.index[rel_new_idx]
        final_slice = torque_data.loc[start_abs_idx:end_abs_idx]
        val = float(final_slice.max().round(1))
        abs_idx = final_slice.idxmax()
        return val, abs_idx

    def find_rpc(start_idx: int) -> Tuple[Optional[float], Optional[int]]:
        if channels_to_record.at["Upstream", 1] or channels_to_record.at["Downstream", 1]:
            ds_slice = downstream_data.iloc[middle_idx:]
            _, rel_start_idx = find_elbow(ds_slice)
            start_abs_idx = ds_slice.index[rel_start_idx]
            tq_slice1 = torque_data.loc[start_abs_idx:]
            min_idx_idx = tq_slice1.idxmin()
            tq_slice2 = torque_data.loc[min_idx_idx:end_idx]
            _, rel_end_idx = find_elbow(tq_slice2)
            end_abs_idx = tq_slice2.index[rel_end_idx]
            final_slice = torque_data.loc[start_abs_idx:end_abs_idx]
            val = float(final_slice.max().round(1))
            abs_idx = final_slice.idxmax()
            return val, abs_idx
        return None, None

    def find_jtc() -> Tuple[float, int]:
        val = float(torque_data.max().round(1))
        abs_idx = torque_data.idxmax()
        return val, abs_idx

    # Main loop
    df_cycle_breakpoints, total_cycles = find_cycle_breakpoints(raw_data)
    torque_signature_values: List[Dict[str, Any]] = []
    torque_signature_indices: List[Dict[str, Any]] = []
    actuator_signature_values: List[Dict[str, Any]] = []
    actuator_signature_indices: List[Dict[str, Any]] = []
    for cycle, start_idx, one_third_idx, middle_idx, two_third_idx, end_idx in df_cycle_breakpoints.itertuples(index=False, name=None):
        backseat_data   = raw_data.loc[start_idx:end_idx, "Backseat"]
        actuator_data   = raw_data.loc[start_idx:end_idx, "Actuator"]
        downstream_data = raw_data.loc[start_idx:end_idx, "Downstream"]
        torque_data     = raw_data.loc[start_idx:end_idx, "Torque"]
        open_data       = raw_data.loc[start_idx:end_idx, "Open"]
        close_data      = raw_data.loc[start_idx:end_idx, "Close"]

        if channels_to_record.at["Torque", 1] is True:
            bto, bto_idx = find_bto()
            rpo, rpo_idx = find_rpo(bto_idx)
            rno, rno_idx = find_rno()
            jto, jto_idx = find_jto()
            btc, btc_idx = find_btc()
            rpc, rpc_idx = find_rpc(btc_idx)
            jtc, jtc_idx = find_jtc()
            rnc, rnc_idx = find_rnc(btc_idx, jtc_idx)
            torque_signature_values.append({
                "Cycle": cycle,
                "BTO": bto,
                "RPO": rpo,
                "RNO": rno,
                "JTO": jto,
                "BTC": btc,
                "RPC": rpc,
                "JTC": jtc,
                "RNC": rnc,
            })
            torque_signature_indices.append({
                "Cycle": cycle,
                "BTO_Index": bto_idx,
                "RPO_Index": rpo_idx,
                "RNO_Index": rno_idx,
                "JTO_Index": jto_idx,
                "BTC_Index": btc_idx,
                "RPC_Index": rpc_idx,
                "JTC_Index": jtc_idx,
                "RNC_Index": rnc_idx,
            })
        else:
            a1, a1_idx = find_a1()
            a3, a3_idx = find_a3()
            a4, a4_idx = find_a4()
            a2, a2_idx = find_a2(a3_idx if a3_idx is not None else one_third_idx)
            a5, a5_idx = find_a5(a4_idx if a4_idx is not None else a2_idx)
            r2, r2_idx = find_r2()
            r1, r1_idx = find_r1(r2_idx if r2_idx is not None else end_idx)
            r3, r3_idx = find_r3()
            r4, r4_idx = find_r4(r3_idx if r3_idx is not None else r1_idx)
            actuator_signature_values.append({
                "Cycle": cycle,
                "A1": a1,
                "A2": a2,
                "A3": a3,
                "A4": a4,
                "A5": a5,
                "R1": r1,
                "R2": r2,
                "R3": r3,
                "R4": r4,
            })
            actuator_signature_indices.append({
                "Cycle": cycle,
                "A1_Index": a1_idx,
                "A2_Index": a2_idx,
                "A3_Index": a3_idx,
                "A4_Index": a4_idx,
                "A5_Index": a5_idx,
                "R1_Index": r1_idx,
                "R2_Index": r2_idx,
                "R3_Index": r3_idx,
                "R4_Index": r4_idx,
            })

    return (
        pd.DataFrame.from_records(torque_signature_values), 
        pd.DataFrame.from_records(torque_signature_indices), 
        pd.DataFrame.from_records(actuator_signature_values), 
        pd.DataFrame.from_records(actuator_signature_indices),
    )