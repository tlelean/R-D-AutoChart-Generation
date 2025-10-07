import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import re
import warnings

def find_cycle_breakpoints(raw_data, channels_to_record, channel_map: dict[str, str]):
    cycle_count_data = raw_data[channel_map["Cycle Count"]]
    total_cycle_count = int(cycle_count_data.max())

    cycle_ranges = []

    for i in range(1, total_cycle_count + 1):
        matching = cycle_count_data[cycle_count_data == i]
        if matching.empty:
            continue

        start_idx = matching.index[0]
        end_idx = matching.index[-1]
        n_points = len(matching)

        if channels_to_record.at[channel_map["Torque"], 1]:
            one_quarter_idx = matching.index[n_points // 4]
            middle_idx = matching.index[n_points // 2]
            three_quarter_idx = matching.index[(3 * n_points) // 4]
        elif channels_to_record.at[channel_map["Actuator"], 1]:
            actuator_series = raw_data.loc[matching.index, channel_map["Actuator"]]
            if actuator_series.empty:
                one_quarter_idx = matching.index[n_points // 4]
                middle_idx = matching.index[n_points // 2]
                three_quarter_idx = matching.index[(3 * n_points) // 4]
            else:
                middle_idx = actuator_series.idxmax()
                middle_pos = list(matching.index).index(middle_idx)
                start_pos = 0
                end_pos = n_points - 1
                one_quarter_idx = matching.index[(start_pos + middle_pos) // 2]
                three_quarter_idx = matching.index[(middle_pos + end_pos) // 2]
        else:
            one_quarter_idx = matching.index[n_points // 4]
            middle_idx = matching.index[n_points // 2]
            three_quarter_idx = matching.index[(3 * n_points) // 4]

        cycle_ranges.append((
            i,
            start_idx,
            one_quarter_idx,
            middle_idx,
            three_quarter_idx,
            end_idx
        ))

    df_ranges = pd.DataFrame(cycle_ranges, columns=[
        "Cycle",
        "Start Index",
        "One-Quarter Index",
        "Middle Index",
        "Three-Quarter Index",
        "End Index",
    ])
    return df_ranges, total_cycle_count

def signed_distances_to_baseline(y: pd.Series) -> np.ndarray:
    if len(y) < 2: 
        return np.zeros(len(y))
    else:
        x = np.arange(len(y))
        m = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
        b = y.iloc[0]
        # distance of each (x, y) to line y = m x + b
        return ((m * x - y + b) / np.hypot(m, -1))

# Find the point in the given data that is furthest **above** the baseline.
def find_above(data):
    sd = signed_distances_to_baseline(data)
    idx = np.argmax(sd)
    return data.iloc[idx], idx

# Find the point in the given data that is furthest **below** the baseline.
def find_below(data):
    sd = signed_distances_to_baseline(data)
    idx = np.argmin(sd)
    return data.iloc[idx], idx

def locate_calibration_points(cleaned_data, additional_info):

    # Ensure Datetime is a datetime type
    calibration_indices = pd.DataFrame(index=range(2), columns=[0, 1, 2, 3, 4, 5])
    calibration_values = pd.DataFrame(index=range(2), columns=[0, 1, 2, 3, 4, 5])
    calibration_times = pd.DataFrame(index=range(2), columns=[0, 1, 2, 3, 4, 5])
    date_time_index = cleaned_data.set_index('Datetime')

    for col in range(1, len(additional_info.columns)):
        calibration_times.at[0, col] = pd.to_datetime(
            additional_info.at[1, col],
            format="%d/%m/%Y %H:%M:%S.%f",
            errors="coerce",
            dayfirst=True,
        )

        calibration_times.at[1, col] = calibration_times.at[0, col] + pd.Timedelta(seconds=10)
        calibration_indices.at[0, col] = date_time_index.index.get_indexer(
            [calibration_times.iloc[0, col]],
            method="nearest",
        )[0]
        calibration_indices.at[1, col] = date_time_index.index.get_indexer(
            [calibration_times.iloc[1, col]],
            method="nearest",
        )[0]
        calibration_values.at[0, col] = cleaned_data.iloc[calibration_indices.iloc[0, col]][additional_info.at[1, 0]]
        calibration_values.at[1, col] = cleaned_data.iloc[calibration_indices.iloc[1, col]][additional_info.at[1, 0]]
    return calibration_indices, calibration_values

def calculate_succesful_calibration(cleaned_data, calibration_indices, additional_info):
    """Calculate calibration table along with unrounded measurement data."""

    display_table = pd.DataFrame()
    expected_counts = {}
    intercepts: List[float] = []

    def _to_float(value) -> float:
        if isinstance(value, str):
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
            if match:
                return float(match.group())
        return float(value)

    start_value = _to_float(additional_info.at[0, 0])
    first_value = _to_float(additional_info.at[0, 1])
    last_value = _to_float(additional_info.at[0, 5])

    if additional_info.at[0, 1] == "-10000":
        slope = (last_value - first_value) / (start_value * 2)
        step = start_value / 2
    else:
        slope = (last_value - first_value) / start_value
        step = start_value / (len(calibration_indices.columns) - 2)

    column_range = list(range(1, len(calibration_indices.columns)))

    for col in column_range:
        if additional_info.at[0, 0] == "7812500.0":
            if additional_info.at[0, 1] == "-10000":
                expected_value = (col - 1) * step - start_value
            else:
                expected_value = (col - 1) * step
        elif additional_info.at[0, 0] == "15700":
            expected_value = ((col - 1) * step) - 2000
        else:
            expected_value = (col - 1) * step

        expected_counts[col] = expected_value
        intercepts.append(_to_float(additional_info.iat[0, col]) - (slope * expected_value))

    intercept_value = float(np.mean(intercepts)) if intercepts else 0.0

    counts_series = pd.Series(dtype=float)
    expected_series = pd.Series(dtype=float)
    abs_error_series = pd.Series(dtype=float)

    channel_name = additional_info.at[1, 0]

    for i, col in enumerate(column_range):
        start_idx = calibration_indices.iloc[0, col]
        end_idx = calibration_indices.iloc[1, col]
        applied = _to_float(additional_info.at[0, col])

        counts = cleaned_data.loc[start_idx:end_idx, channel_name].mean()

        print(f"Calibration Point {i+1}: Slope={slope}, Counts={counts}, Intercept={intercept_value}")

        converted = (slope * counts) + intercept_value
        error = applied - converted

        display_col = i + 1

        counts_series.loc[display_col] = counts
        expected_series.loc[display_col] = expected_counts.get(col, np.nan)
        abs_error_series.loc[display_col] = abs(error)

        counts_display = int(round(counts))
        converted_display = round(converted, 3)
        error_display = round(abs(error), 2)

        display_table.loc[0, [display_col]] = applied
        display_table.loc[1, [display_col]] = counts_display
        display_table.loc[2, [display_col]] = converted_display
        display_table.loc[3, [display_col]] = error_display

    print(additional_info)

    if additional_info.at[0, 0] == "7812500.0" or additional_info.at[0, 0] == "7812500":
        if additional_info.at[0, 1] == "4000":
            display_table.index = [
                'Applied (µA)',
                'Counts (avg)',
                'Converted (µA)',
                'Abs Error (µA) - ±3.6 µA',
            ]
        else:
            display_table.index = [
                'Applied (mV)',
                'Counts (avg)',
                'Converted (mV)',
                'Abs Error (mV) - ±1.0 mV',
            ]
    elif additional_info.at[0, 0] == "15700":
        display_table.index = [
            'Applied (mV)',
            'Counts (avg)',
            'Converted (mV)',
            'Abs Error (mV) - ±0.12 mV',
        ]

    display_table.insert(0, "0", display_table.index)

    return display_table, counts_series, expected_series, abs_error_series

def calculate_calibration_regression(counts: pd.Series, expected_counts: pd.Series) -> pd.Series:
    """Return polynomial coefficients mapping counts to expected counts."""

    labels = ["S3", "S2", "S1", "S0"]
    if counts is None or expected_counts is None:
        return pd.Series([np.nan] * 4, index=labels, dtype=float)

    counts_series = pd.to_numeric(pd.Series(counts), errors="coerce")
    expected_series = pd.to_numeric(pd.Series(expected_counts), errors="coerce")
    mask = ~(counts_series.isna() | expected_series.isna())

    valid_counts = counts_series[mask]
    valid_expected = expected_series[mask]

    if len(valid_counts) < 2:
        return pd.Series([np.nan] * 4, index=labels, dtype=float)

    degree = min(3, len(valid_counts) - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefficients = np.polyfit(valid_counts, valid_expected, deg=degree)

    padded = np.full(4, np.nan)
    padded[-(degree + 1):] = coefficients
    return pd.Series(padded, index=labels, dtype=float)

def locate_key_time_rows(cleaned_data, additional_info):
    """Return indices of key time points closest to provided timestamps."""
    if additional_info.empty:
        return None, None
    else:
        holds_indices = pd.DataFrame(index=range(1), columns=[1, 2, 3])
        holds_values = additional_info.copy()
        date_time_index = cleaned_data.set_index('Datetime')
        for row in range(1, len(holds_values)):
            holds_values.at[row, 1] = pd.to_datetime(
                holds_values.at[row, 1],
                format="%d/%m/%Y %H:%M:%S.%f",
                errors="coerce",
                dayfirst=True,
            )
            if pd.isna(holds_values.iloc[row, 1]):
                continue
            holds_indices.at[0, row] = date_time_index.index.get_indexer(
                [holds_values.iloc[row, 1]],
                method="nearest",
            )[0]
            rowpos = holds_indices.at[0, row]
            colpos_value = cleaned_data.columns.get_loc(holds_values.at[0, 2])
            colpos_temp  = cleaned_data.columns.get_loc("Body Temperature")
            holds_values.at[row, 2] = int(cleaned_data.iloc[rowpos, colpos_value])
            holds_values.at[row, 3] = cleaned_data.iloc[rowpos, colpos_temp]
        holds_indices.columns = ['SOS_Index', 'SOH_Index', 'EOH_Index']
        holds_values.columns = holds_values.iloc[0]
        holds_values.loc[1:, "Datetime"] = (
            pd.to_datetime(
                holds_values.loc[1:, "Datetime"],
                format="%d/%m/%Y %H:%M:%S.%f",
                errors="coerce",
                dayfirst=True,
            )
            .dt.strftime("%d/%m/%Y %H:%M:%S.%f")
            .str.slice(0, 19)   # trims to 3 decimals (.905 instead of .905000)
        )
        holds_values.iat[0, 2] = str(holds_values.iat[0, 2]) + " (psi)"
        holds_values.iat[0, 3] = str(holds_values.iat[0, 3]) + " (°C)"
        holds_values = holds_values.fillna('')

        return holds_indices, holds_values
    
def locate_bto_btc_rows(raw_data, additional_info, channels_to_record, channel_map: dict[str, str]):
    if (
        additional_info.iloc[0, :3].astype(str).tolist() == ["Cycle", "BTO", "BTC"]
        and channels_to_record.at[channel_map["Torque"], 1]
    ):
        breakout_values: List[Dict[str, Any]] = []
        breakout_indices: List[Dict[str, Any]] = []

        torque_series = raw_data[channel_map["Torque"]]

        # Get precomputed cycle boundaries
        indices_ranges, _ = find_cycle_breakpoints(raw_data, channels_to_record, channel_map)

        for cycle, start_idx, one_quarter, middle_idx, three_quarter, end_idx in indices_ranges.itertuples(index=False, name=None):
            # 1st third for BTO
            torque_first = torque_series.loc[start_idx:one_quarter]
            bto = torque_first.min().round(2)
            bto_index = torque_first.idxmin()

            # 3rd third for BTC
            torque_third = torque_series.loc[middle_idx:three_quarter]
            btc = torque_third.max().round(2)
            btc_index = torque_third.idxmax()

            # Store values
            breakout_values.append({
                "Cycle": cycle,
                "BTO (lb·ft)": bto,
                "BTC (lb·ft)": btc,
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
    raw_data: pd.DataFrame,
    channel_map: dict[str, str],
) -> pd.DataFrame:
    """
    Processes raw_data to find signature key points for each cycle.
    Returns a DataFrame with one row per cycle and columns for each key point and its index.
    """
    def find_a1() -> Tuple[Optional[float], Optional[int]]:
        """Finds A1 (Backseat Elbow)."""
        if channels_to_record.at[channel_map["Backseat"], 1]:
            bs_slice = backseat_data.loc[:middle_idx]
            _, idx = find_above(bs_slice)
            abs_idx = bs_slice.index[idx]
            return round(actuator_data.loc[abs_idx], 0), abs_idx
        return None, None

    def find_a2(end_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds A2 (Actuator Elbow before end_idx)."""
        ac_slice = actuator_data.loc[:end_idx]
        _, idx = find_below(ac_slice)
        abs_idx = ac_slice.index[idx]
        return round(actuator_data.loc[abs_idx], 0), abs_idx

    def find_a3() -> Tuple[Optional[float], Optional[int]]:
        """Finds A3 (Downstream Elbow)."""
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[:middle_idx]
            _, idx = find_above(ds_slice)
            abs_idx = ds_slice.index[idx]
            return round(actuator_data.loc[abs_idx], 0), abs_idx
        return None, None

    def find_a4() -> Tuple[Optional[float], Optional[int]]:
        """Finds A4 (Downstream Knee)."""
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[:middle_idx]
            _, idx = find_below(ds_slice)
            abs_idx = ds_slice.index[idx]
            return round(actuator_data.loc[abs_idx], 0), abs_idx
        return None, None

    def find_a5(start_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds A5 (Actuator Elbow after start_idx)."""
        ac_slice = actuator_data.loc[start_idx:middle_idx]
        _, idx = find_above(ac_slice)
        abs_idx = ac_slice.index[idx]
        return round(actuator_data.loc[abs_idx], 0), abs_idx

    def find_r1(end_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds R1 (Actuator Elbow in return stroke)."""
        ac_slice = actuator_data.loc[middle_idx:end_idx]
        _, idx = find_above(ac_slice)
        abs_idx = ac_slice.index[idx]
        return round(actuator_data.loc[abs_idx], 0), abs_idx

    def find_r2() -> Tuple[Optional[float], Optional[int]]:
        """Finds R2 (Downstream Knee in return stroke)."""
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[middle_idx:]
            _, idx = find_below(ds_slice)
            abs_idx = ds_slice.index[idx]
            return round(actuator_data.loc[abs_idx], 0), abs_idx
        return None, None

    def find_r3() -> Tuple[Optional[float], Optional[int]]:
        """Finds R3 (Downstream Elbow in return stroke)."""
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[middle_idx:]
            _, idx = find_above(ds_slice)
            abs_idx = ds_slice.index[idx]
            return round(actuator_data.loc[abs_idx], 0), abs_idx
        return None, None

    def find_r4(r3_idx: int, r1_idx: int) -> Tuple[Optional[float], Optional[int]]:
        """Finds R4 (Actuator Knee after start_idx)."""
        if r3_idx is None:
            recording_speed = datetime_data.loc[r1_idx+1] - datetime_data.loc[r1_idx]
            forty_s_in_rows = pd.Timedelta(seconds=40) / recording_speed
            end_idx = r1_idx + forty_s_in_rows
            ac_slice = actuator_data.loc[r1_idx:end_idx]
        else:
            ac_slice = actuator_data.loc[r3_idx:]
        _, idx = find_below(ac_slice)
        abs_idx = ac_slice.index[idx]
        return round(actuator_data.loc[abs_idx], 0), abs_idx

    def find_bto() -> Tuple[float, int]:
        false_indices = close_data.index[~close_data.astype(bool)]
        end_rel_idx = false_indices[0] if not false_indices.empty else close_data.index[-1]
        tq_slice = torque_data.loc[:end_rel_idx]
        val = round(tq_slice.min(), 2)
        abs_idx = tq_slice.idxmin()
        return val, abs_idx

    def find_rpo(start_idx: int) -> Tuple[Optional[float], Optional[int]]:
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[:middle_idx]
            _, end_idx = find_below(ds_slice)
            end_idx = ds_slice.index[end_idx]
            tq_slice1 = torque_data.loc[start_idx:end_idx]
            _, start_idx = find_above(tq_slice1)
            start_idx = tq_slice1.index[start_idx]
            final_slice = torque_data.loc[start_idx:end_idx]
            val = round(final_slice.min(), 2)
            abs_idx = final_slice.idxmin()
            return val, abs_idx
        return None, None

    def find_rno(start_idx, end_idx) -> Tuple[float, int]:
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[:middle_idx]
            _, start_idx = find_below(ds_slice)
            start_idx = ds_slice.index[start_idx]
        else:
            tq_slice = torque_data.loc[start_idx:one_quarter_idx]
            _, start_idx = find_below(tq_slice)
            start_idx = tq_slice.index[start_idx]

        tq_slice = torque_data.loc[one_quarter_idx:end_idx]
        _, end_idx = find_below(tq_slice)
        end_idx = tq_slice.index[end_idx]

        final_slice = torque_data.loc[start_idx:end_idx-1]
        val = round(final_slice.min(), 2)
        abs_idx = final_slice.idxmin()
        return val, abs_idx

    def find_jto() -> Tuple[float, int]:
        val = round(torque_data.min(), 2)
        abs_idx = torque_data.idxmin()
        return val, abs_idx

    def find_btc() -> Tuple[float, int]:
        # Convert to boolean and find first True/False indices
        true_indices = open_data.index[open_data.astype(bool)]
        end_rel_idx = true_indices[-1] if not true_indices.empty else close_data.index[-1]
        tq_slice = torque_data.loc[:end_rel_idx]
        val = round(tq_slice.max(), 2)
        abs_idx = tq_slice.idxmax()
        return val, abs_idx

    def find_rnc(start_idx: int, end_idx: int) -> Tuple[float, int]:
        tq_slice = torque_data.loc[start_idx:three_quarter_idx]
        _, start_idx = find_above(tq_slice)
        start_idx = tq_slice.index[start_idx]

        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[middle_idx:]
            _, end_idx = find_above(ds_slice)
            end_idx = ds_slice.index[end_idx]
        else:
            tq_slice = torque_data.loc[three_quarter_idx:end_idx]
            _, end_idx = find_above(tq_slice)
            end_idx = tq_slice.index[end_idx]

        final_slice = torque_data.loc[start_idx:end_idx-1]
        val = round(final_slice.max(), 2)
        abs_idx = final_slice.idxmax()
        return val, abs_idx

    def find_rpc(end_idx: int) -> Tuple[Optional[float], Optional[int]]:
        if channels_to_record.at[channel_map["Upstream"], 1] or channels_to_record.at[channel_map["Downstream"], 1]:
            ds_slice = downstream_data.loc[middle_idx:]
            _, start_idx = find_above(ds_slice)
            start_idx = ds_slice.index[start_idx]

            tq_slice1 = torque_data.loc[start_idx:end_idx]
            _, end_idx = find_above(tq_slice1)
            end_idx = tq_slice1.index[end_idx]

            final_slice = torque_data.loc[start_idx:end_idx]
            val = round(final_slice.max(), 2)
            abs_idx = final_slice.idxmax()
            return val, abs_idx
        return None, None

    def find_jtc() -> Tuple[float, int]:
        val = round(torque_data.max(), 2)
        abs_idx = torque_data.idxmax()
        return val, abs_idx

    # Main loop
    df_cycle_breakpoints, total_cycles = find_cycle_breakpoints(raw_data, channels_to_record, channel_map)
    torque_signature_values: List[Dict[str, Any]] = []
    torque_signature_indices: List[Dict[str, Any]] = []
    actuator_signature_values: List[Dict[str, Any]] = []
    actuator_signature_indices: List[Dict[str, Any]] = []
    for cycle, start_idx, one_quarter_idx, middle_idx, three_quarter_idx, end_idx in df_cycle_breakpoints.itertuples(index=False, name=None):
        datetime_data   = raw_data.loc[start_idx:end_idx, "Datetime"]
        backseat_data   = raw_data.loc[start_idx:end_idx, channel_map["Backseat"]]
        actuator_data   = raw_data.loc[start_idx:end_idx, channel_map["Actuator"]]
        downstream_data = raw_data.loc[start_idx:end_idx, channel_map["Downstream"]]
        torque_data     = raw_data.loc[start_idx:end_idx, channel_map["Torque"]]
        open_data       = raw_data.loc[start_idx:end_idx, channel_map["Open"]]
        close_data      = raw_data.loc[start_idx:end_idx, channel_map["Close"]]

        if channels_to_record.at[channel_map["Torque"], 1]:
            bto, bto_idx = find_bto()
            rpo, rpo_idx = find_rpo(bto_idx)
            jto, jto_idx = find_jto()
            rno, rno_idx = find_rno(rpo_idx if rpo_idx is not None else bto_idx, jto_idx)
            btc, btc_idx = find_btc()
            jtc, jtc_idx = find_jtc()
            rpc, rpc_idx = find_rpc(jtc_idx)
            rnc, rnc_idx = find_rnc(btc_idx, jtc_idx)
            torque_signature_values.append({
                "Cycle": cycle,
                "BTO": bto,
                "RPO": rpo,
                "RNO": rno,
                "JTO": jto,
                "BTC": btc,
                "RNC": rnc,
                "RPC": rpc,
                "JTC": jtc,
            })
            torque_signature_indices.append({
                "Cycle": cycle,
                "BTO_Index": bto_idx,
                "RPO_Index": rpo_idx,
                "RNO_Index": rno_idx,
                "JTO_Index": jto_idx,
                "BTC_Index": btc_idx,
                "RNC_Index": rnc_idx,
                "RPC_Index": rpc_idx,
                "JTC_Index": jtc_idx,        
            })
        else:
            a1, a1_idx = find_a1()
            a3, a3_idx = find_a3()
            a4, a4_idx = find_a4()
            a2, a2_idx = find_a2(a3_idx if a3_idx is not None else one_quarter_idx)
            a5, a5_idx = find_a5(a4_idx if a4_idx is not None else a2_idx)
            r2, r2_idx = find_r2()
            r1, r1_idx = find_r1(r2_idx if r2_idx is not None else end_idx)
            r3, r3_idx = find_r3()
            r4, r4_idx = find_r4(r3_idx, r1_idx)
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