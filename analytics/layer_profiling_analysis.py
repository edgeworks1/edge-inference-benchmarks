from pathlib import Path
import pandas as pd

CSV_PATH = Path("layer_profiling.csv")


def load_and_prepare_data(csv_file: Path) -> pd.DataFrame:
    """Lädt die CSV und berechnet grundlegende Metriken auf Layerebene."""
    if not csv_file.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_file}")

    df = pd.read_csv(csv_file)
    df["speedup_cycles"] = df["cycles_fp32"] / df["cycles_int8"]
    df["saved_time_us"] = df["time_us_fp32"] - df["time_us_int8"]
    return df


def print_overall_summary(df: pd.DataFrame) -> tuple[float, float]:
    """Gibt die globale Übersicht zu Latenz und Gesamtspeedup aus."""
    total_time_fp32 = df["time_us_fp32"].sum()
    total_time_int8 = df["time_us_int8"].sum()

    print("=" * 90)
    print("1. GESAMTKONVERTIERUNG (HIGH-LEVEL OVERVIEW)")
    print("=" * 90)
    print(f"FP32 Gesamtlatenz : {total_time_fp32:,} us ({total_time_fp32 / 1000:.2f} ms)")
    print(f"INT8 Gesamtlatenz : {total_time_int8:,} us ({total_time_int8 / 1000:.2f} ms)")
    print(f"Gesamter Speedup  : {total_time_fp32 / total_time_int8:.2f}x schneller\n")

    return total_time_fp32, total_time_int8


def print_operator_summary(
    df: pd.DataFrame, total_time_fp32: float, total_time_int8: float
) -> None:
    """Aggregiert Laufzeiten und Zyklen nach Operator-Typ."""
    print("=" * 90)
    print("2. AGGREGATION NACH OPERATORTYP (Kumulierte Zeiten & Anteile)")
    print("=" * 90)

    op_summary = (
        df.groupby("op_type")
        .agg(
            count=("idx", "count"),
            time_fp32=("time_us_fp32", "sum"),
            time_int8=("time_us_int8", "sum"),
            cycles_fp32=("cycles_fp32", "sum"),
            cycles_int8=("cycles_int8", "sum"),
        )
        .reset_index()
    )

    op_summary["pct_fp32"] = (
        op_summary["time_fp32"] / total_time_fp32 * 100
    ).round(2)
    op_summary["pct_int8"] = (
        op_summary["time_int8"] / total_time_int8 * 100
    ).round(2)
    op_summary["speedup"] = (
        op_summary["cycles_fp32"] / op_summary["cycles_int8"]
    ).round(2)

    op_summary = op_summary.sort_values(by="pct_int8", ascending=False)
    cols = [
        "op_type",
        "count",
        "time_fp32",
        "pct_fp32",
        "time_int8",
        "pct_int8",
        "speedup",
    ]
    print(op_summary[cols].to_string(index=False))


def print_top_bottlenecks(df: pd.DataFrame, top_n: int = 10) -> None:
    """Zeigt die größten Zeitfresser vor und nach der Quantisierung."""
    print("\n" + "=" * 90)
    print(f"3. BOTTLENECK-VERSCHIEBUNG: TOP {top_n} ZEITFRESSER")
    print("=" * 90)

    print(f"\n--- Top {top_n} FP32 Layer ---")
    top_fp32 = df.nlargest(top_n, "time_us_fp32")[
        ["idx", "op_type", "cycles_fp32", "time_us_fp32", "pct_total_fp32"]
    ]
    print(top_fp32.to_string(index=False))

    print(f"\n--- Top {top_n} INT8 Layer (Neue Engpässe!) ---")
    top_int8 = df.nlargest(top_n, "time_us_int8")[
        [
            "idx",
            "op_type",
            "cycles_int8",
            "time_us_int8",
            "pct_total_int8",
            "speedup_cycles",
        ]
    ].copy()
    top_int8["speedup_cycles"] = top_int8["speedup_cycles"].round(2)
    print(top_int8.to_string(index=False))


def print_low_speedup_layers(df: pd.DataFrame, top_n: int = 10) -> None:
    """Findet Layer mit geringer CMSIS-NN SIMD-Beschleunigung."""
    print("\n" + "=" * 90)
    print(f"4. OPERATOREN MIT GERINGSTEM SPEEDUP (Top {top_n})")
    print("=" * 90)

    slowest = df.nsmallest(top_n, "speedup_cycles")[
        [
            "idx",
            "op_type",
            "time_us_fp32",
            "time_us_int8",
            "speedup_cycles",
            "pct_total_int8",
        ]
    ].copy()
    slowest["speedup_cycles"] = slowest_speedup_rounded = slowest[
        "speedup_cycles"
    ].round(2)
    print(slowest.to_string(index=False))


def print_slower_in_int8(df: pd.DataFrame) -> None:
    """Identifiziert Layer, die in INT8 langsamer laufen als in FP32."""
    print("\n" + "=" * 90)
    print("5. REGRESSION-CHECK: LAYER, DIE IN INT8 LANGSAMER SIND (Speedup < 1.0)")
    print("=" * 90)

    # Entweder über Zyklen (präziser) oder Mikrosekunden prüfen
    slower_df = df[df["cycles_int8"] > df["cycles_fp32"]].copy()

    if slower_df.empty:
        print("Keine Regressionen gefunden: Kein Layer ist in INT8 langsamer als in FP32.")
        
        # Hinweis auf Layer mit exakt identischer Laufzeit (z. B. PAD)
        neutral_df = df[df["cycles_int8"] == df["cycles_fp32"]]
        if not neutral_df.empty:
            print(f"Hinweis: {len(neutral_df)} Layer haben identische Zyklen (Speedup = 1.00x):")
            print(
                neutral_df[
                    ["idx", "op_type", "cycles_fp32", "cycles_int8"]
                ].to_string(index=False)
            )
    else:
        slower_df["slowdown_factor"] = (
            slower_df["cycles_int8"] / slower_df["cycles_fp32"]
        ).round(2)
        slower_df["extra_time_us"] = (
            slower_df["time_us_int8"] - slower_df["time_us_fp32"]
        )
        print(f"ACHTUNG: {len(slower_df)} Layer sind in INT8 langsamer:")
        cols = [
            "idx",
            "op_type",
            "cycles_fp32",
            "cycles_int8",
            "time_us_fp32",
            "time_us_int8",
            "slowdown_factor",
            "extra_time_us",
        ]
        print(slower_df[cols].to_string(index=False))


def run_analysis(csv_file: Path):
    df = load_and_prepare_data(csv_file)
    total_fp32, total_int8 = print_overall_summary(df)
    print_operator_summary(df, total_fp32, total_int8)
    print_top_bottlenecks(df)
    print_low_speedup_layers(df)
    print_slower_in_int8(df)


if __name__ == "__main__":
    run_analysis(CSV_PATH)