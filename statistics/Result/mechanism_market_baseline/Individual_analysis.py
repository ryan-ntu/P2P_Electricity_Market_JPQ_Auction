"""
对 `mechanism_market_baseline` 目录下四种机制的日志 CSV 进行对比。
取每个文件末尾的 5 条记录，计算各微网在 reward、紧急购电、
上网售电和储能水平的平均值，输出汇总表便于比较。
注意：reward 结果除以 10；储能按实际容量换算，分别乘以
Grid1/2/3/4 的系数 8/15/15/30。
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

# 目录与文件配置
BASE_DIR = Path(__file__).resolve().parent
MECHANISM_FILES: Dict[str, str] = {
    "MAPPO_simple_training_log_individual_critic.csv": "simple",
    "MAPPO_mrda_training_log_individual_critic.csv": "mrda",
    "MAPPO_msmrda_training_log_individual_critic.csv": "msmrda",
    "MAPPO_vda_training_log_individual_critic.csv": "vda",
}

# 需要关注的指标列名映射：CSV 字段名 -> 输出列名
METRIC_COLUMNS: Dict[str, str] = {
    "reward": "reward",
    "emergency_purchase": "Emergency bought",
    "feed_in_power": "Feed In Sold",
    "storage_level": "Storage",
}

# 将 agent 编号映射到 Grid 名称，便于阅读
AGENT_LABELS: Dict[str, str] = {
    "0": "Grid 1",
    "1": "Grid 2",
    "2": "Grid 3",
    "3": "Grid 4",
}

# 储能换算系数（实际容量），用于将百分比转为实际值
STORAGE_FACTORS: Dict[str, float] = {
    "0": 8,
    "1": 15,
    "2": 15,
    "3": 30,
}

REWARD_DIVISOR = 10


def get_agent_ids(columns: List[str]) -> List[str]:
    """从列名解析出现过的代理编号。"""
    return sorted({col.split("_")[1] for col in columns if col.startswith("agent_")})


def summarize_last_points(file_path: Path, last_n: int = 5) -> pd.DataFrame:
    """读取文件，统计末尾 last_n 行每个代理的均值。"""
    df = pd.read_csv(file_path)
    tail_df = df.tail(last_n)
    agents = get_agent_ids(tail_df.columns.tolist())

    rows = []
    for agent in agents:
        row = {
            "agent": agent,
            "grid": AGENT_LABELS.get(agent, f"Grid {agent}"),
        }
        for metric_key, label in METRIC_COLUMNS.items():
            col = f"agent_{agent}_{metric_key}"
            if col in tail_df.columns:
                series = tail_df[col]
                if metric_key == "reward":
                    series = series / REWARD_DIVISOR
                elif metric_key == "storage_level":
                    factor = STORAGE_FACTORS.get(agent, 1.0)
                    series = series * factor
                row[label] = series.mean()
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary() -> pd.DataFrame:
    """构建所有机制的汇总表。"""
    all_rows = []
    for filename, mechanism in MECHANISM_FILES.items():
        file_path = BASE_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"缺少文件: {file_path}")

        summary = summarize_last_points(file_path)
        summary["mechanism"] = mechanism
        all_rows.append(summary)

    return pd.concat(all_rows, ignore_index=True)


def pivot_by_mechanism(df: pd.DataFrame) -> pd.DataFrame:
    """将汇总表透视，便于横向对比。"""
    value_cols = list(METRIC_COLUMNS.values())
    pivoted = df.pivot_table(
        index=["grid", "agent"], columns="mechanism", values=value_cols, aggfunc="first"
    )
    # pivot_table 产生多级列索引，这里拍平为“机制_指标”
    pivoted.columns = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
    return pivoted.reset_index()


def main() -> None:
    summary = build_summary()
    pivoted = pivot_by_mechanism(summary)

    # 控制台打印
    print("末 5 条记录均值：")
    print(summary)
    print("\n按机制对比：")
    print(pivoted)

    # 写入 txt 文件
    output_path = BASE_DIR / "Individual_analysis_output.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("末 5 条记录均值：\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n按机制对比：\n")
        f.write(pivoted.to_string(index=False))
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

