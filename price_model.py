import os

import matplotlib.pyplot as plt
import numpy as np


def get_price_series():
    """
    直接在代码中给出 24 小时的 Emergency 与 FIT 价格序列。
    """
    emergency = np.array(
        [
            1.6,
            1.56,
            1.52,
            1.5,
            1.5,
            1.56,
            1.7,
            2.4,
            2.6,
            2.8,
            2.4,
            2.0,
            2.0,
            1.9,
            1.9,
            2.0,
            2.1,
            3.0,
            3.2,
            3.2,
            3.0,
            2.6,
            1.8,
            1.7,
        ]
    )

    fit = np.array(
        [
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]
    )

    assert len(emergency) == 24 and len(fit) == 24, "价格序列必须为 24 个小时。"
    return emergency, fit


def plot_dynamic_price(emergency, fit, save_path: str = "./Dataset/dynamic_price.png") -> None:
    """
    按 IEEE journal 风格绘制动态价格图（Emergency & FIT）。
    使用双 y 轴：
        左轴：Emergency price
        右轴：FIT price
    """
    # ============================
    # 图形风格设置（IEEE 常用规范）
    # ============================
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.unicode_minus"] = False

    # x 轴：时间索引（0~23 时，共 24 个小时）
    n = len(emergency)
    x = np.arange(0, n)

    # 创建图形（接近 IEEE 单栏图宽度）
    fig, ax = plt.subplots(figsize=(3.5, 2.6))  # 单栏宽度常用 3.5 inch 左右

    # 条形宽度与位置（并排柱状图）
    width = 0.4
    ax.bar(
        x - width / 2,
        emergency,
        width=width,
        color="#d62728",  # 红色
        edgecolor="black",
        linewidth=0.6,
        label="Emergency",
    )
    ax.bar(
        x + width / 2,
        fit,
        width=width,
        color="#1f77b4",  # 蓝色
        edgecolor="black",
        linewidth=0.6,
        label="FiT",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Price ($/kWh)")

    # 网格（只在主轴上显示），细虚线
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    # 轴刻度样式
    ax.tick_params(direction="in", length=3, width=0.8)

    # x 轴刻度：稀疏显示为具体时间标签（00:00, 04:00, ..., 23:00）
    tick_positions = np.array([0, 4, 8, 12, 16, 20, 23])
    tick_positions = tick_positions[tick_positions < n]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "23:00"][: len(tick_positions)])

    # 图例放在右下角，无边框
    leg = ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"✅ 动态价格图已保存至: {save_path}")


if __name__ == "__main__":
    emergency_series, fit_series = get_price_series()
    plot_dynamic_price(emergency_series, fit_series)


