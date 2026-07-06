import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 读取数据
# =========================
# 这两个 CSV 是你之前筛好的 non-overlap 版本
reduced = pd.read_csv('top_confusions_base.csv')
increased = pd.read_csv('top_confusions_full.csv')

# 如果你想自己重新读原始数据再筛选，可以改成：
# reduced = pd.read_csv('your_reduced.csv')
# increased = pd.read_csv('your_increased.csv')

df = pd.concat([reduced, increased], ignore_index=True)

# =========================
# 2. 画图参数
# =========================
fig, ax = plt.subplots(figsize=(12.8, 8.4))

# y 轴位置（从上到下）
y = np.arange(len(df))[::-1]

# 颜色设置
up_color = '#14b8a6'   # 蓝绿色：上升
down_color = '#d62728' # 红色：下降

# x 轴范围自动扩展
xmin = min(df['rate_in_true_base'].min(), df['rate_in_true_full'].min())
xmax = max(df['rate_in_true_base'].max(), df['rate_in_true_full'].max())
pad = max(0.02, (xmax - xmin) * 0.28)
ax.set_xlim(max(0, xmin - pad * 0.15), xmax + pad * 0.95)

# =========================
# 3. 逐条画箭头
# =========================
for yi, (_, row) in zip(y, df.iterrows()):
    x0 = float(row['rate_in_true_base'])  # 第一个指标值
    x1 = float(row['rate_in_true_full'])  # 第二个指标值

    # 百分比变化（相对第一个值）
    delta_ratio = (x1 - x0) / x0 * 100 if x0 != 0 else np.nan

    # 根据方向选颜色
    color = up_color if delta_ratio > 0 else down_color

    # 只画箭头，不画两端点
    ax.annotate(
        '',
        xy=(x1, yi),
        xytext=(x0, yi),
        arrowprops=dict(
            arrowstyle='->',
            lw=2.8,
            color=color,
            shrinkA=0,
            shrinkB=0
        )
    )

    # 中间文字：起点 → 终点（百分比差）
    xm = (x0 + x1) / 2
    txt = f"{x0:.3f} → {x1:.3f} ({delta_ratio:+.1f}%)"
    ax.text(
        xm, yi + 0.18, txt,
        ha='center', va='bottom',
        fontsize=9, color=color
    )

# =========================
# 4. 分组分隔线
# =========================
sep_y = y[len(reduced) - 1] - 0.5
ax.axhline(sep_y, linewidth=1, color='black', alpha=0.6)

# =========================
# 5. y 轴标签
# =========================
ax.set_yticks(y)
ax.set_yticklabels(df['pair'], fontsize=11)

# 分组标题
left_x = ax.get_xlim()[0]
ax.text(
    left_x, y[0] + 0.7,
    'Reduced representative confusion pairs',
    fontsize=12, fontweight='bold', ha='left'
)
ax.text(
    left_x, y[len(reduced)] + 0.7,
    'Increased representative confusion pairs',
    fontsize=12, fontweight='bold', ha='left'
)

# =========================
# 6. 坐标轴与标题
# =========================
ax.set_xlabel('Confusion rate within the true class', fontsize=12)
ax.set_title(
    'Figure 3. Representative directional confusion changes between the baseline and the full model',
    fontsize=15, pad=12
)

# 右上角说明
ax.text(
    0.99, 1.04,
    'Blue-green right arrow: increase',
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=10, color=up_color
)
ax.text(
    0.99, 1.00,
    'Red left arrow: decrease',
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=10, color=down_color
)

# 网格和边框
ax.grid(axis='x', alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 底部注释
fig.text(
    0.5, 0.012,
    'Note: Each arrow starts from the first metric value and points to the second metric value; '
    'the value in parentheses is the percentage difference between the two values.',
    ha='center', fontsize=10
)

plt.tight_layout(rect=[0.03, 0.04, 0.98, 0.95])

# =========================
# 7. 保存
# =========================
out_dir = Path('.')
png_path = out_dir / 'figure3_confusion_pairs_arrow_style_pct.png'
svg_path = out_dir / 'figure3_confusion_pairs_arrow_style_pct.svg'

fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(svg_path, bbox_inches='tight')
plt.close(fig)

print(f"Saved:\n{png_path}\n{svg_path}")