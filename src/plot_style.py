"""Presentation settings shared by the session charts."""

from matplotlib.colors import LinearSegmentedColormap

INK = '#283c34'
MUTED = '#78847c'
GRID = '#e9ede7'
GREEN = '#4d8a76'
BLUE = '#718bb4'
GOLD = '#bd9a62'
ROSE = '#c87879'
SERIES_COLORS = (GREEN, BLUE, GOLD)
ANGLE_CMAP = LinearSegmentedColormap.from_list(
    'posture_angles', ['#edf3ed', '#aacdba', '#6eaa91', '#d1ad79', '#c77872']
)


def style_axis(ax):
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis='both', colors=MUTED, labelsize=9, length=0, pad=8)
    ax.xaxis.label.set(color=MUTED, fontsize=9)
    ax.yaxis.label.set(color=MUTED, fontsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.7)


def finish_timeline(fig, axes, time_labels, interval, left=0.15):
    # Tick density is presentational; every original sample remains plotted.
    tick_step = max(interval, (len(time_labels) + 5) // 6)
    for ax in axes:
        style_axis(ax)
        ax.set_xticks(time_labels[::tick_step])
        ax.set_xticklabels(time_labels[::tick_step], rotation=0, ha='center')
        ax.margins(x=0.015)
    fig.set_facecolor('white')
    fig.subplots_adjust(left=left, right=0.96, bottom=0.08, top=0.94, hspace=0.64)
