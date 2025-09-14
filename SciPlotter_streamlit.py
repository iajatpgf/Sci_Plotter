import streamlit as st
import pandas as pd
import numpy as np
import squarify
import platform
import json
from io import BytesIO, StringIO

from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt


# --- 中文字体设置 ---
# 确保 Matplotlib 可以显示中文和负号
def setup_chinese_font():
    """根据操作系统设置一个可用的中文字体"""
    try:
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei UI', 'SimHei']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.warning(f"中文字体设置失败，部分中文可能无法正常显示: {e}")


# --- 核心绘图逻辑 ---
def _draw_plot_on_fig(fig, config):
    """核心绘图逻辑，在给定的 Figure 对象上绘制图表"""
    # 设置选定的字体
    selected_font = config.get('selected_font', 'sans-serif')
    plt.rcParams['font.family'] = 'sans-serif'
    # 将选定的字体放在列表首位，作为优先使用字体
    plt.rcParams['font.sans-serif'] = [selected_font, 'Microsoft YaHei UI', 'SimHei', 'PingFang SC',
                                       'WenQuanYi Micro Hei']

    ax = fig.add_subplot(111)
    ax2 = None

    # 从配置字典中获取参数
    plot_type = config['plot_type']
    series_configs = config['series_data']

    title_fontweight = 'bold' if config['title_bold'] else 'normal'
    label_fontweight = 'bold' if config['label_bold'] else 'normal'
    tick_fontweight = 'bold' if config['tick_bold'] else 'normal'
    legend_fontweight = 'bold' if config['legend_bold'] else 'normal'

    # 检查是否需要双Y轴
    if any(s['yaxis'] == 'right' for s in series_configs) and plot_type not in ["矩形树图 (Treemap)",
                                                                                "雷达图 (Radar Chart)",
                                                                                "饼图 (Pie Chart)",
                                                                                "圆环图 (Donut Chart)"]:
        ax2 = ax.twinx()

    # 根据图表类型绘图
    if plot_type == "矩形树图 (Treemap)":
        draw_treemap(ax, series_configs, config)
    elif plot_type == "雷达图 (Radar Chart)":
        fig.clear()
        draw_radar(fig, series_configs, config)
    elif plot_type in ["饼图 (Pie Chart)", "圆环图 (Donut Chart)"]:
        draw_pie_or_donut(ax, series_configs, config)
    else:
        for series in series_configs:
            current_ax = ax2 if series.get('yaxis') == 'right' and ax2 else ax
            if plot_type == "折线图 (Line Plot)":
                current_ax.plot(series['x'], series['y'], label=series['label'], color=series['color'],
                                linewidth=series['linewidth'])
            elif plot_type == "散点图 (Scatter Plot)":
                current_ax.scatter(series['x'], series['y'], label=series['label'], color=series['color'],
                                   s=series['markersize'], marker=series['marker'])
            elif plot_type == "点线图 (Line & Scatter)":
                current_ax.plot(series['x'], series['y'], label=series['label'], color=series['color'],
                                linewidth=series['linewidth'], marker=series['marker'],
                                markersize=np.sqrt(series['markersize']), linestyle='-')
            elif plot_type == "柱状图 (Bar Chart)":
                current_ax.bar(series['x'], series['y'], label=series['label'], color=series['color'])
            elif plot_type == "气泡图 (Bubble Chart)":
                if series['z'] is not None and not series['z'].empty:
                    sizes = (series['z'] - series['z'].min() + 1) * series['markersize']
                    current_ax.scatter(series['x'], series['y'], s=sizes, label=series['label'], color=series['color'],
                                       alpha=0.6, marker=series['marker'])
                else:
                    st.warning(f"系列 '{series['label']}' 未指定大小(Z)列，无法绘制气泡图。")

    # --- 全局样式设置 ---
    if "雷达图" not in plot_type:
        ax.set_title(config['title'], fontsize=config['title_fontsize'], fontweight=title_fontweight)
        ax.set_xlabel(config['xlabel'], fontsize=config['label_fontsize'], fontweight=label_fontweight)
        ax.set_ylabel(config['ylabel'], fontsize=config['label_fontsize'], fontweight=label_fontweight)
        if ax2: ax2.set_ylabel(config['y2label'], fontsize=config['label_fontsize'], fontweight=label_fontweight)

        if config['xlim_check']: ax.set_xlim(config['xlim_min'], config['xlim_max'])
        if config['ylim_check']:
            ax.set_ylim(config['ylim_min'], config['ylim_max'])
            if ax2: ax2.set_ylim(config['ylim_min'], config['ylim_max'])

        if config['x_locator_check'] and config['x_locator_val'] > 0:
            ax.xaxis.set_major_locator(plt.MultipleLocator(config['x_locator_val']))
        if config['y_locator_check'] and config['y_locator_val'] > 0:
            ax.yaxis.set_major_locator(plt.MultipleLocator(config['y_locator_val']))
            if ax2: ax2.yaxis.set_major_locator(plt.MultipleLocator(config['y_locator_val']))

        minor_count = config['minor_tick_count']
        if minor_count > 1:
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_count))
            ax.yaxis.set_minor_locator(AutoMinorLocator(minor_count))
            if ax2: ax2.yaxis.set_minor_locator(AutoMinorLocator(minor_count))
        else:
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_minor_locator(plt.NullLocator())
            if ax2: ax2.yaxis.set_minor_locator(plt.NullLocator())

        if config['show_legend']:
            lines, labels = ax.get_legend_handles_labels()
            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines += lines2
                labels += labels2
            if lines:
                pos = config['legend_pos']
                loc_map = {"Best": "best", "Upper Right": "upper right", "Upper Left": "upper left",
                           "Lower Left": "lower left", "Lower Right": "lower right", "Right": "right",
                           "Center Left": "center left", "Center Right": "center right", "Lower Center": "lower center",
                           "Upper Center": "upper center", "Center": "center"}
                legend_prop = {'size': config['legend_fontsize'], 'weight': legend_fontweight}
                if pos == "Custom":
                    legend = ax.legend(lines, labels, prop=legend_prop,
                                       bbox_to_anchor=(config['legend_x'], config['legend_y']), loc='best')
                else:
                    legend = ax.legend(lines, labels, prop=legend_prop, loc=loc_map.get(pos, "best"))

                if config.get('legend_transparent'):
                    legend.get_frame().set_alpha(0.0)

        ax.grid(config['show_grid'])
        bg_color_to_set = 'none' if config.get('bg_transparent', False) else config.get('bg_color', '#FFFFFF')
        fig.set_facecolor(bg_color_to_set)
        ax.set_facecolor(bg_color_to_set)

        ax.xaxis.label.set_color(config['xaxis_color'])
        ax.yaxis.label.set_color(config['yaxis_color'])
        ax.title.set_color(config['xaxis_color'])

        for spine in ax.spines.values():
            spine.set_linewidth(config['border_width'])
        ax.spines['bottom'].set_edgecolor(config['xaxis_color'])
        ax.spines['top'].set_edgecolor(config['xaxis_color'])
        ax.spines['left'].set_edgecolor(config['yaxis_color'])
        ax.spines['right'].set_edgecolor(config['yaxis_color'])

        direction_map = {"朝外 (Out)": 'out', "朝内 (In)": 'in', "内外 (In/Out)": 'inout'}
        tick_dir = direction_map.get(config['tick_direction'], 'out')

        ax.tick_params(axis='x', which='major', colors=config['xaxis_color'], width=config['major_tick_width'],
                       length=config['major_tick_length'] if config['show_xticks'] else 0,
                       labelsize=config['tick_fontsize'], labelbottom=config['show_xticklabels'], direction=tick_dir)
        ax.tick_params(axis='x', which='minor', colors=config['xaxis_color'], width=config['minor_tick_width'],
                       length=config['minor_tick_length'] if config['show_xticks'] else 0, direction=tick_dir)
        ax.tick_params(axis='y', which='major', colors=config['yaxis_color'], width=config['major_tick_width'],
                       length=config['major_tick_length'] if config['show_yticks'] else 0,
                       labelsize=config['tick_fontsize'], labelleft=config['show_yticklabels'], direction=tick_dir)
        ax.tick_params(axis='y', which='minor', colors=config['yaxis_color'], width=config['minor_tick_width'],
                       length=config['minor_tick_length'] if config['show_yticks'] else 0, direction=tick_dir)

        if ax2:
            ax2.yaxis.label.set_color(config['y2axis_color'])
            ax2.spines['right'].set_edgecolor(config['y2axis_color'])
            ax2.spines['right'].set_linewidth(config['border_width'])
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.tick_params(axis='y', which='major', colors=config['y2axis_color'], width=config['major_tick_width'],
                            length=config['major_tick_length'] if config['show_yticks'] else 0,
                            labelsize=config['tick_fontsize'], labelright=True, direction=tick_dir)
            ax2.tick_params(axis='y', which='minor', colors=config['y2axis_color'], width=config['minor_tick_width'],
                            length=config['minor_tick_length'] if config['show_yticks'] else 0, direction=tick_dir)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(tick_fontweight)
        if ax2:
            for label in ax2.get_yticklabels():
                label.set_fontweight(tick_fontweight)

    fig.tight_layout()


# --- 特殊图表绘制函数 ---
def draw_pie_or_donut(ax, series_configs, config):
    if not series_configs: return
    series = series_configs[0]
    pie_labels = series['x'] if config['show_xticklabels'] else None
    autopct_format = '%1.1f%%' if config['show_yticklabels'] else None

    colors = [s['color'] for s in series_configs] if len(series_configs) > 1 else None
    wedges, texts, autotexts = ax.pie(series['y'], labels=pie_labels, autopct=autopct_format, startangle=90,
                                      colors=colors)

    for text in texts + autotexts:
        text.set_fontsize(config['tick_fontsize'])
        text.set_fontweight('bold' if config['tick_bold'] else 'normal')

    ax.axis('equal')
    if config['plot_type'] == "圆环图 (Donut Chart)":
        bg_color_to_set = 'none' if config.get('bg_transparent', False) else config.get('bg_color', '#FFFFFF')
        ax.add_artist(plt.Circle((0, 0), 0.70, fc=bg_color_to_set))
    ax.set_title(config['title'], fontsize=config['title_fontsize'],
                 fontweight='bold' if config['title_bold'] else 'normal')


def draw_treemap(ax, series_configs, config):
    if not series_configs: return
    series = series_configs[0]
    sizes, labels = series['y'].values, [f"{l}\n({s})" for l, s in zip(series['x'], series['y'].values)]
    colors = [s['color'] for s in series_configs] if len(series_configs) >= len(sizes) else plt.cm.viridis(
        np.linspace(0, 1, len(sizes)))
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax,
                  text_kwargs={'fontsize': config['tick_fontsize'],
                               'fontweight': 'bold' if config['tick_bold'] else 'normal'})
    ax.set_title(config['title'], fontsize=config['title_fontsize'],
                 fontweight='bold' if config['title_bold'] else 'normal')
    plt.axis('off')


def draw_radar(fig, series_configs, config):
    if not series_configs: return
    ax = fig.add_subplot(111, polar=True)
    labels = series_configs[0]['x'].values
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for s_config in series_configs:
        data = s_config['y'].values.tolist()
        data += data[:1]
        ax.plot(angles, data, color=s_config['color'], linewidth=s_config['linewidth'], label=s_config['label'])
        ax.fill(angles, data, color=s_config['color'], alpha=0.25)

    thetagrid_labels = labels if config['show_xticklabels'] else []
    ax.set_thetagrids(np.degrees(angles[:-1]), thetagrid_labels, fontsize=config['tick_fontsize'],
                      weight='bold' if config['tick_bold'] else 'normal')

    if not config['show_yticklabels']: ax.set_yticklabels([])
    ax.set_title(config['title'], fontsize=config['title_fontsize'], y=1.1,
                 fontweight='bold' if config['title_bold'] else 'normal')

    if config['show_legend']:
        legend_prop = {'size': config['legend_fontsize'], 'weight': 'bold' if config['legend_bold'] else 'normal'}
        ax.legend(prop=legend_prop, loc='upper right', bbox_to_anchor=(1.3, 1.1))


# ==============================================================================
# Callbacks
# ==============================================================================
def on_config_upload():
    """Callback function to handle config file uploads."""
    uploaded_file = st.session_state.get('config_uploader')
    if not uploaded_file:
        return
    try:
        imported_config = json.load(uploaded_file)
        # Clear any previous error
        st.session_state['config_upload_error'] = None
        # Update session state.
        for key, value in imported_config.items():
            st.session_state[key] = value
    except Exception as e:
        # Store the error message in session state to display it after the rerun
        st.session_state['config_upload_error'] = f"导入配置文件失败: {e}"


def reset_app():
    """
    Clears all items from the session state to reset the application to its initial state.
    This is like a hard refresh (F5).
    """
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]


# ==============================================================================
# Streamlit App
# ==============================================================================

# 页面配置
st.set_page_config(page_title="交互式科研绘图工具", page_icon="📈", layout="wide")
st.title("📈 交互式科研绘图工具 (Streamlit 版本)")

# --- Session State 初始化 ---
# 仅在第一次运行时初始化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'series_configs' not in st.session_state:
    st.session_state.series_configs = []
if 'colors' not in st.session_state:
    st.session_state.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                               '#bcbd22', '#17becf']
if 'config_upload_error' not in st.session_state:
    st.session_state['config_upload_error'] = None

# ==============================================================================
# Sidebar Controls
# ==============================================================================

with st.sidebar:
    if st.button("🔄 重置应用 (硬刷新)"):
        reset_app()
        st.rerun()

    st.header("1. 数据加载与管理")
    uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'dta', 'txt'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith('.xlsx'):
                st.session_state.df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.lower().endswith('.dta'):
                st.session_state.df = pd.read_stata(uploaded_file)
            elif uploaded_file.name.lower().endswith('.txt'):
                try:
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    st.session_state.df = pd.read_csv(stringio, sep=r'\s+', engine='python')
                except Exception:
                    stringio = StringIO(uploaded_file.getvalue().decode("gbk"))
                    st.session_state.df = pd.read_csv(stringio, sep=r'\s+', engine='python')
            st.session_state.df.columns = [f'Unnamed_{i}' if col is None or str(col).strip() == '' else str(col) for
                                           i, col in enumerate(st.session_state.df.columns)]
        except Exception as e:
            st.error(f"加载文件失败: {e}")
            st.session_state.df = None

    if st.session_state.df is not None:
        st.header("2. 图表类型")
        plot_type_selected = st.selectbox("选择图表类型:", [
            "折线图 (Line Plot)", "散点图 (Scatter Plot)", "点线图 (Line & Scatter)",
            "柱状图 (Bar Chart)", "饼图 (Pie Chart)", "气泡图 (Bubble Chart)",
            "雷达图 (Radar Chart)", "圆环图 (Donut Chart)", "矩形树图 (Treemap)"
        ], key='plot_type')

        st.header("3. 数据系列配置")
        headers = ["-"] + list(st.session_state.df.columns)

        col1, col2 = st.columns(2)
        if col1.button("➕ 添加系列"):
            num_series = len(st.session_state.series_configs)
            new_series = {
                'x_col': headers[1] if len(headers) > 1 else '-',
                'y_col': headers[2] if len(headers) > 2 else '-',
                'z_col': '-',
                'label': f"系列 {num_series + 1}",
                'color': st.session_state.colors[num_series % len(st.session_state.colors)],
                'linewidth': 2.0,
                'markersize': 20.0,
                'marker': 'o',
                'yaxis': '左 (Left)'
            }
            st.session_state.series_configs.append(new_series)
            st.rerun()

        if col2.button("➖ 删除最后一个系列") and st.session_state.series_configs:
            st.session_state.series_configs.pop()
            st.rerun()

        for i, s_config in enumerate(st.session_state.series_configs):
            with st.expander(f"系列 {i + 1}: {s_config['label']}", expanded=True):
                s_config['label'] = st.text_input("标签", value=s_config['label'], key=f"label_{i}")
                c1, c2, c3 = st.columns(3)
                s_config['x_col'] = c1.selectbox("X轴", options=headers,
                                                 index=headers.index(s_config['x_col']) if s_config[
                                                                                               'x_col'] in headers else 0,
                                                 key=f"x_{i}")
                s_config['y_col'] = c2.selectbox("Y轴", options=headers,
                                                 index=headers.index(s_config['y_col']) if s_config[
                                                                                               'y_col'] in headers else 0,
                                                 key=f"y_{i}")
                s_config['z_col'] = c3.selectbox("大小(Z)", options=headers,
                                                 index=headers.index(s_config['z_col']) if s_config[
                                                                                               'z_col'] in headers else 0,
                                                 key=f"z_{i}")
                c1, c2, c3, c4, c5 = st.columns(5)
                s_config['color'] = c1.color_picker("颜色", value=s_config['color'], key=f"color_{i}")
                s_config['linewidth'] = c2.number_input("线宽", min_value=0.0, value=s_config['linewidth'], step=0.5,
                                                        key=f"lw_{i}")
                s_config['markersize'] = c3.number_input("点大小", min_value=0.0, value=s_config['markersize'],
                                                         step=1.0, key=f"ms_{i}")
                s_config['marker'] = c4.selectbox("点形状", options=['o', 's', '^', 'v', 'd', 'p', '*', '+', 'x', '.'],
                                                  index=['o', 's', '^', 'v', 'd', 'p', '*', '+', 'x', '.'].index(
                                                      s_config['marker']), key=f"marker_{i}")
                s_config['yaxis'] = c5.selectbox("Y轴侧", options=["左 (Left)", "右 (Right)"],
                                                 index=0 if s_config['yaxis'] == "左 (Left)" else 1, key=f"yaxis_{i}")

        # --- 4. 全局图表设置 ---
        with st.expander("4. 全局图表设置", expanded=False):
            latex_help = "支持 LaTeX: 使用 $...$ 包裹, ^ 为上标, _ 为下标。例如: $X_{1}$ 或 $m^{2}$"
            st.subheader("标题与标签")
            st.text_input("图表标题", "图表标题", key='title', help=latex_help)
            c1, c2, c3 = st.columns(3)
            c1.text_input("X轴标签", "X轴", key='xlabel', help=latex_help)
            c2.text_input("左Y轴标签", "Y轴", key='ylabel', help=latex_help)
            c3.text_input("右Y轴标签", "右侧Y轴", key='y2label', help=latex_help)

            st.subheader("字体")
            font_list = ["Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana", "Georgia", "Palatino",
                         "Garamond", "sans-serif"]
            st.selectbox("全局字体", font_list, index=0, key='selected_font',
                         help="选择图表中主要的英文字体。中文字体将自动选择。")

            c1, c2, c3, c4 = st.columns(4)
            c1.number_input("标题字号", 1, 100, 16, key='title_fontsize')
            c2.number_input("标签字号", 1, 100, 26, key='label_fontsize')
            c3.number_input("刻度字号", 1, 100, 24, key='tick_fontsize')
            c4.number_input("图例字号", 1, 100, 24, key='legend_fontsize')
            c1, c2, c3, c4 = st.columns(4)
            c1.checkbox("标题加粗", True, key='title_bold')
            c2.checkbox("标签加粗", True, key='label_bold')
            c3.checkbox("刻度加粗", True, key='tick_bold')
            c4.checkbox("图例加粗", True, key='legend_bold')

            st.subheader("坐标轴范围与刻度")
            st.checkbox("自定义X轴范围", key='xlim_check')
            c1, c2 = st.columns(2)
            c1.number_input("X轴最小值", value=0.0, format="%.2f", disabled=not st.session_state.xlim_check,
                            key='xlim_min')
            c2.number_input("X轴最大值", value=10.0, format="%.2f", disabled=not st.session_state.xlim_check,
                            key='xlim_max')

            st.checkbox("自定义Y轴范围", key='ylim_check')
            c1, c2 = st.columns(2)
            c1.number_input("Y轴最小值", value=0.0, format="%.2f", disabled=not st.session_state.ylim_check,
                            key='ylim_min')
            c2.number_input("Y轴最大值", value=10.0, format="%.2f", disabled=not st.session_state.ylim_check,
                            key='ylim_max')

            st.checkbox("自定义X轴主刻度间距", key='x_locator_check')
            st.number_input("X轴主刻度间距", 0.01, 10000.0, 1.0, 0.5, disabled=not st.session_state.x_locator_check,
                            key='x_locator_val')

            st.checkbox("自定义Y轴主刻度间距", key='y_locator_check')
            st.number_input("Y轴主刻度间距", 0.01, 10000.0, 1.0, 0.5, disabled=not st.session_state.y_locator_check,
                            key='y_locator_val')

            st.slider("次刻线间隔数 (1为不显示)", 1, 10, 2, key='minor_tick_count')

            st.subheader("边框与刻度线样式")
            c1, c2 = st.columns(2)
            c1.number_input("边框宽度", 0.0, 10.0, 2.0, 0.5, key='border_width')
            c2.selectbox("刻度朝向", ["朝内 (In)", "朝外 (Out)", "内外 (In/Out)"], index=1, key='tick_direction')
            c1, c2 = st.columns(2)
            c1.number_input("主刻度线宽", 0.0, 5.0, 2.0, 0.2, key='major_tick_width')
            c2.number_input("次刻度线宽", 0.0, 5.0, 1.0, 0.2, key='minor_tick_width')
            c1, c2 = st.columns(2)
            c1.number_input("主刻度线长", 0.0, 20.0, 8.0, 0.5, key='major_tick_length')
            c2.number_input("次刻度线长", 0.0, 20.0, 5.0, 0.5, key='minor_tick_length')

            st.subheader("图例与网格")
            c1, c2 = st.columns(2)
            c1.checkbox("显示图例", True, key='show_legend')
            c2.checkbox("显示网格", False, key='show_grid')
            c1, c2 = st.columns(2)
            c1.selectbox("图例位置",
                         ["Best", "Upper Right", "Upper Left", "Lower Left", "Lower Right", "Right", "Center Left",
                          "Center Right", "Lower Center", "Upper Center", "Center", "Custom"],
                         disabled=not st.session_state.show_legend, key='legend_pos')
            c2.checkbox("图例背景透明", False, key='legend_transparent', disabled=not st.session_state.show_legend)

            if st.session_state.legend_pos == "Custom":
                c1, c2 = st.columns(2)
                c1.number_input("图例X位置", -2.0, 2.0, 1.0, 0.05, key='legend_x')
                c2.number_input("图例Y位置", -2.0, 2.0, 1.0, 0.05, key='legend_y')

            st.subheader("元素可见性")
            c1, c2 = st.columns(2)
            c1.checkbox("显示X轴数值", True, key='show_xticklabels')
            c2.checkbox("显示Y轴数值", True, key='show_yticklabels')
            c1, c2 = st.columns(2)
            c1.checkbox("显示X轴刻度线", True, key='show_xticks')
            c2.checkbox("显示Y轴刻度线", True, key='show_yticks')

            st.subheader("颜色")
            bg_transparent = st.checkbox("透明背景", False, key='bg_transparent')
            c1, c2, c3, c4 = st.columns(4)
            c1.color_picker("背景", "#FFFFFF", key='bg_color', disabled=bg_transparent)
            c2.color_picker("X轴", "#000000", key='xaxis_color')
            c3.color_picker("左Y轴", "#000000", key='yaxis_color')
            c4.color_picker("右Y轴", "#000000", key='y2axis_color')

        # --- 5. 导出设置 ---
        with st.expander("5. 导出", expanded=True):
            c1, c2, c3 = st.columns(3)
            st.number_input("导出宽度(英寸)", 0.1, 100.0, 10.7, 0.1, key='export_width')
            st.number_input("导出高度(英寸)", 0.1, 100.0, 6.6, 0.1, key='export_height')
            st.number_input("分辨率(DPI)", 72, 1200, 300, 50, key='dpi')
            st.selectbox("导出格式", ["PNG", "JPEG", "SVG", "PDF"], key='export_format')

        # --- 6. 参数配置导入导出 ---
        with st.expander("6. 参数配置", expanded=False):
            st.write("将当前所有设置导出为JSON文件，或从JSON文件导入设置。")

            # 收集所有在 session_state 中管理的配置项
            config_keys_to_export = [k for k in st.session_state.keys() if
                                     k not in ['df', 'colors', 'series_configs', 'config_uploader',
                                               'config_upload_error']]
            current_config_dict = {key: st.session_state[key] for key in config_keys_to_export}
            current_config_dict['series_configs'] = st.session_state.series_configs  # 单独添加系列配置

            config_json = json.dumps(current_config_dict, indent=4, ensure_ascii=False)
            st.download_button(
                label="📥 导出配置",
                data=config_json,
                file_name="sciplotter_config.json",
                mime="application/json"
            )

            st.file_uploader(
                "导入配置 (JSON)",
                type=['json'],
                key='config_uploader',
                on_change=on_config_upload
            )

            # Display any error message that was stored during the callback
            if st.session_state.get('config_upload_error'):
                st.error(st.session_state['config_upload_error'])

# ==============================================================================
# Main Panel
# ==============================================================================

if st.session_state.df is None:
    st.info("👋 欢迎使用！请从左侧侧边栏上传您的数据文件开始。")
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=300)
    st.markdown("---")
    st.markdown("""
    ### 功能简介:
    - **支持多种数据格式**: CSV, Excel (.xlsx), Stata (.dta), TXT。
    - **丰富的图表类型**: 折线图, 散点图, 柱状图, 饼图, 雷达图等。
    - **高度可定制化**: 实时调整几乎所有绘图参数，从标题、颜色到坐标轴的每一个细节。
    - **多数据系列与双Y轴**: 在一张图上绘制多个数据系列，并支持左右双Y轴。
    - **高质量导出**: 以自定义的分辨率和尺寸导出图表为 PNG, JPEG, SVG 或 PDF。
    - **配置管理**: 保存您的图表设置，方便下次复用。
    """)
else:
    st.subheader("数据预览")
    st.dataframe(st.session_state.df.head())

    st.subheader("图表预览")

    # 准备绘图数据和配置
    series_data_for_plot = []
    valid_series_found = False
    for s_config in st.session_state.series_configs:
        if s_config['x_col'] != '-' and s_config['y_col'] != '-':
            try:
                series_data = {
                    'x': st.session_state.df[s_config['x_col']],
                    'y': st.session_state.df[s_config['y_col']],
                    'z': st.session_state.df[s_config['z_col']] if s_config['z_col'] != '-' else None,
                    **s_config  # 将系列的所有配置都传入
                }
                series_data_for_plot.append(series_data)
                valid_series_found = True
            except KeyError as e:
                st.error(f"列名 '{e}' 不存在，请检查数据系列配置。")
            except Exception as e:
                st.error(f"处理数据系列时发生错误: {e}")

    # 最终传递给绘图函数的配置字典
    plot_config = {key: st.session_state.get(key) for key in st.session_state.keys() if key not in ['df', 'colors']}
    plot_config['series_data'] = series_data_for_plot

    if valid_series_found:
        setup_chinese_font()
        # --- Figure for Display ---
        fig_display = Figure(
            figsize=(st.session_state.get('export_width', 10.7), st.session_state.get('export_height', 6.6)))
        _draw_plot_on_fig(fig_display, plot_config)
        st.pyplot(fig_display)

        # --- Download Logic ---
        export_format = st.session_state.get('export_format', 'PNG')
        dpi = st.session_state.get('dpi', 300)
        width = st.session_state.get('export_width', 10.7)
        height = st.session_state.get('export_height', 6.6)

        if export_format == 'PNG':
            # Button 1: Download with current settings
            img_buffer_current = BytesIO()
            fig_current = Figure(figsize=(width, height))
            _draw_plot_on_fig(fig_current, plot_config)
            fig_current.savefig(img_buffer_current, format='png', dpi=dpi, facecolor=fig_current.get_facecolor())

            st.sidebar.download_button(
                label="💾 下载 PNG (当前设置)",
                data=img_buffer_current,
                file_name="plot_current.png",
                mime="image/png"
            )

            # Button 2: Download with transparent background
            config_transparent = plot_config.copy()
            config_transparent['bg_transparent'] = True
            config_transparent['legend_transparent'] = True

            img_buffer_transparent = BytesIO()
            fig_transparent = Figure(figsize=(width, height))
            _draw_plot_on_fig(fig_transparent, config_transparent)
            fig_transparent.savefig(img_buffer_transparent, format='png', dpi=dpi, facecolor='none')

            st.sidebar.download_button(
                label="💾 下载 PNG (透明背景)",
                data=img_buffer_transparent,
                file_name="plot_transparent.png",
                mime="image/png"
            )
        else:
            # Handle other formats (JPEG, SVG, PDF)
            mime_types = {
                "JPEG": "image/jpeg",
                "SVG": "image/svg+xml",
                "PDF": "application/pdf"
            }
            img_buffer = BytesIO()
            fig_download = Figure(figsize=(width, height))
            _draw_plot_on_fig(fig_download, plot_config)
            fig_download.savefig(img_buffer, format=export_format.lower(), dpi=dpi,
                                 facecolor=fig_download.get_facecolor())

            st.sidebar.download_button(
                label=f"💾 下载图片 ({export_format})",
                data=img_buffer,
                file_name=f"plot.{export_format.lower()}",
                mime=mime_types.get(export_format)
            )

    elif not st.session_state.series_configs:
        st.warning("请在左侧添加至少一个数据系列。")
    else:
        st.warning("请在数据系列中选择有效的 X 和 Y 轴数据。")

