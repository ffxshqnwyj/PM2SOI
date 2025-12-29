import argparse
import math
import pickle
import re
from ast import literal_eval
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import _init_paths  # noqa: F401


def get_plot_draw_styles():
    return [
        {'color': (1.0, 0.0, 0.0), 'line_style': '-'},
        {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
        {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
        {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
        {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
        {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
        {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
        {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
        {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
        {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
        {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
        {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
        {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
        {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
        {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
        {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
        {'color': (0.7, 0.6, 0.2), 'line_style': '-'},
        {'color': (255.0 / 255.0, 102.0 / 255.0, 102.0 / 255.0), 'line_style': '-'},
        {'color': (153.0 / 255.0, 255.0 / 255.0, 153.0 / 255.0), 'line_style': '-'},
        {'color': (102.0 / 255.0, 102.0 / 255.0, 255.0 / 255.0), 'line_style': '-'},
        {'color': (255.0 / 255.0, 192.0 / 255.0, 203.0 / 255.0), 'line_style': '-'},
        {'color': (255.0 / 255.0, 215.0 / 255.0, 0 / 255.0), 'line_style': '-'},
    ]


def get_tracker_display_name(tracker):
    disp_name = tracker.get('disp_name')
    if disp_name:
        return disp_name
    if tracker.get('run_id') is None:
        return f"{tracker['name']}_{tracker['param']}"
    return f"{tracker['name']}_{tracker['param']}_{tracker['run_id']:03d}"


def parse_lasot_attributes():
    lasot_file = Path('lib/test/evaluation/lasotdataset.py')
    text = lasot_file.read_text()
    match = re.search(r'def _get_sequence_list\(self\):(.*?)return sequence_list', text, re.S)
    if not match:
        raise RuntimeError('Failed to locate attribute lists in lasotdataset.py')

    attr_map = {}
    pattern = re.compile(r'\s+([A-Za-z_]+)\s*=\s*(\[[^\]]*\])', re.S)
    for name, list_txt in pattern.findall(match.group(1)):
        lower = name.lower()
        if lower == 'sequencelist':
            continue
        try:
            seqs = literal_eval(list_txt)
        except Exception:
            continue
        if isinstance(seqs, list) and seqs and isinstance(seqs[0], str):
            attr_map[name.replace('_', ' ')] = seqs
    return attr_map


def load_eval_data(eval_path: Path):
    if not eval_path.is_file():
        raise FileNotFoundError(f'eval_data.pkl not found at {eval_path}')
    with open(eval_path, 'rb') as f:
        return pickle.load(f)


def compute_attribute_scores(eval_data):
    attr_map = parse_lasot_attributes()
    attr_map.pop('sequence list', None)
    attr_map.pop('SequenceList', None)

    seq_names = eval_data['sequences']
    seq_to_idx = {name: idx for idx, name in enumerate(seq_names)}

    success_arr = np.array(eval_data['avg_overlap_all'], dtype=float)  # [seq, tracker], already 0-1

    scores = {}
    for attr, seqs in attr_map.items():
        idxs = [seq_to_idx[s] for s in seqs if s in seq_to_idx]
        if not idxs:
            continue
        values = success_arr[idxs, :]
        scores[attr] = values.mean(axis=0)  # keep 0-1
    return scores


def plot_radar_draw_save(scores, trackers, save_path):
    plot_draw_styles = get_plot_draw_styles()

    font_size = 9
    line_width = 2
    marker_size = 4
    marker = 'o'
    bbox_to_anchor = (0.5, -0.15)
    font_size_legend = 10
    handlelength = 1
    legend_loc = 'upper center'
    ylim = [0, 1.05]

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams["legend.labelspacing"] = 0.2
    matplotlib.rcParams["legend.borderpad"] = 0.3

    attributes = list(scores.keys())
    num_attr = len(attributes)
    if num_attr == 0:
        raise RuntimeError('No attributes found to plot.')

    angles = [n / float(num_attr) * 2 * math.pi for n in range(num_attr)]
    angles += angles[:1]

    score_matrix = np.stack([scores[attr] for attr in attributes], axis=0)  # [attr, tracker]
    avg_score_per_tracker = score_matrix.mean(axis=0)

    ax = plt.subplot(111, polar=True)
    ax.set_thetagrids([])
    ax.set_ylim(ylim)
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(1.2)

    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, ylim[1]], color='lightgrey', linewidth=0.8)

    plotted_lines = []
    legend_text = []

    tracker_order = np.argsort(avg_score_per_tracker)[::-1]

    for idx, t_idx in enumerate(tracker_order):
        values = score_matrix[:, t_idx].tolist()
        values += values[:1]
        style = plot_draw_styles[idx % len(plot_draw_styles)]
        line = ax.plot(angles, values, linewidth=line_width, markersize=marker_size,
                       marker=marker, color=style['color'], linestyle=style['line_style'])
        plotted_lines.append(line[0])
        disp_name = get_tracker_display_name(trackers[t_idx])
        legend_text.append(f'{disp_name} [{avg_score_per_tracker[t_idx]:.3f}]')

    if plotted_lines:
        ax.legend(plotted_lines, legend_text, loc=legend_loc, fancybox=False, edgecolor='black',
                  fontsize=font_size_legend, framealpha=1.0, handlelength=handlelength, handletextpad=0.3,
                  bbox_to_anchor=bbox_to_anchor, ncol=min(len(trackers), 3))

    # Attribute labels with Chinese translation and values (use first tracker)
    chinese_map = {
        'IlluminationVariation': '光照变化',
        'ScaleVariation': '尺度变化',
        'Deformation': '形变',
        'MotionBlur': '运动模糊',
        'FastMotion': '快速运动',
        'BackgroundClutter': '背景杂乱',
        'CameraMotion': '相机运动',
        'Rotation': '旋转',
        'PartialOcclusion': '部分遮挡',
        'FullOcclusion': '完全遮挡',
        'OutofView': '出视野',
        'ViewpointChange': '视角变化',
        'LowResolution': '低分辨率',
        'AspectRatioChange': '纵横比变化',
        'AspectRationChange': '纵横比变化',
    }

    for i, attr in enumerate(attributes):
        angle = angles[i]
        label_en = re.sub(r'([a-z])([A-Z])', r'\1 \2', attr).strip()
        value = score_matrix[i, tracker_order[0]]
        text = f'{label_en}\n({value:.3f})'
        ax.text(angle, ylim[1] + 0.05, text, ha='center', va='center', fontsize=8)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Plot LaSOT attribute radar chart (pytracking style).')
    parser.add_argument('--eval_path', type=str, default='output/test/result_plots/lasot/eval_data.pkl',
                        help='Path to eval_data.pkl produced by analysis_results.py')
    parser.add_argument('--save', type=str, default='lasot_attr_radar.png',
                        help='Output file path (leave empty to only display)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval_data = load_eval_data(Path(args.eval_path))
    scores = compute_attribute_scores(eval_data)
    trackers = eval_data['trackers']
    save_path = args.save if args.save else None
    plot_radar_draw_save(scores, trackers, save_path)
