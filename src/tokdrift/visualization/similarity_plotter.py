import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from typing import Optional
from .plot_style import get_column_display_name, get_variant_order


class SimilarityPlotter:
    def __init__(self):
        pass
    
    def _get_task_language_prefix(self, task_name):
        """Get the language prefix for accessing similarity data."""
        if 'java2python' in task_name.lower():
            return "Java"
        elif 'python2java' in task_name.lower():
            return "Python"
        elif 'python' in task_name.lower():
            return "Python"
        elif 'java' in task_name.lower():
            return "Java"
        else:
            print(f"      Unknown task name: {task_name}, defaulting to Python")
        return "Python"

    def _prepare_similarity_plot_data(self,
                                      data_dir: str,
                                      target_model: str) -> Optional[dict]:
        base_path = Path(data_dir)
        model_dir = base_path / target_model

        if not model_dir.is_dir():
            print(f"Missing similarity directory for {target_model}")
            return None

        preferred_order = ["snake_case", "camel_case", "pascal_case", "screaming_snake_case"]
        tab10_colors = list(plt.cm.tab10.colors)
        naming_palette = tab10_colors[:8]
        spacing_palette = list(plt.cm.tab20.colors)
        average_palette = list(plt.cm.tab20b.colors)
        naming_avg_color = average_palette[0]
        spacing_avg_color = average_palette[4]

        variant_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not variant_dirs:
            print(f"No variant directories found for {target_model}")
            return None

        ordered_variant_dirs = []
        for preferred in preferred_order:
            for variant_dir in variant_dirs:
                if variant_dir.name == preferred:
                    ordered_variant_dirs.append(variant_dir)
                    break
        remaining_dirs = sorted(
            [d for d in variant_dirs if d.name not in preferred_order],
            key=lambda x: x.name
        )
        ordered_variant_dirs.extend(remaining_dirs)

        variant_series = OrderedDict()
        num_layers = None

        for variant_dir in ordered_variant_dirs:
            json_files = sorted(variant_dir.glob("*.json"))
            if not json_files:
                continue

            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                except Exception as exc:
                    print(f"Error reading {json_file}: {exc}")
                    continue

                averages = data.get('averages', {})
                layer_averages = averages.get('layer_averages', {})
                weight = averages.get('total_identifiers') or averages.get('total_operators') or 0

                if not layer_averages or weight <= 0:
                    continue

                if num_layers is None:
                    layer_indices = [
                        int(key.split('_')[-1])
                        for key in layer_averages.keys()
                        if key.startswith('layer_') and key.split('_')[-1].isdigit()
                    ]
                    if layer_indices:
                        num_layers = max(layer_indices) + 1
                    else:
                        num_layers = len(layer_averages)

                if num_layers is None:
                    continue

                layer_values = np.array(
                    [layer_averages.get(f"layer_{idx}", 0.0) for idx in range(num_layers)],
                    dtype=float
                )

                if variant_dir.name in preferred_order:
                    task_name = json_file.stem
                    language_prefix = self._get_task_language_prefix(task_name)
                    base_variant = 'snake_case' if language_prefix.lower() == 'python' else 'camel_case'
                    series_key = f"{base_variant}-{variant_dir.name}"
                    series_type = 'identifier'
                else:
                    series_key = variant_dir.name
                    base_variant = None
                    series_type = 'operator'

                entry = variant_series.setdefault(
                    series_key,
                    {
                        'type': series_type,
                        'variant': variant_dir.name,
                        'base_variant': base_variant,
                        'sum': np.zeros(num_layers, dtype=float),
                        'weight': 0.0,
                    }
                )

                entry['sum'] += layer_values * weight
                entry['weight'] += weight

        if not variant_series or num_layers is None:
            print(f"No valid similarity data for {target_model}")
            return None

        identifier_entries = []
        operator_entries = []
        for key, entry in variant_series.items():
            if entry['weight'] <= 0:
                continue
            entry['averages'] = entry['sum'] / entry['weight']
            entry['label_key'] = key
            if entry['type'] == 'identifier':
                identifier_entries.append(entry)
            else:
                operator_entries.append(entry)

        variant_order = get_variant_order()
        order_index = {name: idx for idx, name in enumerate(variant_order)}
        identifier_entries.sort(key=lambda e: order_index.get(e['label_key'], len(order_index)))
        operator_entries.sort(key=lambda e: order_index.get(e['label_key'], len(order_index)))

        layer_percent = np.linspace(0, 100, num_layers)

        for idx, entry in enumerate(identifier_entries):
            entry['color'] = naming_palette[idx % len(naming_palette)]
            entry['label'] = get_column_display_name(entry['label_key'])

        for idx, entry in enumerate(operator_entries):
            entry['color'] = spacing_palette[idx % len(spacing_palette)]
            entry['label'] = get_column_display_name(entry['label_key'])

        naming_avg = None
        spacing_avg = None

        if identifier_entries:
            total_weight = sum(entry['weight'] for entry in identifier_entries)
            if total_weight > 0:
                naming_avg = sum(entry['averages'] * entry['weight'] for entry in identifier_entries) / total_weight

        if operator_entries:
            total_weight = sum(entry['weight'] for entry in operator_entries)
            if total_weight > 0:
                spacing_avg = sum(entry['averages'] * entry['weight'] for entry in operator_entries) / total_weight

        return {
            'layer_percent': layer_percent,
            'identifier_entries': identifier_entries,
            'operator_entries': operator_entries,
            'naming_avg': naming_avg,
            'spacing_avg': spacing_avg,
            'naming_avg_color': naming_avg_color,
            'spacing_avg_color': spacing_avg_color,
        }

    def plot_similarity(self,
                          data_dir="./data/output/hidden-states-data/analysis_hidden_states",
                          target_model: str = "Qwen2.5-Coder-32B-Instruct"):
        """Export separate naming and spacing similarity plots for a single model."""

        data = self._prepare_similarity_plot_data(data_dir, target_model)
        if not data:
            return

        layer_percent = data['layer_percent']
        identifier_entries = data['identifier_entries']
        operator_entries = data['operator_entries']
        naming_avg = data['naming_avg']
        spacing_avg = data['spacing_avg']
        naming_avg_color = data['naming_avg_color']
        spacing_avg_color = data['spacing_avg_color']

        output_dir = Path("./data/output/plots/similarity")
        output_dir.mkdir(parents=True, exist_ok=True)

        if identifier_entries:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            naming_handles = []

            for entry in identifier_entries:
                line, = ax.plot(
                    layer_percent,
                    entry['averages'],
                    color=entry['color'],
                    linewidth=2,
                    label=entry['label']
                )
                naming_handles.append(line)

            if naming_avg is not None:
                avg_line, = ax.plot(
                    layer_percent,
                    naming_avg,
                    color=naming_avg_color,
                    linewidth=3,
                    linestyle='--',
                    label='Naming\nAverage'
                )
                naming_handles.append(avg_line)

            ax.set_xlim(-3, 103)
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlabel('Layer Depth (%)', fontsize=16)
            ax.set_ylabel('', fontsize=16)

            legend_handles = [
                Line2D([0], [0], color=line.get_color(), linewidth=3,
                       linestyle=line.get_linestyle(), label=line.get_label())
                for line in naming_handles
            ]
            ax.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc='upper right',
                frameon=True,
                bbox_to_anchor=(1.30, 1),
                fontsize=12,
            )

            plt.tight_layout()
            naming_path = output_dir / f"similarity_{target_model.replace('/', '_')}_naming.png"
            plt.savefig(naming_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Naming similarity plot saved to {naming_path}")
        else:
            print("No naming variant data available for naming plot.")

        if operator_entries:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            spacing_handles = []

            for entry in operator_entries:
                line, = ax.plot(
                    layer_percent,
                    entry['averages'],
                    color=entry['color'],
                    linewidth=2,
                    label=entry['label']
                )
                spacing_handles.append(line)

            if spacing_avg is not None:
                avg_line, = ax.plot(
                    layer_percent,
                    spacing_avg,
                    color=spacing_avg_color,
                    linewidth=3,
                    linestyle='--',
                    label='Spacing\nAverage'
                )
                spacing_handles.append(avg_line)

            ax.set_xlim(-3, 103)
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_xlabel('Layer Depth (%)', fontsize=16)
            ax.set_ylabel('', fontsize=16)

            legend_handles = [
                Line2D([0], [0], color=line.get_color(), linewidth=3,
                       linestyle=line.get_linestyle(), label=line.get_label())
                for line in spacing_handles
            ]
            ax.legend(
                legend_handles,
                [handle.get_label() for handle in legend_handles],
                loc='center right',
                frameon=True,
                bbox_to_anchor=(1.30, 0.46),
                fontsize=12,
            )

            plt.tight_layout()
            spacing_path = output_dir / f"similarity_{target_model.replace('/', '_')}_spacing.png"
            plt.savefig(spacing_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Spacing similarity plot saved to {spacing_path}")
        else:
            print("No spacing variant data available for spacing plot.")
