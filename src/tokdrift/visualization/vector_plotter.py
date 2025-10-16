from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from .plot_style import get_column_display_name, get_variant_order


class VectorPlotter:
    def __init__(self):
        pass
    
    def _get_task_language_prefix(self, task_name):
        """Get the language prefix for determining base variant."""
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

    def _find_last_layer(self, shifts_dir):
        """Find the maximum layer number in the shifts directory."""
        layer_files = list(shifts_dir.glob("layer_*.pt"))
        if not layer_files:
            return None

        layer_numbers = []
        for layer_file in layer_files:
            try:
                layer_num = int(layer_file.stem.split('_')[1])
                layer_numbers.append(layer_num)
            except (ValueError, IndexError):
                continue

        return max(layer_numbers) if layer_numbers else None

    def _load_vectors_from_variant(self, variant_dir, layer_to_plot=None, target_layer=0):
        """
        Load vectors from a variant directory.

        Args:
            variant_dir: Path to the variant directory
            layer_to_plot: Specific layer number to plot (None for last layer)

        Returns:
            dict with 'vectors', 'variant_name', 'task_names', 'layer_number', 'vector_task_mapping'
        """
        variant_name = variant_dir.name

        # Get all task directories (excluding 'prev')
        task_dirs = [d for d in variant_dir.iterdir() if d.is_dir() and d.name != 'prev']

        if not task_dirs:
            print(f"    No task directories found for variant {variant_name}")
            return None

        all_vectors = []
        task_names = []
        actual_layer = None
        vector_task_mapping = []  # List of (start_idx, end_idx, task_name) tuples

        current_idx = 0
        for task_dir in task_dirs:
            task_name = task_dir.name
            shifts_dir = task_dir / "shifts"

            if not shifts_dir.exists():
                continue

            # Find the last layer if not specified
            if layer_to_plot is None:
                last_layer = self._find_last_layer(shifts_dir)
                if last_layer is None:
                    continue
                layer_num = last_layer - target_layer
            else:
                layer_num = layer_to_plot

            layer_file = shifts_dir / f"layer_{layer_num}.pt"

            if not layer_file.exists():
                continue

            try:
                # Load the layer vectors
                layer_vectors = torch.load(layer_file, map_location='cpu', weights_only=True)

                if layer_vectors.shape[0] > 0:
                    n_vectors = layer_vectors.shape[0]
                    all_vectors.append(layer_vectors)
                    if task_name not in task_names:
                        task_names.append(task_name)
                    if actual_layer is None:
                        actual_layer = layer_num

                    # Record which vectors belong to which task
                    vector_task_mapping.append((current_idx, current_idx + n_vectors, task_name))
                    current_idx += n_vectors

                    print(f"      Loaded {layer_vectors.shape[0]} vectors from {task_name} layer {layer_num}")

            except Exception as e:
                print(f"      Error loading layer {layer_num} from task {task_name}: {e}")
                continue

        if not all_vectors:
            print(f"    No vectors found for variant {variant_name}")
            return None

        # Combine all vectors
        combined_vectors = torch.cat(all_vectors, dim=0)
        vectors_np = combined_vectors.numpy()

        print(f"    Total: {vectors_np.shape[0]} vectors from {len(all_vectors)} tasks, layer {actual_layer}")

        return {
            'vectors': vectors_np,
            'variant_name': variant_name,
            'task_names': task_names,
            'layer_number': actual_layer,
            'vector_task_mapping': vector_task_mapping
        }

    def _apply_reduction(self, vectors, method="pca", n_components=2):
        """
        Apply dimensionality reduction.

        Args:
            vectors: Input vectors
            method: "pca" or "tsne"
            n_components: Number of components (2 or 3)

        Returns:
            Reduced vectors and reducer object
        """
        if method.lower() == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_vectors = reducer.fit_transform(vectors)
            return reduced_vectors, reducer
        elif method.lower() == "tsne":
            perplexity = 70
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                max_iter=500,
                learning_rate='auto',
                init='pca',
                method='barnes_hut',
                n_jobs=16
            )
            reduced_vectors = reducer.fit_transform(vectors)
            return reduced_vectors, reducer
        else:
            raise ValueError(f"Unsupported method: {method}")

    def _get_axis_labels(self, method, reducer, n_components):
        """Get axis labels for the reduction method."""
        method = method.lower()

        if method == "pca":
            if hasattr(reducer, 'explained_variance_ratio_'):
                if n_components == 2:
                    return (
                        f'PC1 ({reducer.explained_variance_ratio_[0]:.2%})',
                        f'PC2 ({reducer.explained_variance_ratio_[1]:.2%})'
                    )
                elif n_components == 3:
                    return (
                        f'PC1 ({reducer.explained_variance_ratio_[0]:.2%})',
                        f'PC2 ({reducer.explained_variance_ratio_[1]:.2%})',
                        f'PC3 ({reducer.explained_variance_ratio_[2]:.2%})'
                    )
            return ('PC1', 'PC2') if n_components == 2 else ('PC1', 'PC2', 'PC3')
        elif method == "tsne":
            if n_components == 2:
                return ('t-SNE 1', 't-SNE 2')
            elif n_components == 3:
                return ('t-SNE 1', 't-SNE 2', 't-SNE 3')

        return (f'{method.upper()} 1', f'{method.upper()} 2')

    

    def _prepare_combined_vector_plot_data(self, model_dir: Path, target_layer: int = 0):
        """Load and organize vectors for combined plots with aggregated and per-variant groupings."""
        all_variant_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name != 'prev']
        if not all_variant_dirs:
            return None

        available_variant_names = {d.name for d in all_variant_dirs}

        naming_dir_names = [name for name in self.identifier_variants if name in available_variant_names]
        spacing_dir_names = sorted(name for name in available_variant_names if name not in naming_dir_names)

        variant_order = get_variant_order()
        naming_variant_order = [variant for variant in variant_order if '-' in variant]
        spacing_variant_order = [
            variant for variant in variant_order
            if '-' not in variant and variant not in {'op_name', 'op_all'}
        ]

        tab10_colors = list(plt.cm.tab10.colors)
        naming_color_map = {
            variant: tab10_colors[idx % len(tab10_colors)]
            for idx, variant in enumerate(naming_variant_order)
        }

        tab20_colors = list(plt.cm.tab20.colors)
        spacing_color_map = {
            variant: tab20_colors[idx % len(tab20_colors)]
            for idx, variant in enumerate(spacing_variant_order)
        }

        extra_spacing_variants = [variant for variant in spacing_dir_names if variant not in spacing_color_map]
        offset = len(spacing_color_map)
        for idx, variant in enumerate(extra_spacing_variants):
            spacing_color_map[variant] = tab20_colors[(offset + idx) % len(tab20_colors)]

        naming_data = None
        if naming_dir_names:
            naming_data = self._collect_variant_data_by_language(
                model_dir, naming_dir_names, naming_color_map, "Naming Variants", target_layer
            )

        spacing_data = None
        if spacing_dir_names:
            spacing_data = self._collect_variant_data_simple(
                model_dir, spacing_dir_names, spacing_color_map, "Spacing Variants", target_layer
            )

        naming_entries = []
        naming_order_available = []
        if naming_data:
            entry_map = {entry['series_key']: entry for entry in naming_data['variant_data']}
            for entry in entry_map.values():
                entry['color'] = naming_color_map.get(entry['series_key'], entry['color'])
            for variant in naming_variant_order:
                if variant in entry_map:
                    naming_entries.append(entry_map[variant])
                    naming_order_available.append(variant)
            for key, entry in entry_map.items():
                if key not in naming_order_available:
                    naming_entries.append(entry)
                    naming_order_available.append(key)

        spacing_entries = []
        spacing_order_available = []
        if spacing_data:
            entry_map = {entry['series_key']: entry for entry in spacing_data['variant_data']}
            for entry in entry_map.values():
                entry['color'] = spacing_color_map.get(entry['series_key'], entry['color'])
            for variant in spacing_variant_order:
                if variant in entry_map:
                    spacing_entries.append(entry_map[variant])
                    spacing_order_available.append(variant)
            for key, entry in entry_map.items():
                if key in {'op_name', 'op_all'}:
                    continue
                if key not in spacing_order_available:
                    spacing_entries.append(entry)
                    spacing_order_available.append(key)

        spacing_entries = [entry for entry in spacing_entries if entry['series_key'] not in {'op_name', 'op_all'}]
        spacing_order_available = [variant for variant in spacing_order_available if variant not in {'op_name', 'op_all'}]

        naming_vectors_all = None
        if naming_entries:
            naming_vectors_all = np.vstack([entry['vectors'] for entry in naming_entries])

        spacing_vectors_all = None
        if spacing_entries:
            spacing_vectors_all = np.vstack([entry['vectors'] for entry in spacing_entries])

        if naming_vectors_all is None and spacing_vectors_all is None:
            return None

        tab20b_colors = list(plt.cm.tab20b.colors)
        aggregated_colors = {
            'naming': tab20b_colors[0],
            'spacing': tab20b_colors[4],
        }

        return {
            'naming_entries': naming_entries,
            'spacing_entries': spacing_entries,
            'naming_vectors_all': naming_vectors_all,
            'spacing_vectors_all': spacing_vectors_all,
            'aggregated_colors': aggregated_colors,
            'naming_variant_order': naming_order_available,
            'spacing_variant_order': spacing_order_available,
        }

    def _build_combined_vector_legend(self, fig, handles, labels,
                                      naming_entries, spacing_entries,
                                      naming_order, spacing_order,
                                      bbox_to_anchor, loc) -> bool:
        """Create an ordered legend for combined vector plots."""
        legend_entries = OrderedDict()

        aggregated_labels = ['Naming (All)', 'Spacing (All)']
        for agg_label in aggregated_labels:
            for handle, label in zip(handles, labels):
                if label == agg_label and label not in legend_entries:
                    legend_entries[label] = handle
                    break

        naming_display_set = {entry['label'] for entry in naming_entries}
        for variant in naming_order:
            display_label = get_column_display_name(variant)
            if display_label in naming_display_set and display_label not in legend_entries:
                for handle, label in zip(handles, labels):
                    if label == display_label:
                        legend_entries[label] = handle
                        break

        for handle, label in zip(handles, labels):
            if label in naming_display_set and label not in legend_entries:
                legend_entries[label] = handle

        spacing_display_set = {entry['label'] for entry in spacing_entries}
        for variant in spacing_order:
            display_label = get_column_display_name(variant)
            if display_label in spacing_display_set and display_label not in legend_entries:
                for handle, label in zip(handles, labels):
                    if label == display_label:
                        legend_entries[label] = handle
                        break

        for handle, label in zip(handles, labels):
            if label in spacing_display_set and label not in legend_entries:
                legend_entries[label] = handle

        average_labels = ['Naming Average', 'Spacing Average']
        for avg_label in average_labels:
            for handle, label in zip(handles, labels):
                if label == avg_label and label not in legend_entries:
                    legend_entries[label] = handle
                    break

        if legend_entries:
            fig.legend(
                legend_entries.values(),
                legend_entries.keys(),
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                frameon=True,
                fontsize=16,
            )
            return True

        return False

    def plot_vector(self, data_dir="./data/output/hidden-states-data/analysis_hidden_states",
                              model_name="Qwen2.5-Coder-32B-Instruct", target_layer=0):
        """Plot vector 2D visualizations and export three PNGs (all/naming/spacing)."""
        base_path = Path(data_dir)
        model_dir = base_path / model_name

        if not model_dir.is_dir():
            print(f"Model directory not found: {model_dir}")
            return

        combined_data = self._prepare_combined_vector_plot_data(model_dir, target_layer=target_layer)
        if not combined_data:
            print(f"No vector data available for {model_name}")
            return

        naming_entries = combined_data['naming_entries']
        spacing_entries = combined_data['spacing_entries']
        naming_vectors_all = combined_data['naming_vectors_all']
        spacing_vectors_all = combined_data['spacing_vectors_all']
        aggregated_colors = combined_data['aggregated_colors']
        naming_order = combined_data['naming_variant_order']
        spacing_order = combined_data['spacing_variant_order']

        output_dir = Path(f"./data/output/plots/vectors/{model_name}/{target_layer}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for method in ["pca", "tsne"]:
            def _scatter_groups(vectors_groups, colors, labels, title):
                if not vectors_groups:
                    return None
                fig, ax = plt.subplots(figsize=(7, 6))
                fig.patch.set_facecolor('white')

                combined_vectors = np.vstack(vectors_groups)
                reduced_vectors, _ = self._apply_reduction(
                    combined_vectors, method=method, n_components=2
                )

                start_idx = 0
                legend_handles = []
                legend_labels = []

                for vectors, color, label in zip(vectors_groups, colors, labels):
                    end_idx = start_idx + vectors.shape[0]
                    scatter = ax.scatter(
                        reduced_vectors[start_idx:end_idx, 0],
                        reduced_vectors[start_idx:end_idx, 1],
                        color=color,
                        label=label,
                        alpha=0.6,
                        s=20,
                    )
                    handle = Line2D([0], [0], marker='o', color=color, linestyle='None', label=label,
                                    markersize=6)
                    legend_handles.append(handle)
                    legend_labels.append(label)
                    start_idx = end_idx

                # ax.set_title(title, loc='left', fontsize=16, fontweight='bold', pad=10)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                ax.set_facecolor('white')

                if title == "(a) Naming vs Spacing (All)":
                    bbox_to_anchor = (1.32, 0.5)
                else:
                    bbox_to_anchor = (1.18, 0.5)

                has_legend = False
                if legend_handles:
                    has_legend = self._build_combined_vector_legend(
                        fig, legend_handles, legend_labels,
                        naming_entries if 'Naming' in title else [],
                        spacing_entries if 'Spacing' in title else [],
                        naming_order,
                        spacing_order,
                        bbox_to_anchor=bbox_to_anchor,
                        loc='center right'
                    )

                if has_legend:
                    plt.tight_layout(rect=[0, 0, 1, 1])
                else:
                    plt.tight_layout()

                return fig

            overview_groups = []
            overview_colors = []
            overview_labels = []
            if naming_vectors_all is not None:
                overview_groups.append(naming_vectors_all)
                overview_colors.append(aggregated_colors['naming'])
                overview_labels.append('Naming (All)')
            if spacing_vectors_all is not None:
                overview_groups.append(spacing_vectors_all)
                overview_colors.append(aggregated_colors['spacing'])
                overview_labels.append('Spacing (All)')

            fig_all = _scatter_groups(
                overview_groups,
                overview_colors,
                overview_labels,
                title="(a) Naming vs Spacing (All)"
            )
            if fig_all:
                output_path = output_dir / f"combined_vector_{method}_all.png"
                fig_all.savefig(output_path, dpi=200, bbox_inches='tight')
                plt.close(fig_all)
                print(f"{method.upper()} combined-all plot saved to {output_path}")

            naming_groups = [entry['vectors'] for entry in naming_entries]
            naming_colors = [entry['color'] for entry in naming_entries]
            naming_labels = [entry['label'] for entry in naming_entries]

            fig_naming = _scatter_groups(
                naming_groups,
                naming_colors,
                naming_labels,
                title="(b) Naming Variants"
            )
            if fig_naming:
                output_path = output_dir / f"combined_vector_{method}_naming.png"
                fig_naming.savefig(output_path, dpi=200, bbox_inches='tight')
                plt.close(fig_naming)
                print(f"{method.upper()} combined-naming plot saved to {output_path}")

            spacing_groups = [entry['vectors'] for entry in spacing_entries]
            spacing_colors = [entry['color'] for entry in spacing_entries]
            spacing_labels = [entry['label'] for entry in spacing_entries]

            fig_spacing = _scatter_groups(
                spacing_groups,
                spacing_colors,
                spacing_labels,
                title="(c) Spacing Variants"
            )
            if fig_spacing:
                output_path = output_dir / f"combined_vector_{method}_spacing.png"
                fig_spacing.savefig(output_path, dpi=200, bbox_inches='tight')
                plt.close(fig_spacing)
                print(f"{method.upper()} combined-spacing plot saved to {output_path}")
            
            # raise Exception("Stop here")

    def plot_vector_3d(self, data_dir="./data/output/hidden-states-data/analysis_hidden_states",
                                 model_name="Qwen2.5-Coder-32B-Instruct", target_layer=0):
        """Plot combined 3D PCA visualizations and export three PNGs (all/naming/spacing)."""
        base_path = Path(data_dir)
        model_dir = base_path / model_name

        if not model_dir.is_dir():
            print(f"Model directory not found: {model_dir}")
            return

        combined_data = self._prepare_combined_vector_plot_data(model_dir, target_layer=target_layer)
        if not combined_data:
            print(f"No vector data available for {model_name}")
            return

        naming_entries = combined_data['naming_entries']
        spacing_entries = combined_data['spacing_entries']
        naming_vectors_all = combined_data['naming_vectors_all']
        spacing_vectors_all = combined_data['spacing_vectors_all']
        aggregated_colors = combined_data['aggregated_colors']
        naming_order = combined_data['naming_variant_order']
        spacing_order = combined_data['spacing_variant_order']

        output_dir = Path(f"./data/output/plots/vectors/{model_name}/{target_layer}")
        output_dir.mkdir(parents=True, exist_ok=True)

        def _scatter_groups_3d(vectors_groups, colors, labels, title):
            if not vectors_groups:
                return None
            fig = plt.figure(figsize=(7, 7))
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            combined_vectors = np.vstack(vectors_groups)
            reduced_vectors, _ = self._apply_reduction(
                combined_vectors, method='pca', n_components=3
            )

            start_idx = 0
            legend_handles = []
            legend_labels = []

            for vectors, color, label in zip(vectors_groups, colors, labels):
                end_idx = start_idx + vectors.shape[0]
                ax.scatter(
                    reduced_vectors[start_idx:end_idx, 0],
                    reduced_vectors[start_idx:end_idx, 1],
                    reduced_vectors[start_idx:end_idx, 2],
                    color=color,
                    label=label,
                    alpha=0.6,
                    s=20,
                )
                legend_handles.append(Line2D([0], [0], marker='o', color=color,
                                             linestyle='None', markersize=6, label=label))
                legend_labels.append(label)
                start_idx = end_idx

            # ax.set_title(title, loc='left', fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.set_facecolor('white')

            if title == "(a) Naming vs Spacing (All)":
                bbox_to_anchor = (1.27, 0.5)
            else:
                bbox_to_anchor = (1.16, 0.5)

            has_legend = False
            if legend_handles:
                has_legend = self._build_combined_vector_legend(
                    fig, legend_handles, legend_labels,
                    naming_entries if 'Naming' in title else [],
                    spacing_entries if 'Spacing' in title else [],
                    naming_order,
                    spacing_order,
                    bbox_to_anchor=bbox_to_anchor,
                    loc='center right'
                )

            if has_legend:
                plt.tight_layout(rect=[0, 0, 1, 1])
            else:
                plt.tight_layout()

            return fig

        overview_groups = []
        overview_colors = []
        overview_labels = []
        if naming_vectors_all is not None:
            overview_groups.append(naming_vectors_all)
            overview_colors.append(aggregated_colors['naming'])
            overview_labels.append('Naming (All)')
        if spacing_vectors_all is not None:
            overview_groups.append(spacing_vectors_all)
            overview_colors.append(aggregated_colors['spacing'])
            overview_labels.append('Spacing (All)')

        fig_all = _scatter_groups_3d(
            overview_groups,
            overview_colors,
            overview_labels,
            title="(a) Naming vs Spacing (All)"
        )
        if fig_all:
            output_path = output_dir / "combined_vector_3d_all.png"
            fig_all.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig_all)
            print(f"3D combined-all plot saved to {output_path}")

        naming_groups = [entry['vectors'] for entry in naming_entries]
        naming_colors = [entry['color'] for entry in naming_entries]
        naming_labels = [entry['label'] for entry in naming_entries]

        fig_naming = _scatter_groups_3d(
            naming_groups,
            naming_colors,
            naming_labels,
            title="(b) Naming Variants"
        )
        if fig_naming:
            output_path = output_dir / "combined_vector_3d_naming.png"
            fig_naming.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig_naming)
            print(f"3D combined-naming plot saved to {output_path}")

        spacing_groups = [entry['vectors'] for entry in spacing_entries]
        spacing_colors = [entry['color'] for entry in spacing_entries]
        spacing_labels = [entry['label'] for entry in spacing_entries]

        fig_spacing = _scatter_groups_3d(
            spacing_groups,
            spacing_colors,
            spacing_labels,
            title="(c) Spacing Variants"
        )
        if fig_spacing:
            output_path = output_dir / "combined_vector_3d_spacing.png"
            fig_spacing.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig_spacing)
            print(f"3D combined-spacing plot saved to {output_path}")

    def _collect_variant_data_by_language(self, model_dir, variant_list, color_map, title, target_layer=0):
        """
        Collect variant data separated by task language (for naming variants).

        Returns:
            dict with 'variant_data' (list of variant entries) and 'title'
        """
        all_variant_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name != 'prev']
        variant_dirs = [d for d in all_variant_dirs if d.name in variant_list]

        if not variant_dirs:
            print(f"  No variant directories found for {title}")
            return None

        all_variant_data = []

        for variant_dir in variant_dirs:
            variant_data = self._load_vectors_from_variant(variant_dir, target_layer=target_layer)
            if variant_data is None:
                continue

            # Group vectors by task language (Python vs Java)
            python_vectors = []
            java_vectors = []

            for start_idx, end_idx, task_name in variant_data['vector_task_mapping']:
                language_prefix = self._get_task_language_prefix(task_name)
                task_vectors = variant_data['vectors'][start_idx:end_idx]

                if language_prefix.lower() == 'python':
                    python_vectors.append(task_vectors)
                else:  # Java
                    java_vectors.append(task_vectors)

            # Create separate entries for Python and Java tasks
            if python_vectors:
                combined_python = np.vstack(python_vectors)
                base_variant = 'snake_case'
                series_key = f"{base_variant}-{variant_dir.name}"
                label = get_column_display_name(series_key)
                color = color_map.get(series_key, sns.color_palette("Set2", 6)[0])

                print(f"    {title} - Variant {variant_dir.name} (Python): {combined_python.shape[0]} vectors, series_key={series_key}")

                all_variant_data.append({
                    'vectors': combined_python,
                    'color': color,
                    'label': label,
                    'series_key': series_key
                })

            if java_vectors:
                combined_java = np.vstack(java_vectors)
                base_variant = 'camel_case'
                series_key = f"{base_variant}-{variant_dir.name}"
                label = get_column_display_name(series_key)
                color = color_map.get(series_key, sns.color_palette("Set2", 6)[0])

                print(f"    {title} - Variant {variant_dir.name} (Java): {combined_java.shape[0]} vectors, series_key={series_key}")

                all_variant_data.append({
                    'vectors': combined_java,
                    'color': color,
                    'label': label,
                    'series_key': series_key
                })

        if not all_variant_data:
            return None

        return {
            'variant_data': all_variant_data,
            'title': title
        }

    def _collect_variant_data_simple(self, model_dir, variant_list, color_map, title, target_layer=0):
        """
        Collect variant data without language separation (for spacing variants).

        Returns:
            dict with 'variant_data' (list of variant entries) and 'title'
        """
        all_variant_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name != 'prev']
        variant_dirs = [d for d in all_variant_dirs if d.name in variant_list]

        if not variant_dirs:
            print(f"  No variant directories found for {title}")
            return None

        all_variant_data = []

        for variant_dir in variant_dirs:
            variant_data = self._load_vectors_from_variant(variant_dir, target_layer=target_layer)
            if variant_data is None:
                continue

            vectors = variant_data['vectors']
            variant_name = variant_dir.name
            label = get_column_display_name(variant_name)
            color = color_map.get(variant_name, (0.5, 0.5, 0.5))  # fallback to gray

            print(f"    {title} - Variant {variant_name}: {vectors.shape[0]} vectors")

            all_variant_data.append({
                'vectors': vectors,
                'color': color,
                'label': label,
                'series_key': variant_name
            })

        if not all_variant_data:
            return None

        return {
            'variant_data': all_variant_data,
            'title': title
        }
