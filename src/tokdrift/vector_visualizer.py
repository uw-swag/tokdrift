import argparse

from .visualization.similarity_plotter import SimilarityPlotter
from .visualization.vector_plotter import VectorPlotter


def plot_similarity(data_dir="./data/output/hidden-states-data/analysis_hidden_states", target_model="Qwen2.5-Coder-32B-Instruct"):
    """Generate similarity figure for selected large models."""
    plotter = SimilarityPlotter()
    plotter.plot_similarity(data_dir, target_model=target_model)

def plot_vector(data_dir="./data/output/hidden-states-data/analysis_hidden_states", model_name="Qwen2.5-Coder-32B-Instruct"):
    """Plot vector 2D visualizations and export three PNGs (all/naming/spacing)."""
    plotter = VectorPlotter()
    layer_values = [1, 2, 0, 4]
    # layer=4 means processing the middle layer, layer=0 means processing the last layer
    for layer in layer_values:
        plotter.plot_vector(data_dir, model_name, layer)

def plot_vector_3d(data_dir="./data/output/hidden-states-data/analysis_hidden_states", model_name="Qwen2.5-Coder-32B-Instruct"):
    """Plot vector 3D visualizations and export three PNGs (all/naming/spacing)."""
    plotter = VectorPlotter()
    layer_values = [1, 2, 0, 4]
    # layer=4 means processing the middle layer, layer=0 means processing the last layer
    for layer in layer_values:
        plotter.plot_vector_3d(data_dir, model_name, layer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot vector analysis results')
    parser.add_argument('--vector', action='store_true', 
                       help='Plot vector')
    parser.add_argument('--vector_3d', action='store_true',
                       help='Plot vector 3d')
    parser.add_argument('--similarity', action='store_true',
                       help='Plot similarity')
    parser.add_argument('--model', default="Qwen2.5-Coder-32B-Instruct", help='Model name')
    
    args = parser.parse_args()

    model_name = args.model if "/" not in args.model else args.model.split("/")[-1]
    
    if args.similarity:
        plot_similarity(target_model=model_name)
    if args.vector:
        plot_vector(model_name=model_name)
    if args.vector_3d:
        plot_vector_3d(model_name=model_name)