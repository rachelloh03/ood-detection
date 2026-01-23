import torch
import numpy as np
import plotly.graph_objects as go
from main.ood_detector import OODDetector


def get_graph_visualization(
    ood_detector: OODDetector,
    id_data: torch.Tensor,
    ood_data: torch.Tensor,
    save_path="graph_visualization.html",
    bins=50,
    show_indices: bool = True,
    show_plot: bool = True,
    return_fig: bool = True,
    output_format: str = "html",
    image_format: str = "png",
    image_width: int | None = None,
    image_height: int | None = None,
):
    """
    Get interactive graph visualization for OOD detection.
    Click or hover over bars to see which sample indices correspond to each bin.

    Args:
        ood_detector: OOD detector instance
        id_data: ID data tensor
        ood_data: OOD data tensor
        save_path: Path to save the file (HTML or image depending on output_format)
        bins: Number of bins for the histogram
        show_indices: If True, hover text includes sample indices for each bin.
        show_plot: If True, calls fig.show(). In notebooks you may want False to avoid double display.
        return_fig: If True, returns the Plotly figure. In notebooks you may want False to avoid auto-display.
        output_format: "html" to save as HTML, "image" to save as image (png/jpeg/etc)
        image_format: Image format when output_format="image". Options: "png", "jpeg", "svg", "pdf", "webp"
        image_width: Width of the image in pixels (when output_format="image"). If None, automatically determined.
        image_height: Height of the image in pixels (when output_format="image"). If None, automatically determined.
    """
    id_scores = ood_detector.score(id_data)
    ood_scores = ood_detector.score(ood_data)

    id_np = id_scores.cpu().numpy()
    ood_np = ood_scores.cpu().numpy()

    bin_edges = np.histogram_bin_edges(np.concatenate([id_np, ood_np]), bins=bins)

    id_counts, _ = np.histogram(id_np, bins=bin_edges)
    ood_counts, _ = np.histogram(ood_np, bins=bin_edges)

    id_bin_indices = []
    ood_bin_indices = []

    for i in range(len(bin_edges) - 1):

        id_mask = (id_np >= bin_edges[i]) & (id_np < bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            id_mask = (id_np >= bin_edges[i]) & (id_np <= bin_edges[i + 1])
        id_indices_in_bin = np.where(id_mask)[0].tolist()
        id_bin_indices.append(id_indices_in_bin)

        ood_mask = (ood_np >= bin_edges[i]) & (ood_np < bin_edges[i + 1])
        if i == len(bin_edges) - 2:
            ood_mask = (ood_np >= bin_edges[i]) & (ood_np <= bin_edges[i + 1])
        ood_indices_in_bin = np.where(ood_mask)[0].tolist()
        ood_bin_indices.append(ood_indices_in_bin)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    id_hover_text = []
    ood_hover_text = []

    for i in range(len(bin_centers)):
        id_indices = id_bin_indices[i]
        ood_indices = ood_bin_indices[i]

        if show_indices and len(id_indices) > 0:
            if len(id_indices) <= 20:
                indices_str = ", ".join(map(str, id_indices))
            else:
                indices_str = ", ".join(map(str, id_indices[:20])) + (
                    f", ... (+{len(id_indices) - 20} more)"
                )
            id_hover_text.append(
                f"Bin: [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]<br>"
                f"Count: {id_counts[i]}<br>"
                f"ID Sample Indices: [{indices_str}]"
            )
        else:
            id_hover_text.append(
                f"Bin: [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]<br>"
                f"Count: {id_counts[i]}"
            )

        if show_indices and len(ood_indices) > 0:
            if len(ood_indices) <= 20:
                indices_str = ", ".join(map(str, ood_indices))
            else:
                indices_str = ", ".join(map(str, ood_indices[:20])) + (
                    f", ... (+{len(ood_indices) - 20} more)"
                )
            ood_hover_text.append(
                f"Bin: [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]<br>"
                f"Count: {ood_counts[i]}<br>"
                f"OOD Sample Indices: [{indices_str}]"
            )
        else:
            ood_hover_text.append(
                f"Bin: [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]<br>"
                f"Count: {ood_counts[i]}"
            )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=id_counts,
            name="ID",
            marker_color="lightblue",
            opacity=0.6,
            hovertemplate="%{text}<extra>ID</extra>",
            text=id_hover_text,
            customdata=(id_bin_indices if show_indices else None),
        )
    )

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=ood_counts,
            name="OOD",
            marker_color="lightsalmon",
            opacity=0.6,
            hovertemplate="%{text}<extra>OOD</extra>",
            text=ood_hover_text,
            customdata=(ood_bin_indices if show_indices else None),
        )
    )

    if output_format == "image":
        if image_width is None or image_height is None:
            base_width = max(800, min(2000, bins * 20))
            if image_width is None:
                image_width = base_width
            if image_height is None:
                image_height = base_width // 2

    fig.update_layout(
        title="Score Distributions (Interactive)",
        xaxis_title="OOD Score",
        yaxis_title="Frequency",
        barmode="overlay",
        hovermode="closest",
        legend=dict(x=0.7, y=0.95),
        width=image_width if output_format == "image" else None,
        height=image_height if output_format == "image" else None,
    )

    if output_format == "image":
        if not any(save_path.endswith(ext) for ext in [".png", ".jpeg", ".jpg"]):
            save_path = f"{save_path.rsplit('.', 1)[0] if '.' in save_path else save_path}.{image_format}"
        fig.write_image(
            save_path, format=image_format, width=image_width, height=image_height
        )
    else:
        if not save_path.endswith(".html"):
            save_path = (
                f"{save_path.rsplit('.', 1)[0] if '.' in save_path else save_path}.html"
            )
        fig.write_html(save_path)

    if show_plot:
        fig.show()

    return fig if return_fig else None
