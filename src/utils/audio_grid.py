"""
Utility function to create a grid visualization of audio samples.

This is used in example.ipynb to visualize the audio samples.

Thanks cursor! - Joel
"""

import json
import torch
from IPython.display import HTML, display
from utils.convert import sequence_to_wav
import os
import tempfile
import shutil
import base64


def create_audio_grid(
    output_samples: torch.Tensor,
    input_samples: torch.Tensor | None = None,
    indices: list[int] | None = None,
    step: int = 10,
    max_samples: int = 20,
    cols: int = 4,
    include_velocity: bool = False,
    output_path: str | None = None,
):
    """
    Create a grid visualization of audio samples.

    Args:
        output_samples: Tensor of shape (N, L) for model output sequences to display.
        input_samples: Optional tensor of shape (N, L_in) for input sequences. If provided,
            each cell shows both input and output audio for that index.
        indices: Optional list of specific indices to display. If None, uses step-based selection.
        step: Step size for selecting indices (e.g., 10 means indices 0, 10, 20, ...)
        max_samples: Maximum number of samples to display
        cols: Number of columns in the grid
        include_velocity: Whether to include velocity when converting to audio
        output_path: If set, write the grid to this HTML file instead of displaying
            in the notebook (keeps notebook file size small). Open the file in a browser to view.

    Returns:
        None. If output_path is set, writes HTML to that file; otherwise displays in notebook.
    """
    samples = output_samples
    if indices is None:
        indices = list(range(0, len(samples), step))[:max_samples]
    else:
        indices = indices[:max_samples]

    temp_dir = tempfile.mkdtemp()
    # List of (idx, output_base64, input_base64?, output_tokens, input_tokens?)
    audio_data_list: list[tuple[int, str, str | None, list, list | None]] = []

    try:
        for idx in indices:
            if idx >= len(samples):
                break

            input_base64: str | None = None
            input_tokens: list | None = None
            if input_samples is not None and idx < len(input_samples):
                try:
                    input_sequence = input_samples[idx].tolist()
                    input_tokens = input_sequence
                    input_path = os.path.join(temp_dir, f"input_{idx}.wav")
                    sequence_to_wav(
                        input_sequence, input_path, include_velocity=include_velocity
                    )
                    with open(input_path, "rb") as f:
                        input_base64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    print(f"Warning: Could not create input audio for index {idx}: {e}")

            sequence = samples[idx].tolist()
            audio_path = os.path.join(temp_dir, f"sample_{idx}.wav")
            try:
                sequence_to_wav(sequence, audio_path, include_velocity=include_velocity)
                with open(audio_path, "rb") as f:
                    audio_data = f.read()
                    output_base64 = base64.b64encode(audio_data).decode("utf-8")
                    audio_data_list.append(
                        (idx, output_base64, input_base64, sequence, input_tokens)
                    )
            except Exception as e:
                print(f"Warning: Could not create audio for index {idx}: {e}")
                continue

        # Build token data for detail view: { idx: { input: [...], output: [...] }, ... }
        token_data: dict[int, dict[str, list]] = {}
        for idx, _o, _i, out_tok, in_tok in audio_data_list:
            token_data[idx] = {"output": out_tok}
            if in_tok is not None:
                token_data[idx]["input"] = in_tok
        token_data_json = json.dumps(token_data)

        html_content = (
            '<div id="grid-view" style="display: grid; grid-template-columns: repeat('
            + str(cols)
            + ', 1fr); gap: 20px; padding: 20px; max-width: 1200px;">'
        )

        for idx, output_base64, input_base64, _out_tok, _in_tok in audio_data_list:
            cells = []
            if input_base64 is not None:
                cells.append(
                    '<div style="margin-bottom: 8px;" onclick="event.stopPropagation();">'
                    '<div style="font-size: 12px; color: #666; margin-bottom: 4px;">Input</div>'
                    '<audio controls style="width: 100%; max-width: 250px;">'
                    f'<source src="data:audio/wav;base64,{input_base64}" type="audio/wav">'
                    "Your browser does not support the audio element.</audio></div>"
                )
            cells.append(
                '<div onclick="event.stopPropagation();"><div style="font-size: 12px; color: #666; margin-bottom: 4px;">Output</div>'
                '<audio controls style="width: 100%; max-width: 250px;">'
                f'<source src="data:audio/wav;base64,{output_base64}" type="audio/wav">'
                "Your browser does not support the audio element.</audio></div>"
            )
            cell_style = (
                "border: 1px solid #ddd; padding: 15px; border-radius: 8px; "
                "text-align: center; background-color: #f9f9f9; "
                "box-shadow: 0 2px 4px rgba(0,0,0,0.1); "
                "cursor: pointer;"
            )
            html_content += f"""
            <div class="grid-cell" style="{cell_style}" data-idx="{idx}" onclick="showTokenDetail({idx})" title="Click to view full token sequences">
                <div style="font-weight: bold; margin-bottom: 10px; font-size: 14px; color: #333;">Index {idx}</div>
                {"".join(cells)}
            </div>
            """

        html_content += "</div>"

        detail_view_html = """
        <div id="detail-view" style="display: none; padding: 24px; max-width: 900px; margin: 0 auto;">
            <button onclick="hideTokenDetail()" style="margin-bottom: 16px; padding: 8px 16px; cursor: pointer; font-size: 14px;">&larr; Back to grid</button>
            <h2 id="detail-title" style="margin-top: 0;">Token sequences</h2>
            <div id="detail-content"></div>
        </div>
        """

        script_html = f"""
        <script>
        var tokenData = {token_data_json};
        function showTokenDetail(idx) {{
            var data = tokenData[idx];
            if (!data) return;
            var html = '';
            function esc(s) {{ return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }}
            if (data.input) {{
                html += '<h3>Input tokens (length ' + data.input.length + ')</h3>';
                html += '<pre style="background: #f0f0f0; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; max-height: 300px; overflow-y: auto;">' + esc(JSON.stringify(data.input, null, 2)) + '</pre>';
            }}
            html += '<h3>Output tokens (length ' + data.output.length + ')</h3>';
            html += '<pre style="background: #f0f0f0; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; max-height: 400px; overflow-y: auto;">' + esc(JSON.stringify(data.output, null, 2)) + '</pre>';
            document.getElementById('detail-content').innerHTML = html;
            document.getElementById('detail-title').textContent = 'Token sequences for index ' + idx;
            document.getElementById('grid-view').style.display = 'none';
            document.getElementById('detail-view').style.display = 'block';
        }}
        function hideTokenDetail() {{
            document.getElementById('detail-view').style.display = 'none';
            document.getElementById('grid-view').style.display = 'grid';
        }}
        </script>
        """

        full_html = html_content + detail_view_html + script_html

        if output_path is not None:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
                f.write("<title>Audio grid</title></head><body>")
                f.write(full_html)
                f.write("</body></html>")
            print(f"Audio grid written to {output_path}")
        else:
            display(HTML(full_html))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
