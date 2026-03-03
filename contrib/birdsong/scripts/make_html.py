"""
Generate a single-file HTML with embedded spectrograms and audio clips for SAE features.

Edit the FEATURES list below, then run:
    uv run scripts/make_html.py
"""

import base64
import pathlib

RUNS_ROOT = pathlib.Path("/fs/ess/PAS2136/samuelstevens/saev/runs")
OUTPUT_PATH = pathlib.Path("notebooks/examples.html")

# Edit this list to add/remove features
FEATURES = [
    {
        "ckpt": "0y6vhggq",
        "shards": "5e37a03c",
        "feature": 11,
        "notes": "I think this is cicada noises.",
    },
    {
        "ckpt": "0y6vhggq",
        "shards": "5e37a03c",
        "feature": 15,
        "notes": "",
    },
    {
        "ckpt": "0y6vhggq",
        "shards": "5e37a03c",
        "feature": 12438,
        "notes": "",
    },
    {
        "ckpt": "x5m3a1pw",
        "shards": "5e37a03c",
        "feature": 9,
        "notes": "",
    },
    {
        "ckpt": "x5m3a1pw",
        "shards": "5e37a03c",
        "feature": 12,
        "notes": "It sounds like someone picking the recorder up?",
    },
    {
        "ckpt": "x5m3a1pw",
        "shards": "5e37a03c",
        "feature": 13,
        "notes": "",
    },
    {
        "ckpt": "x5m3a1pw",
        "shards": "5e37a03c",
        "feature": 15,
        "notes": "",
    },
    {
        "ckpt": "j0x7a7go",
        "shards": "5e37a03c",
        "feature": 12,
        "notes": "",
    },
]


def encode_file(path: pathlib.Path) -> str:
    with open(path, "rb") as fd:
        return base64.b64encode(fd.read()).decode("utf-8")


def make_example_html(clips_dir: pathlib.Path, i: int) -> str:
    spec_b64 = encode_file(clips_dir / f"{i}_spectrogram.png")
    sae_spec_b64 = encode_file(clips_dir / f"{i}_sae_spectrogram.png")
    time_clip_b64 = encode_file(clips_dir / f"{i}_time_clip.ogg")
    time_freq_clip_b64 = encode_file(clips_dir / f"{i}_time_freq_clip.ogg")

    return f"""
            <div class="example">
                <h4>Example {i + 1}</h4>
                <div class="spectrogram-row">
                    <div>
                        <img src="data:image/png;base64,{spec_b64}" alt="Spectrogram {i}">
                        <div class="caption">Original Spectrogram</div>
                    </div>
                    <div>
                        <img src="data:image/png;base64,{sae_spec_b64}" alt="SAE Spectrogram {i}">
                        <div class="caption">SAE Highlighted Spectrogram</div>
                    </div>
                </div>
                <div class="audio-row">
                    <div class="audio-item">
                        <label>Time-Clipped Audio</label>
                        <audio controls>
                            <source src="data:audio/ogg;base64,{time_clip_b64}" type="audio/ogg">
                        </audio>
                    </div>
                    <div class="audio-item">
                        <label>Time+Freq-Clipped Audio</label>
                        <audio controls>
                            <source src="data:audio/ogg;base64,{time_freq_clip_b64}" type="audio/ogg">
                        </audio>
                    </div>
                </div>
            </div>"""


def make_feature_section(
    ckpt: str, shards: str, feature: int, notes: str | None = None
) -> str:
    clips_dir = RUNS_ROOT / ckpt / "inference" / shards / "clips" / str(feature)
    examples_html = [make_example_html(clips_dir, i) for i in range(4)]
    notes_html = f"<p><strong>Sam's notes:</strong> {notes}</p>" if notes else ""

    return f"""
    <div class="feature-section">
        <div class="feature-header">
            <h2>Checkpoint: {ckpt} | Feature: {feature}</h2>
            <p>Top 4 activating examples with spectrograms and audio clips</p>
            {notes_html}
        </div>

        <div class="example-grid">
{"".join(examples_html)}
        </div>
    </div>"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAE Feature Examples - Birdsong</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .feature-section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .feature-header {{
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .example-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }}
        .example {{
            background: #fafafa;
            border-radius: 6px;
            padding: 15px;
            border: 1px solid #eee;
        }}
        .example h4 {{
            margin: 0 0 10px 0;
            color: #555;
        }}
        .spectrogram-row {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .spectrogram-row > div {{
            flex: 1;
        }}
        .spectrogram-row img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .audio-row {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .audio-item {{
        }}
        .audio-item label {{
            display: block;
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }}
        audio {{
            width: 100%;
        }}
        .caption {{
            font-size: 11px;
            color: #888;
            text-align: center;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <h1>SAE Feature Examples - Birdsong Spectrograms</h1>
{sections}
</body>
</html>"""


def main():
    sections = [make_feature_section(**f) for f in FEATURES]
    html = HTML_TEMPLATE.format(sections="".join(sections))
    OUTPUT_PATH.write_text(html)
    print(f"Wrote {len(html):,} bytes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
