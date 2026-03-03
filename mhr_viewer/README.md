# MHR Mesh Viewer – Streamlit Application

Interactive 3D viewer for [Meta's Momentum Human Rig (MHR)](https://github.com/facebookresearch/MHR) body model.
Built to help researchers and developers inspect the mesh topology and determine which vertex/triangle indices correspond to specific anatomical regions on the body envelope.

---

## Features

### 3D Mesh Visualization
- **Interactive 3D mesh** — rotate, zoom, and pan the full body mesh using Plotly's built-in 3D controls (mouse drag to orbit, scroll to zoom, right-drag to pan)
- **Color modes** — colour vertices by `height` (Z), `depth` (Y), `x-axis` (X), or `distance` (from origin) for spatial orientation
- **Wireframe overlay** — optional triangle edge rendering on top of the shaded mesh
- **Adjustable vertex point size** — control scatter-marker size for vertex picking

### Parameter Controls (Sidebar)
- **Pose parameters (204)** — all joint rotations, translations, and scales grouped by anatomical region:
  - Root (Translation + Rotation), Spine, Neck & Head, Right/Left Arm, Right/Left Leg, Right/Left Hand Fingers, Foot Flex, Scale/Proportions
- **Identity coefficients (45)** — body-shape blendshapes split into Body (20), Head (20), and Hand (5) groups
- **Expression coefficients (72)** — facial expression blendshape weights
- **Preset poses** — one-click load of common poses: T-Pose, Arms Down, Arms Forward, Squat, Wave, Head Turn, Lean Forward
- **Reset button** — clear all parameters back to zero

### Selection & Inspection
- **Vertex lookup** — enter comma-separated vertex indices to highlight them in red on the mesh
- **Triangle lookup** — enter triangle indices to highlight edges and see constituent vertex IDs
- **Click-to-select** — click or box-select vertices directly on the 3D plot using Plotly's selection tools
- **Vertex range selector** — highlight a contiguous index range (e.g. vertices 5000–6000) to explore spatial groupings
- **Selection details table** — shows Index, X, Y, Z for each selected vertex plus the selection centroid
- **Copy-friendly output** — selected indices are displayed in a code block ready for copy-paste

### Anatomical Region Finder
- **3D bounding-box search** — use X / Y / Z range sliders to define a spatial region and find all vertices inside it
- Coordinates are in **centimetres**, **Z-up**, **Y-forward** (MHR convention)
- Found vertices are automatically highlighted on the mesh

### Export
- **OBJ export** — download the current posed mesh as a Wavefront `.obj` file
- **Selection export** — download selected vertex indices as a `.txt` file

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Conda environment** | `mhr` — must contain `streamlit`, `plotly`, `torch`, `numpy`, `pymomentum-cpu` (or `pymomentum-gpu`) |
| **MHR repository** | Cloned at `../MHR` relative to this workspace root (i.e. both repos sit side-by-side) |
| **MHR assets** | The `MHR/assets/` folder must contain the model files — download from the [MHR v1.0.0 release](https://github.com/facebookresearch/MHR/releases/tag/v1.0.0) and unzip |

### Required asset files

```
MHR/assets/
├── compact_v6_1.model          # Model parameterization
├── lod1.fbx                    # Rig (fallback face loading)
├── lod1_faces.npy              # Pre-extracted face indices (preferred)
└── mhr_model.pt                # TorchScript model (LOD-1)
```

The app loads the **TorchScript model** (`mhr_model.pt`) for forward-pass inference (avoids `pymomentum` segfault issues in some environments) and reads face topology from `lod1_faces.npy` (falling back to the FBX if the `.npy` is missing).

---

## Installation

If you already have the `mhr` conda environment set up for the MHR repo, the only additional packages needed are `streamlit` and `plotly`:

```bash
conda activate mhr
pip install streamlit plotly
```

If starting from scratch:

```bash
conda create -n mhr python=3.11
conda activate mhr
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pymomentum-cpu streamlit plotly numpy trimesh
cd /path/to/MHR
pip install -e .
```

---

## Usage

```bash
conda activate mhr
cd mhr_viewer
streamlit run app.py
```

Open the URL printed in the terminal (default: **http://localhost:8501**).

---

## Application Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  SIDEBAR                          │  MAIN AREA                  │
│                                   │                             │
│  🦴 MHR Controls                  │  Display Options            │
│  ├─ Quick Presets                 │  [wireframe] [color] [size] │
│  ├─ Model Settings (info)        │                             │
│  ├─ Identity (45 sliders)        │  ┌─────────────────────┐    │
│  ├─ Pose Groups                  │  │                     │    │
│  │   ├─ Root (6)                 │  │   3D Plotly Mesh    │    │
│  │   ├─ Spine (18)              │  │     (interactive)    │    │
│  │   ├─ Neck & Head (6)         │  │                     │    │
│  │   ├─ Right Arm (10)          │  └─────────────────────┘    │
│  │   ├─ Left Arm (10)           │                             │
│  │   ├─ Right Leg (9)           │  🔍 Vertex Lookup │ 🔺 Tri  │
│  │   ├─ Left Leg (9)            │  📐 Range Selector          │
│  │   ├─ R. Hand Fingers (27)    │  📋 Selection Details       │
│  │   ├─ L. Hand Fingers (27)    │  🦴 Region Finder           │
│  │   ├─ Foot Flex (8)           │  💾 Export                   │
│  │   └─ Scale (74)              │                             │
│  └─ Expression (72 sliders)     │                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Workflow: Identifying Anatomical Regions

1. **Load a neutral pose** — select "T-Pose (Default)" from presets
2. **Orient the view** — rotate the mesh to face the region of interest (e.g. left shoulder)
3. **Narrow with bounding box** — use the Anatomical Region Finder Z/X/Y sliders to isolate the rough area
4. **Click "Find vertices in region"** — the matching vertices will highlight in red
5. **Refine** — use vertex range selector or click individual vertices to fine-tune
6. **Read off indices** — copy from the Selection Details panel or download as `.txt`
7. **Pose-test** — adjust pose sliders to verify the selected vertices deform with the expected body part

---

## Model Details

| Property | Value |
|---|---|
| Vertices (LOD-1) | 18,439 |
| Triangles (LOD-1) | 36,874 |
| Joints | 127 |
| Identity blendshapes | 45 (20 body + 20 head + 5 hand) |
| Pose parameters | 204 (rotations, translations, scales) |
| Expression blendshapes | 72 |
| Coordinate system | Centimetres, Z-up, Y-forward |
| Inference backend | TorchScript (CPU) |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `TorchScript model not found` | Download assets from MHR releases and unzip into `MHR/assets/` |
| `libtorch.so not found` | Make sure `import torch` is available in the `mhr` conda env |
| Segfault on `MHR.from_files()` | This app uses the TorchScript model instead — no action needed |
| Slow first load | The model is cached after first load; subsequent interactions are fast |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| Click selection not working | Use Plotly's **box select** or **lasso select** tools in the toolbar above the plot |

---

## File Structure

```
mhr_viewer/
├── app.py          # Streamlit application (single file)
└── README.md       # This file
```

---

## License

This viewer is a utility tool. The MHR model and assets are subject to [Meta's Apache 2.0 License](https://github.com/facebookresearch/MHR/blob/main/LICENSE).
