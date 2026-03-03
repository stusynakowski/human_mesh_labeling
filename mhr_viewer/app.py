"""
MHR (Momentum Human Rig) Interactive Mesh Viewer
=================================================
A Streamlit application for visualizing and inspecting the MHR human body model.
Allows interactive pose/identity/expression parameter editing, 3D mesh rotation,
and vertex/triangle selection with index readout.

Usage:
    conda activate mhr
    cd <this-directory>
    streamlit run app.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch

# ---------------------------------------------------------------------------
# Resolve MHR repo that sits next to this repo
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MHR_REPO = SCRIPT_DIR.parent.parent / "MHR"  # ../MHR relative to the workspace
MHR_ASSETS = MHR_REPO / "assets"
SELECTIONS_DIR = SCRIPT_DIR / "selections"
SELECTIONS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MHR Mesh Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants from the MHR model
# ---------------------------------------------------------------------------
NUM_IDENTITY = 45
NUM_MODEL_PARAMS = 204
NUM_EXPRESSION = 72

# Human-readable groups for the 204 model parameters.
# Index ranges reference the parameter names extracted from the MHR model.
POSE_GROUPS = {
    "Root (Translation + Rotation)": list(range(0, 6)),
    "Spine": list(range(6, 24)),
    "Neck & Head": list(range(24, 30)),
    "Right Arm": list(range(30, 40)),
    "Left Arm": list(range(40, 50)),
    "Right Leg": list(range(50, 59)),
    "Left Leg": list(range(59, 68)),
    "Right Hand Fingers": list(range(68, 95)),
    "Left Hand Fingers": list(range(95, 122)),
    "Foot Flex": list(range(122, 130)),
    "Scale / Proportions": list(range(130, 204)),
}

ALL_PARAM_NAMES = [
    "root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz",
    "spine0_rx_flexible", "spine_twist0", "spine0_ry_flexible", "spine_lean0",
    "spine0_rz_flexible", "spine_bend0", "spine1_rx_flexible", "spine_twist1",
    "spine1_ry_flexible", "spine_lean1", "spine1_rz_flexible", "spine_bend1",
    "spine2_rx_flexible", "spine2_ry_flexible", "spine2_rz_flexible",
    "spine3_rx_flexible", "spine3_ry_flexible", "spine3_rz_flexible",
    "neck_twist", "neck_lean", "neck_bend", "head_twist", "head_lean", "head_bend",
    "r_clavicle_rx", "r_clavicle_ry", "r_clavicle_rz",
    "r_uparm_twist", "r_uparm_ry", "r_uparm_rz", "r_elbow_bend",
    "r_lowarm_twist", "r_wrist_ry", "r_wrist_rz",
    "l_clavicle_rx", "l_clavicle_ry", "l_clavicle_rz",
    "l_uparm_twist", "l_uparm_ry", "l_uparm_rz", "l_elbow_bend",
    "l_lowarm_twist", "l_wrist_ry", "l_wrist_rz",
    "r_upleg_twist", "r_upleg_ry", "r_upleg_rz", "r_knee_bend",
    "r_lowleg_twist", "r_foot_bend", "r_foot_lean0", "r_foot_lean1", "r_ball_bend",
    "l_upleg_twist", "l_upleg_ry", "l_upleg_rz", "l_knee_bend",
    "l_lowleg_twist", "l_foot_bend", "l_foot_lean0", "l_foot_lean1", "l_ball_bend",
    "r_thumb0_ry", "r_thumb0_rz", "r_thumb1_rx", "r_thumb1_ry", "r_thumb1_rz",
    "r_thumb2_rz", "r_thumb3_rz",
    "r_index1_ry", "r_ring1_ry", "r_pinky1_ry", "r_middle1_ry",
    "r_index1_rz", "r_index2_rz", "r_index3_rz",
    "r_middle1_rz", "r_middle2_rz", "r_middle3_rz",
    "r_ring1_rz", "r_ring2_rz", "r_ring3_rz",
    "r_pinky1_rz", "r_pinky2_rz", "r_pinky3_rz",
    "r_index1_rx", "r_ring1_rx", "r_pinky1_rx", "r_middle1_rx",
    "l_thumb0_ry", "l_thumb0_rz", "l_thumb1_rx", "l_thumb1_ry", "l_thumb1_rz",
    "l_thumb2_rz", "l_thumb3_rz",
    "l_index1_ry", "l_ring1_ry", "l_pinky1_ry", "l_middle1_ry",
    "l_index1_rz", "l_index2_rz", "l_index3_rz",
    "l_middle1_rz", "l_middle2_rz", "l_middle3_rz",
    "l_ring1_rz", "l_ring2_rz", "l_ring3_rz",
    "l_pinky1_rz", "l_pinky2_rz", "l_pinky3_rz",
    "l_index1_rx", "l_ring1_rx", "l_pinky1_rx", "l_middle1_rx",
    "l_foot_ry_flexible", "l_subtalar_rz_flexible",
    "l_talocrural_rx_flexible", "l_ball_rx_flexible",
    "r_foot_ry_flexible", "r_subtalar_rz_flexible",
    "r_talocrural_rx_flexible", "r_ball_rx_flexible",
    "spine_length_flexible", "neck_length_flexible",
    "shoulder_width_flexible", "arm_length_flexible",
    "hip_width_flexible", "leg_length_flexible",
    "scale_eye_width", "scale_eye_height", "scale_eye_depth",
    "scale_spine_length", "scale_neck_length", "scale_shoulder_width",
    "scale_uparms", "scale_lowarms", "scale_r_hands", "scale_l_hands",
    "scale_hip_width", "scale_hip_height", "scale_hip_depth",
    "scale_uplegs", "scale_lowlegs", "scale_knee_knock",
    "scale_ankle_height", "scale_foot_length",
    "scale_r_index1_length", "scale_r_middle1_length",
    "scale_r_ring1_length", "scale_r_pinky1_length", "scale_r_thumb1_length",
    "scale_r_index1_offset", "scale_r_middle1_offset",
    "scale_r_ring1_offset", "scale_r_pinky1_offset", "scale_r_thumb1_offset",
    "scale_r_index2_length", "scale_r_middle2_length",
    "scale_r_ring2_length", "scale_r_pinky2_length", "scale_r_thumb2_length",
    "scale_r_index3_length", "scale_r_middle3_length",
    "scale_r_ring3_length", "scale_r_pinky3_length", "scale_r_thumb3_length",
    "scale_r_index_null_tx", "scale_r_middle_null_tx",
    "scale_r_ring_null_tx", "scale_r_pinky_null_tx", "scale_r_thumb_null_tx",
    "scale_l_index1_length", "scale_l_middle1_length",
    "scale_l_ring1_length", "scale_l_pinky1_length", "scale_l_thumb1_length",
    "scale_l_index1_offset", "scale_l_middle1_offset",
    "scale_l_ring1_offset", "scale_l_pinky1_offset", "scale_l_thumb1_offset",
    "scale_l_index2_length", "scale_l_middle2_length",
    "scale_l_ring2_length", "scale_l_pinky2_length", "scale_l_thumb2_length",
    "scale_l_index3_length", "scale_l_middle3_length",
    "scale_l_ring3_length", "scale_l_pinky3_length", "scale_l_thumb3_length",
    "scale_l_index_null_tx", "scale_l_middle_null_tx",
    "scale_l_ring_null_tx", "scale_l_pinky_null_tx", "scale_l_thumb_null_tx",
]

# Preset poses for quick exploration
PRESET_POSES = {
    "T-Pose (Default)": {},
    "Arms Down": {
        "r_uparm_ry": -1.2,
        "l_uparm_ry": -1.2,
    },
    "Arms Forward": {
        "r_uparm_rz": 1.2,
        "l_uparm_rz": 1.2,
    },
    "Squat": {
        "r_knee_bend": 1.0,
        "l_knee_bend": 1.0,
        "r_upleg_ry": 0.6,
        "l_upleg_ry": 0.6,
    },
    "Wave Right Hand": {
        "r_uparm_rz": -1.5,
        "r_elbow_bend": -1.2,
    },
    "Head Turn Left": {
        "head_twist": 0.6,
    },
    "Lean Forward": {
        "spine_bend0": 0.4,
        "spine_bend1": 0.4,
    },
}


# ---------------------------------------------------------------------------
# Body-region grouping  (joint index → human-readable region label)
# Each of the 127 skeleton joints is mapped to one of ~18 body regions.
# ---------------------------------------------------------------------------
JOINT_TO_REGION = {
    # Root / Hips
    0: "Hips", 1: "Hips",
    # Left leg
    2: "Left Upper Leg", 13: "Left Upper Leg", 14: "Left Upper Leg",
    15: "Left Upper Leg", 16: "Left Upper Leg", 17: "Left Upper Leg",
    3: "Left Lower Leg", 9: "Left Lower Leg", 10: "Left Lower Leg",
    11: "Left Lower Leg", 12: "Left Lower Leg",
    4: "Left Foot", 5: "Left Foot", 6: "Left Foot",
    7: "Left Foot", 8: "Left Foot",
    # Right leg
    18: "Right Upper Leg", 29: "Right Upper Leg", 30: "Right Upper Leg",
    31: "Right Upper Leg", 32: "Right Upper Leg", 33: "Right Upper Leg",
    19: "Right Lower Leg", 25: "Right Lower Leg", 26: "Right Lower Leg",
    27: "Right Lower Leg", 28: "Right Lower Leg",
    20: "Right Foot", 21: "Right Foot", 22: "Right Foot",
    23: "Right Foot", 24: "Right Foot",
    # Spine / Torso
    34: "Torso", 35: "Torso", 36: "Torso", 37: "Torso",
    # Right arm
    38: "Right Shoulder",
    39: "Right Upper Arm", 69: "Right Upper Arm", 70: "Right Upper Arm",
    71: "Right Upper Arm", 72: "Right Upper Arm", 73: "Right Upper Arm",
    40: "Right Lower Arm", 41: "Right Lower Arm",
    65: "Right Lower Arm", 66: "Right Lower Arm",
    67: "Right Lower Arm", 68: "Right Lower Arm",
    42: "Right Hand",
    43: "Right Hand", 44: "Right Hand", 45: "Right Hand", 46: "Right Hand", 47: "Right Hand",
    48: "Right Hand", 49: "Right Hand", 50: "Right Hand", 51: "Right Hand",
    52: "Right Hand", 53: "Right Hand", 54: "Right Hand", 55: "Right Hand",
    56: "Right Hand", 57: "Right Hand", 58: "Right Hand", 59: "Right Hand",
    60: "Right Hand", 61: "Right Hand", 62: "Right Hand", 63: "Right Hand", 64: "Right Hand",
    # Left arm
    74: "Left Shoulder",
    75: "Left Upper Arm", 105: "Left Upper Arm", 106: "Left Upper Arm",
    107: "Left Upper Arm", 108: "Left Upper Arm", 109: "Left Upper Arm",
    76: "Left Lower Arm", 77: "Left Lower Arm",
    101: "Left Lower Arm", 102: "Left Lower Arm",
    103: "Left Lower Arm", 104: "Left Lower Arm",
    78: "Left Hand",
    79: "Left Hand", 80: "Left Hand", 81: "Left Hand", 82: "Left Hand", 83: "Left Hand",
    84: "Left Hand", 85: "Left Hand", 86: "Left Hand", 87: "Left Hand",
    88: "Left Hand", 89: "Left Hand", 90: "Left Hand", 91: "Left Hand",
    92: "Left Hand", 93: "Left Hand", 94: "Left Hand", 95: "Left Hand",
    96: "Left Hand", 97: "Left Hand", 98: "Left Hand", 99: "Left Hand", 100: "Left Hand",
    # Neck & Head
    110: "Neck", 111: "Neck", 112: "Neck",
    113: "Head", 126: "Head",
    114: "Jaw", 115: "Jaw", 116: "Jaw",
    117: "Jaw", 118: "Jaw", 119: "Jaw", 120: "Jaw", 121: "Jaw",
    122: "Right Eye", 123: "Right Eye",
    124: "Left Eye", 125: "Left Eye",
}

# Distinct colours for each body region (CSS rgba strings)
REGION_COLORS = {
    "Hips":             "#e6194b",
    "Torso":            "#3cb44b",
    "Left Upper Leg":   "#ffe119",
    "Left Lower Leg":   "#f58231",
    "Left Foot":        "#911eb4",
    "Right Upper Leg":  "#42d4f4",
    "Right Lower Leg":  "#f032e6",
    "Right Foot":       "#bfef45",
    "Left Shoulder":    "#fabed4",
    "Left Upper Arm":   "#469990",
    "Left Lower Arm":   "#dcbeff",
    "Left Hand":        "#9a6324",
    "Right Shoulder":   "#fffac8",
    "Right Upper Arm":  "#800000",
    "Right Lower Arm":  "#aaffc3",
    "Right Hand":       "#808000",
    "Neck":             "#ffd8b1",
    "Head":             "#000075",
    "Jaw":              "#a9a9a9",
    "Right Eye":        "#ffffff",
    "Left Eye":         "#e6beff",
}

ALL_REGIONS = list(REGION_COLORS.keys())


# ===================================================================
# Model loading (cached so it only loads once per session)
# ===================================================================
@st.cache_resource(show_spinner="Loading MHR model …")
def load_model():
    """Load the TorchScript model, mesh topology, and UV data."""
    # Faces from precomputed numpy file
    faces_path = MHR_ASSETS / "lod1_faces.npy"
    if not faces_path.exists():
        # Fallback: load from FBX via pymomentum
        import pymomentum.geometry as pym_geometry
        fbx_path = str(MHR_ASSETS / "lod1.fbx")
        model_path = str(MHR_ASSETS / "compact_v6_1.model")
        character = pym_geometry.Character.load_fbx(
            fbx_path, model_path, load_blendshapes=True
        )
        faces = character.mesh.faces.copy()
    else:
        faces = np.load(str(faces_path))

    # TorchScript model
    ts_path = MHR_ASSETS / "mhr_model.pt"
    assert ts_path.exists(), f"TorchScript model not found at {ts_path}"
    scripted_model = torch.jit.load(str(ts_path), map_location="cpu")
    scripted_model.eval()

    # UV data (pre-extracted from FBX)
    texcoords_path = MHR_ASSETS / "lod1_texcoords.npy"
    texcoord_faces_path = MHR_ASSETS / "lod1_texcoord_faces.npy"
    uv_to_mesh_path = MHR_ASSETS / "lod1_uv_to_mesh.npy"
    if texcoords_path.exists():
        texcoords = np.load(str(texcoords_path))        # (19455, 2)
        texcoord_faces = np.load(str(texcoord_faces_path))  # (36874, 3)
        uv_to_mesh = np.load(str(uv_to_mesh_path))     # (19455,)
    else:
        texcoords = None
        texcoord_faces = None
        uv_to_mesh = None

    # Skinning data → per-vertex body-region label
    dominant_joint_path = MHR_ASSETS / "lod1_dominant_joint.npy"
    joint_names_path = MHR_ASSETS / "lod1_joint_names.npy"
    if dominant_joint_path.exists():
        dominant_joint = np.load(str(dominant_joint_path))   # (18439,)
        joint_names_arr = np.load(str(joint_names_path))     # (127,)
    else:
        dominant_joint = None
        joint_names_arr = None

    return scripted_model, faces, texcoords, texcoord_faces, uv_to_mesh, dominant_joint, joint_names_arr


def compute_mesh(model, identity_tuple, pose_tuple, expression_tuple):
    """Run the MHR forward pass and return vertices (N,3) as numpy."""
    identity = torch.tensor([identity_tuple], dtype=torch.float32)
    pose = torch.tensor([pose_tuple], dtype=torch.float32)
    expression = torch.tensor([expression_tuple], dtype=torch.float32)
    with torch.no_grad():
        verts, _ = model(identity, pose, expression)
    return verts[0].numpy()


# ===================================================================
# Plotly visualisation helpers
# ===================================================================
def _get_region_label(joint_idx: int) -> str:
    """Return the body-region label for a given joint index."""
    return JOINT_TO_REGION.get(int(joint_idx), "Unknown")


def _build_vertex_region_colors(dominant_joint: np.ndarray) -> tuple[list[str], list[str]]:
    """Return (region_labels, hex_colors) arrays aligned with vertex indices."""
    labels = [_get_region_label(j) for j in dominant_joint]
    colors = [REGION_COLORS.get(lbl, "#888888") for lbl in labels]
    return labels, colors


# ===================================================================
# Selection save / load helpers
# ===================================================================
def _sanitize_filename(name: str) -> str:
    """Turn a human-readable name into a safe filename stem."""
    safe = "".join(c if (c.isalnum() or c in "_- ") else "_" for c in name).strip()
    safe = safe.replace(" ", "_")
    return safe or "untitled"


def list_saved_selections() -> list[Path]:
    """Return a sorted list of .json selection files."""
    return sorted(SELECTIONS_DIR.glob("*.json"), key=lambda p: p.stem)


def save_selection(
    name: str,
    selected_verts: set[int],
    selected_tris: set[int] | None = None,
    dominant_joint: np.ndarray | None = None,
) -> Path:
    """Persist a selection to *selections/<name>.json* and return the path."""
    # Gather region breakdown
    region_summary: dict[str, int] = {}
    if dominant_joint is not None:
        for vi in selected_verts:
            lbl = _get_region_label(dominant_joint[vi])
            region_summary[lbl] = region_summary.get(lbl, 0) + 1

    payload = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "vertex_count": len(selected_verts),
        "triangle_count": len(selected_tris) if selected_tris else 0,
        "vertices": sorted(selected_verts),
        "triangles": sorted(selected_tris) if selected_tris else [],
        "region_summary": dict(sorted(region_summary.items())),
    }
    path = SELECTIONS_DIR / f"{_sanitize_filename(name)}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_selection(path: Path) -> dict:
    """Load a selection JSON file and return the parsed dict."""
    return json.loads(path.read_text())


def delete_selection(path: Path) -> None:
    """Remove a saved selection file."""
    if path.exists():
        path.unlink()


def build_mesh_figure(
    verts: np.ndarray,
    faces: np.ndarray,
    selected_verts: set | None = None,
    selected_tris: set | None = None,
    show_faces: bool = True,
    show_wireframe: bool = False,
    show_vertices: bool = True,
    color_mode: str = "height",
    point_size: int = 3,
    dominant_joint: np.ndarray | None = None,
    visible_regions: set[str] | None = None,
):
    """Build an interactive Plotly 3D figure of the mesh."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # ---- Filter faces by visible regions --------------------------------
    if visible_regions is not None and dominant_joint is not None:
        vert_visible = np.array([
            _get_region_label(dominant_joint[vi]) in visible_regions
            for vi in range(len(verts))
        ])
        face_mask = vert_visible[faces[:, 0]] & vert_visible[faces[:, 1]] & vert_visible[faces[:, 2]]
        vis_faces = faces[face_mask]
    else:
        vert_visible = None
        vis_faces = faces

    i, j, k = vis_faces[:, 0], vis_faces[:, 1], vis_faces[:, 2]

    # ---- colour the mesh ------------------------------------------------
    use_region_color = (color_mode == "body region" and dominant_joint is not None)

    if use_region_color:
        region_labels, region_hex = _build_vertex_region_colors(dominant_joint)
        intensity = None
        colorscale = None
        # Mesh3d vertexcolor expects a list of css colour strings
        vertex_colors = region_hex
    elif color_mode == "height":
        intensity = z
        colorscale = "Viridis"
    elif color_mode == "depth":
        intensity = y
        colorscale = "Plasma"
    elif color_mode == "x-axis":
        intensity = x
        colorscale = "RdBu"
    else:
        intensity = np.linalg.norm(verts, axis=1)
        colorscale = "Cividis"

    # Vertex index hover text
    if use_region_color:
        hover_text = [
            f"Vertex {idx}<br>{region_labels[idx]}<br>"
            f"x={x[idx]:.2f}  y={y[idx]:.2f}  z={z[idx]:.2f}"
            for idx in range(len(x))
        ]
    else:
        hover_text = [f"Vertex {idx}<br>x={x[idx]:.2f}  y={y[idx]:.2f}  z={z[idx]:.2f}" for idx in range(len(x))]

    fig = go.Figure()

    # Main mesh
    mesh_kwargs = dict(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.85,
        flatshading=True,
        lighting=dict(
            ambient=0.5,
            diffuse=0.7,
            specular=0.3,
            roughness=0.4,
            fresnel=0.2,
        ),
        lightposition=dict(x=100, y=200, z=300),
        hoverinfo="skip",
        name="mesh",
    )
    if use_region_color:
        mesh_kwargs["vertexcolor"] = vertex_colors
    else:
        mesh_kwargs["intensity"] = intensity
        mesh_kwargs["colorscale"] = colorscale
    if show_faces:
        fig.add_trace(go.Mesh3d(**mesh_kwargs))

    # Wireframe overlay
    if show_wireframe:
        edge_x, edge_y, edge_z = [], [], []
        for face in vis_faces:
            for s, e in [(0, 1), (1, 2), (2, 0)]:
                edge_x += [x[face[s]], x[face[e]], None]
                edge_y += [y[face[s]], y[face[e]], None]
                edge_z += [z[face[s]], z[face[e]], None]
        fig.add_trace(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode="lines",
                line=dict(color="rgba(80,80,80,0.3)", width=1),
                hoverinfo="skip",
                name="wireframe",
            )
        )

    # Vertex scatter (on top for picking)
    if show_vertices:
        if use_region_color:
            vert_colors = np.array(region_hex, dtype=object)
        else:
            vert_colors = np.full(len(x), "rgba(100,100,255,0.4)", dtype=object)
        vert_sizes = np.full(len(x), point_size, dtype=float)
        # Hide vertices of filtered-out regions
        if vert_visible is not None:
            for vi in range(len(x)):
                if not vert_visible[vi]:
                    vert_colors[vi] = "rgba(0,0,0,0)"
                    vert_sizes[vi] = 0
        if selected_verts:
            for vi in selected_verts:
                if 0 <= vi < len(x):
                    vert_colors[vi] = "rgba(255,50,50,1.0)"
                    vert_sizes[vi] = point_size + 6

        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=vert_sizes,
                    color=vert_colors.tolist(),
                ),
                text=hover_text,
                hoverinfo="text",
                name="vertices",
                customdata=np.arange(len(x)),
            )
        )

    # Highlight selected triangles
    if selected_tris:
        for tri_idx in selected_tris:
            if 0 <= tri_idx < len(faces):
                f = faces[tri_idx]
                tri_x = [x[f[0]], x[f[1]], x[f[2]], x[f[0]]]
                tri_y = [y[f[0]], y[f[1]], y[f[2]], y[f[0]]]
                tri_z = [z[f[0]], z[f[1]], z[f[2]], z[f[0]]]
                fig.add_trace(
                    go.Scatter3d(
                        x=tri_x, y=tri_y, z=tri_z,
                        mode="lines+markers",
                        line=dict(color="yellow", width=5),
                        marker=dict(size=6, color="yellow"),
                        hoverinfo="text",
                        text=[
                            f"Tri {tri_idx} · v{f[0]}",
                            f"Tri {tri_idx} · v{f[1]}",
                            f"Tri {tri_idx} · v{f[2]}",
                            "",
                        ],
                        name=f"tri_{tri_idx}",
                    )
                )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X",
                backgroundcolor="#1e1e1e",
                gridcolor="#444444",
                zerolinecolor="#555555",
                color="#cccccc",
            ),
            yaxis=dict(
                title="Y",
                backgroundcolor="#1e1e1e",
                gridcolor="#444444",
                zerolinecolor="#555555",
                color="#cccccc",
            ),
            zaxis=dict(
                title="Z",
                backgroundcolor="#1e1e1e",
                gridcolor="#444444",
                zerolinecolor="#555555",
                color="#cccccc",
            ),
            aspectmode="data",
            dragmode="turntable",
            camera=dict(
                eye=dict(x=0, y=-2.0, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=750,
        showlegend=False,
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font_color="white",
    )
    return fig


def build_uv_figure(
    texcoords: np.ndarray,
    texcoord_faces: np.ndarray,
    uv_to_mesh: np.ndarray,
    selected_verts: set | None = None,
    point_size: int = 3,
    dominant_joint: np.ndarray | None = None,
    color_mode: str = "height",
    visible_regions: set[str] | None = None,
    show_legend: bool = False,
    show_faces: bool = True,
    show_vertices: bool = True,
):
    """Build an interactive Plotly 2D figure of the UV layout.

    This is a flat (U, V) scatter + wireframe that supports box/lasso
    selection in Plotly 2D mode.  Selected UV vertices are mapped back to
    mesh vertex indices through *uv_to_mesh*.
    """
    u = texcoords[:, 0]
    v = texcoords[:, 1]

    # Filter UV faces by visible body regions
    if visible_regions is not None and dominant_joint is not None:
        uv_vert_visible = np.array([
            _get_region_label(dominant_joint[uv_to_mesh[ui]]) in visible_regions
            for ui in range(len(u))
        ])
        face_mask = (
            uv_vert_visible[texcoord_faces[:, 0]]
            & uv_vert_visible[texcoord_faces[:, 1]]
            & uv_vert_visible[texcoord_faces[:, 2]]
        )
        vis_tc_faces = texcoord_faces[face_mask]
    else:
        uv_vert_visible = None
        vis_tc_faces = texcoord_faces

    use_region_color = (color_mode == "body region" and dominant_joint is not None)

    # Pre-compute region colours per UV vertex (via uv_to_mesh → dominant_joint)
    if use_region_color:
        mesh_region_labels, mesh_region_hex = _build_vertex_region_colors(dominant_joint)
        uv_region_labels = [mesh_region_labels[uv_to_mesh[i]] for i in range(len(u))]
        uv_region_hex = [mesh_region_hex[uv_to_mesh[i]] for i in range(len(u))]

    # Build the set of UV indices that correspond to the selected mesh verts
    # (reverse lookup: mesh_vert -> UV verts)
    selected_uv_set: set[int] = set()
    if selected_verts:
        from collections import defaultdict
        mesh_to_uv: dict[int, list[int]] = defaultdict(list)
        for uv_idx, mesh_idx in enumerate(uv_to_mesh):
            mesh_to_uv[int(mesh_idx)].append(uv_idx)
        for mv in selected_verts:
            selected_uv_set.update(mesh_to_uv.get(mv, []))

    fig = go.Figure()

    # ---- Filled triangles (face display) --------------------------------
    if not show_faces:
        pass  # skip faces entirely
    elif use_region_color:
        # Group closed-triangle paths by body region → filled + outlined
        region_tris: dict[str, tuple[list, list]] = {r: ([], []) for r in REGION_COLORS}
        for face in vis_tc_faces:
            mesh_vi = uv_to_mesh[face[0]]
            region = _get_region_label(dominant_joint[mesh_vi])
            tu, tv = region_tris.get(region, ([], []))
            # Closed triangle path + None separator
            tu += [float(u[face[0]]), float(u[face[1]]), float(u[face[2]]), float(u[face[0]]), None]
            tv += [float(v[face[0]]), float(v[face[1]]), float(v[face[2]]), float(v[face[0]]), None]
        for region, (tu, tv) in region_tris.items():
            if tu:
                hex_c = REGION_COLORS[region]
                r_val, g_val, b_val = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
                fig.add_trace(
                    go.Scattergl(
                        x=tu, y=tv,
                        mode="lines",
                        fill="toself",
                        fillcolor=f"rgba({r_val},{g_val},{b_val},0.35)",
                        line=dict(color=hex_c, width=0.5),
                        hoverinfo="skip",
                        name=region,
                        showlegend=show_legend and use_region_color,
                        legendgroup=region,
                    )
                )
    else:
        tri_u: list[float | None] = []
        tri_v: list[float | None] = []
        for face in vis_tc_faces:
            tri_u += [float(u[face[0]]), float(u[face[1]]), float(u[face[2]]), float(u[face[0]]), None]
            tri_v += [float(v[face[0]]), float(v[face[1]]), float(v[face[2]]), float(v[face[0]]), None]
        fig.add_trace(
            go.Scattergl(
                x=tri_u, y=tri_v,
                mode="lines",
                fill="toself",
                fillcolor="rgba(100,100,255,0.15)",
                line=dict(color="rgba(100,100,255,0.25)", width=0.5),
                hoverinfo="skip",
                name="uv_faces",
                showlegend=False,
            )
        )

    # ---- Vertex scatter -----------------------------------------------
    if show_vertices:
        if use_region_color:
            hover_text = [
                f"UV {idx}  →  Mesh v{uv_to_mesh[idx]}<br>{uv_region_labels[idx]}<br>"
                f"u={u[idx]:.4f}  v={v[idx]:.4f}"
                for idx in range(len(u))
            ]
        else:
            hover_text = [
                f"UV {idx}  →  Mesh v{uv_to_mesh[idx]}<br>u={u[idx]:.4f}  v={v[idx]:.4f}"
                for idx in range(len(u))
            ]

        if use_region_color:
            vert_colors = np.array(uv_region_hex, dtype=object)
        else:
            vert_colors = np.full(len(u), "rgba(100,100,255,0.5)", dtype=object)
        vert_sizes = np.full(len(u), point_size, dtype=float)
        # Hide UV vertices of filtered-out regions
        if uv_vert_visible is not None:
            for ui in range(len(u)):
                if not uv_vert_visible[ui]:
                    vert_colors[ui] = "rgba(0,0,0,0)"
                    vert_sizes[ui] = 0
        if selected_uv_set:
            for ui in selected_uv_set:
                if 0 <= ui < len(u):
                    vert_colors[ui] = "rgba(255,50,50,1.0)"
                    vert_sizes[ui] = point_size + 4

        fig.add_trace(
            go.Scattergl(
                x=u, y=v,
                mode="markers",
                marker=dict(
                    size=vert_sizes,
                    color=vert_colors.tolist(),
                ),
                text=hover_text,
                hoverinfo="text",
                name="uv_vertices",
                showlegend=False,
                # customdata carries the *mesh* vertex index for each UV point
                customdata=uv_to_mesh.tolist(),
            )
        )

    fig.update_layout(
        xaxis=dict(
            title="U",
            range=[-0.02, 1.02],
            scaleanchor="y",
            scaleratio=1,
            gridcolor="#333333",
            zerolinecolor="#555555",
            color="#cccccc",
        ),
        yaxis=dict(
            title="V",
            range=[-0.02, 1.02],
            gridcolor="#333333",
            zerolinecolor="#555555",
            color="#cccccc",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=750,
        showlegend=show_legend and use_region_color,
        legend=dict(
            title="Body Region",
            bgcolor="rgba(30,30,30,0.85)",
            bordercolor="#555555",
            borderwidth=1,
            font=dict(size=11, color="white"),
            itemsizing="constant",
        ),
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font_color="white",
        dragmode="lasso",  # default to lasso for easy selection
    )
    return fig


# ===================================================================
# Sidebar – parameter controls
# ===================================================================
def sidebar_controls():
    """Render sidebar controls and return parameter arrays."""
    st.sidebar.title("🦴 MHR Controls")

    # ---- Preset poses ------------------------------------------------
    st.sidebar.header("Quick Presets")
    preset_name = st.sidebar.selectbox(
        "Load preset pose", list(PRESET_POSES.keys()), index=0
    )
    preset = PRESET_POSES[preset_name]

    if st.sidebar.button("🔄 Reset all to zero"):
        # Clear all parameter keys from session state
        for key in list(st.session_state.keys()):
            if key.startswith("param_") or key.startswith("id_") or key.startswith("expr_"):
                del st.session_state[key]
        st.rerun()

    # ---- LOD selection ------------------------------------------------
    st.sidebar.header("Model Settings")
    st.sidebar.info(
        "Using TorchScript LOD-1 model\n\n"
        f"• 18 439 vertices\n"
        f"• 36 874 triangles\n"
        f"• 127 joints"
    )

    # ---- Identity parameters (body shape) ----------------------------
    st.sidebar.header("Identity (Body Shape)")
    id_expand = st.sidebar.expander("Identity coefficients (45)", expanded=False)
    identity = np.zeros(NUM_IDENTITY)
    with id_expand:
        col1, col2, col3 = st.columns(3)
        labels = [
            *[f"Body shape {i}" for i in range(20)],
            *[f"Head shape {i}" for i in range(20)],
            *[f"Hand shape {i}" for i in range(5)],
        ]
        for idx in range(NUM_IDENTITY):
            target_col = [col1, col2, col3][idx % 3]
            with target_col:
                identity[idx] = st.slider(
                    labels[idx],
                    min_value=-3.0,
                    max_value=3.0,
                    value=0.0,
                    step=0.05,
                    key=f"id_{idx}",
                )

    # ---- Pose parameters --------------------------------------------
    st.sidebar.header("Pose Parameters")
    pose = np.zeros(NUM_MODEL_PARAMS)
    for group_name, indices in POSE_GROUPS.items():
        with st.sidebar.expander(f"{group_name} ({len(indices)} params)", expanded=False):
            for idx in indices:
                pname = ALL_PARAM_NAMES[idx] if idx < len(ALL_PARAM_NAMES) else f"param_{idx}"
                default_val = preset.get(pname, 0.0)
                # Determine range based on param type
                if "scale" in pname:
                    lo, hi = -2.0, 2.0
                elif pname.startswith("root_t"):
                    lo, hi = -100.0, 100.0
                else:
                    lo, hi = -3.14, 3.14
                pose[idx] = st.slider(
                    pname,
                    min_value=lo,
                    max_value=hi,
                    value=float(default_val),
                    step=0.01,
                    key=f"param_{idx}",
                )

    # ---- Expression parameters --------------------------------------
    st.sidebar.header("Facial Expression")
    expr_expand = st.sidebar.expander("Expression coefficients (72)", expanded=False)
    expression = np.zeros(NUM_EXPRESSION)
    with expr_expand:
        for idx in range(NUM_EXPRESSION):
            expression[idx] = st.slider(
                f"Expression {idx}",
                min_value=-1.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                key=f"expr_{idx}",
            )

    return identity, pose, expression


# ===================================================================
# Main application
# ===================================================================
def main():
    st.title("🧍 MHR Human Mesh Viewer")
    st.caption(
        "Interactive viewer for Meta's Momentum Human Rig (MHR). "
        "Adjust pose, identity, and expression in the sidebar. "
        "Click vertices in the 3D view to inspect indices and coordinates."
    )

    # Load model
    model, faces, texcoords, texcoord_faces, uv_to_mesh, dominant_joint, joint_names_arr = load_model()

    # Sidebar → parameter arrays
    identity, pose, expression = sidebar_controls()

    with st.expander("Top-row display options", expanded=False):
        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
        with opt_col1:
            show_faces = st.checkbox("Show faces", value=True)
            show_wireframe = st.checkbox("Show wireframe", value=False)
            show_vertices = st.checkbox("Show vertices", value=True)
            show_uv_legend = st.checkbox("Show UV color legend", value=False)
        with opt_col2:
            color_options = ["height", "depth", "x-axis", "distance"]
            if dominant_joint is not None:
                color_options.insert(0, "body region")
            color_mode = st.selectbox("Color by", color_options, index=0)
        with opt_col3:
            point_size = st.slider("Vertex point size",0.0, 5.0, 1.0, step=0.25)
        with opt_col4:
            st.metric("Vertices", f"{faces.max() + 1:,}")

        # Region visibility filter
        if dominant_joint is not None:
            visible_regions = st.multiselect(
                "Visible body regions",
                options=ALL_REGIONS,
                default=ALL_REGIONS,
                key="visible_regions",
                help="Deselect regions to hide them from both the 3D and UV views.",
            )
            visible_regions_set: set[str] | None = set(visible_regions) if len(visible_regions) < len(ALL_REGIONS) else None
        else:
            visible_regions_set = None

        # ------------------------------------------------------------------
        # Compute mesh – stored in session_state, recomputed when params change
        # ------------------------------------------------------------------
        current_params = (
            tuple(identity.tolist()),
            tuple(pose.tolist()),
            tuple(expression.tolist()),
        )

        if (
            "mesh_verts" not in st.session_state
            or st.session_state.get("_last_params") != current_params
        ):
            st.session_state.mesh_verts = compute_mesh(model, *current_params)
            st.session_state._last_params = current_params

        verts = st.session_state.mesh_verts

        # ------------------------------------------------------------------
        # Selection state
        # ------------------------------------------------------------------
        if "selected_verts" not in st.session_state:
            st.session_state.selected_verts = set()
        if "selected_tris" not in st.session_state:
            st.session_state.selected_tris = set()

        # ------------------------------------------------------------------
        # Vertex / Triangle lookup
        # ------------------------------------------------------------------
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            st.subheader("🔍 Vertex Lookup")
            vert_input = st.text_input(
                "Enter vertex indices (comma-separated)",
                placeholder="e.g. 0, 100, 5000, 12345",
                key="vert_input",
            )
            if vert_input.strip():
                try:
                    new_verts = {
                        int(v.strip())
                        for v in vert_input.split(",")
                        if v.strip().lstrip("-").isdigit()
                    }
                    st.session_state.selected_verts = new_verts
                except ValueError:
                    st.error("Please enter valid comma-separated integers.")

        with sel_col2:
            st.subheader("🔺 Triangle Lookup")
            tri_input = st.text_input(
                "Enter triangle indices (comma-separated)",
                placeholder="e.g. 0, 500, 10000",
                key="tri_input",
            )
            if tri_input.strip():
                try:
                    new_tris = {
                        int(t.strip())
                        for t in tri_input.split(",")
                        if t.strip().lstrip("-").isdigit()
                    }
                    st.session_state.selected_tris = new_tris
                except ValueError:
                    st.error("Please enter valid comma-separated integers.")

        # ------------------------------------------------------------------
        # Vertex range selector for region exploration
        # ------------------------------------------------------------------
        st.subheader("📐 Vertex Range Selector")
        range_col1, range_col2, range_col3 = st.columns([2, 2, 1])
        with range_col1:
            range_start = st.number_input("Start vertex", min_value=0, max_value=len(verts) - 1, value=0, step=100)
        with range_col2:
            range_end = st.number_input("End vertex", min_value=0, max_value=len(verts) - 1, value=min(100, len(verts) - 1), step=100)
        with range_col3:
            if st.button("🎯 Highlight Range"):
                st.session_state.selected_verts = set(range(int(range_start), int(range_end) + 1))
                st.rerun()

        # ------------------------------------------------------------------
        # Interaction mode toggle
        # ------------------------------------------------------------------
        mode_col1, mode_col2 = st.columns([1, 4])
        with mode_col1:
            interaction_mode = st.radio(
                "Interaction mode",
                ["🔄 Navigate", "🎯 Select"],
                index=0,
                horizontal=True,
                help=(
                    "**Navigate** – drag to rotate, scroll to zoom, shift-drag to pan.\n\n"
                    "**Select** – box/lasso select vertices in the 3D view."
                ),
            )
        select_mode = interaction_mode.startswith("🎯")

    # ------------------------------------------------------------------
    # 3D plot + 2D UV map side-by-side
    # ------------------------------------------------------------------
    with st.expander("Whole body Mesh View", expanded=True):
        fig = build_mesh_figure(
            verts,
            faces,
            selected_verts=st.session_state.selected_verts,
            selected_tris=st.session_state.selected_tris,
            show_faces=show_faces,
            show_wireframe=show_wireframe,
            show_vertices=show_vertices,
            color_mode=color_mode,
            point_size=point_size,
            dominant_joint=dominant_joint,
            visible_regions=visible_regions_set,
        )

        has_uv = texcoords is not None

        if has_uv:
            plot_col_3d, plot_col_uv = st.columns(2)
        else:
            plot_col_3d = st.container()

        # ---- 3D mesh plot ----
        with plot_col_3d:
            st.markdown("**3D Mesh View**")
            # In Navigate mode we do NOT pass on_select so Plotly keeps its native
            # turntable-rotate / pan / zoom interactions.  In Select mode we enable
            # on_select so the user can box-select or lasso vertices.
            if select_mode:
                selected_point = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="mesh_plot_select",
                    on_select="rerun",
                )
            else:
                selected_point = None
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="mesh_plot_nav",
                    config=dict(scrollZoom=True),
                )

        # ---- 2D UV map plot (always has lasso/box select) ----
        uv_selected_point = None
        if has_uv:
            with plot_col_uv:
                st.markdown("**2D UV Map** — *lasso / box-select here to pick vertices*")
                uv_fig = build_uv_figure(
                    texcoords,
                    texcoord_faces,
                    uv_to_mesh,
                    selected_verts=st.session_state.selected_verts,
                    point_size=point_size,
                    dominant_joint=dominant_joint,
                    color_mode=color_mode,
                    visible_regions=visible_regions_set,
                    show_legend=show_uv_legend,
                    show_faces=show_faces,
                    show_vertices=show_vertices,
                )
                uv_selected_point = st.plotly_chart(
                    uv_fig,
                    use_container_width=True,
                    key="uv_plot",
                    on_select="rerun",
                )

    # Process selection from 3D plot (only in Select mode)
    if selected_point and selected_point.selection and selected_point.selection.points:
        points = selected_point.selection.points
        clicked_verts = set()
        for pt in points:
            # The customdata carries vertex index from the scatter trace
            if pt.get("customdata") is not None:
                clicked_verts.add(int(pt["customdata"][0]) if isinstance(pt["customdata"], list) else int(pt["customdata"]))
            elif "point_number" in pt and pt.get("curve_number") == 1 + int(show_wireframe):
                # vertices trace
                clicked_verts.add(pt["point_number"])
        if clicked_verts:
            st.session_state.selected_verts = st.session_state.selected_verts | clicked_verts

    # Process selection from 2D UV plot → map back to mesh vertices
    if uv_selected_point and uv_selected_point.selection and uv_selected_point.selection.points:
        uv_points = uv_selected_point.selection.points
        uv_clicked_mesh_verts = set()
        for pt in uv_points:
            if pt.get("customdata") is not None:
                cd = pt["customdata"]
                mesh_vi = int(cd[0]) if isinstance(cd, list) else int(cd)
                if mesh_vi >= 0:
                    uv_clicked_mesh_verts.add(mesh_vi)
        if uv_clicked_mesh_verts:
            st.session_state.selected_verts = st.session_state.selected_verts | uv_clicked_mesh_verts

    # ------------------------------------------------------------------
    # Selected-region sub-mesh visualization (3D + UV side-by-side)
    # ------------------------------------------------------------------
    if st.session_state.selected_verts and len(st.session_state.selected_verts) >= 3:
        with st.expander("🔎 Selected Region Mesh View", expanded=True):
            sel_set = st.session_state.selected_verts

            # --- build sub-mesh: keep faces that have ALL 3 verts in selection ---
            sel_face_mask = np.array([
                (int(f[0]) in sel_set and int(f[1]) in sel_set and int(f[2]) in sel_set)
                for f in faces
            ])
            sub_face_indices = np.where(sel_face_mask)[0]  # original face indices

            if len(sub_face_indices) == 0:
                st.info("No complete triangles found within the selected vertices. "
                        "Try selecting a larger contiguous region.")
            else:
                sub_faces_orig = faces[sub_face_indices]  # (M, 3) with original vert indices

                # Collect the unique original vertex indices actually used by these faces
                used_orig = np.unique(sub_faces_orig.ravel())
                # Map: original vert index → new compact index
                orig_to_compact = {int(ov): ci for ci, ov in enumerate(used_orig)}
                sub_verts = verts[used_orig]  # (K, 3)
                sub_faces = np.array([
                    [orig_to_compact[int(v)] for v in f] for f in sub_faces_orig
                ])
                # Dominant joint for sub-verts
                sub_dominant_joint = dominant_joint[used_orig] if dominant_joint is not None else None

                # --- Sub-mesh 3D figure ---
                sub_fig_3d = build_mesh_figure(
                    sub_verts,
                    sub_faces,
                    selected_verts=None,  # all shown, no need to re-highlight
                    selected_tris=None,
                    show_faces=show_faces,
                    show_wireframe=True,
                    show_vertices=show_vertices,
                    color_mode=color_mode,
                    point_size=max(point_size, 2.0),
                    dominant_joint=sub_dominant_joint,
                    visible_regions=visible_regions_set,
                )

                # --- Sub-mesh UV figure ---
                sub_uv_fig = None
                if has_uv:
                    # Find UV faces that correspond to the selected mesh faces
                    sub_tc_faces_orig = texcoord_faces[sub_face_indices]
                    used_uv_orig = np.unique(sub_tc_faces_orig.ravel())
                    uv_orig_to_compact = {int(ov): ci for ci, ov in enumerate(used_uv_orig)}
                    sub_texcoords = texcoords[used_uv_orig]
                    sub_tc_faces = np.array([
                        [uv_orig_to_compact[int(v)] for v in f] for f in sub_tc_faces_orig
                    ])
                    sub_uv_to_mesh_compact = np.array([
                        orig_to_compact.get(int(uv_to_mesh[oi]), 0)
                        for oi in used_uv_orig
                    ])
                    # Also need a uv_to_mesh mapping using ORIGINAL mesh indices
                    # (for customdata so selection still works)
                    sub_uv_to_mesh_orig = uv_to_mesh[used_uv_orig]

                    sub_uv_fig = build_uv_figure(
                        sub_texcoords,
                        sub_tc_faces,
                        sub_uv_to_mesh_orig,
                        selected_verts=None,
                        point_size=max(point_size, 2.0),
                        dominant_joint=dominant_joint,  # full array — uv_to_mesh_orig indexes into it
                        color_mode=color_mode,
                        visible_regions=visible_regions_set,
                        show_legend=show_uv_legend,
                        show_faces=show_faces,
                        show_vertices=show_vertices,
                    )

                st.write(
                    f"**Sub-mesh:** {len(sub_verts):,} vertices, "
                    f"{len(sub_faces):,} triangles "
                    f"(from {len(sel_set):,} selected vertices)"
                )

                if sub_uv_fig is not None:
                    sub_col_3d, sub_col_uv = st.columns(2)
                else:
                    sub_col_3d = st.container()

                with sub_col_3d:
                    st.markdown("**Selected Region — 3D**")
                    st.plotly_chart(
                        sub_fig_3d,
                        use_container_width=True,
                        key="sub_mesh_3d",
                        config=dict(scrollZoom=True),
                    )

                if sub_uv_fig is not None:
                    with sub_col_uv:
                        st.markdown("**Selected Region — UV**")
                        sub_uv_sel = st.plotly_chart(
                            sub_uv_fig,
                            use_container_width=True,
                            key="sub_uv_plot",
                            on_select="rerun",
                        )
                        # Process lasso/box selection on the sub UV plot
                        if sub_uv_sel and sub_uv_sel.selection and sub_uv_sel.selection.points:
                            sub_uv_new = set()
                            for pt in sub_uv_sel.selection.points:
                                if pt.get("customdata") is not None:
                                    cd = pt["customdata"]
                                    mv = int(cd[0]) if isinstance(cd, list) else int(cd)
                                    if mv >= 0:
                                        sub_uv_new.add(mv)
                            if sub_uv_new:
                                st.session_state.selected_verts = (
                                    st.session_state.selected_verts | sub_uv_new
                                )

    # ------------------------------------------------------------------
    # Selection info panel
    # ------------------------------------------------------------------
    if st.session_state.selected_verts or st.session_state.selected_tris:

        with st.expander("Selection Details", expanded=True):
            st.subheader("📋 Selection Details")
            info_col1, info_col2 = st.columns(2)

            with info_col1:
                if st.session_state.selected_verts:
                    sorted_verts = sorted(st.session_state.selected_verts)
                    n_sel = len(sorted_verts)
                    st.write(f"**Selected vertices:** {n_sel}")
                    if n_sel <= 500:
                        # Build a table
                        rows = []
                        for vi in sorted_verts:
                            if 0 <= vi < len(verts):
                                rows.append({
                                    "Vertex Index": vi,
                                    "X": f"{verts[vi, 0]:.4f}",
                                    "Y": f"{verts[vi, 1]:.4f}",
                                    "Z": f"{verts[vi, 2]:.4f}",
                                })
                        st.dataframe(rows, use_container_width=True, height=300)
                    else:
                        st.write(f"Indices: {sorted_verts[:20]} … (showing first 20 of {n_sel})")

                    # Centroid of selection
                    sel_verts_arr = verts[sorted_verts]
                    centroid = sel_verts_arr.mean(axis=0)
                    st.write(f"**Centroid:** ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")

                    # Copy-friendly list
                    st.code(", ".join(str(v) for v in sorted_verts), language=None)

                    if st.button("Clear vertex selection"):
                        st.session_state.selected_verts = set()
                        st.rerun()

            with info_col2:
                if st.session_state.selected_tris:
                    sorted_tris = sorted(st.session_state.selected_tris)
                    st.write(f"**Selected triangles:** {len(sorted_tris)}")
                    rows = []
                    for ti in sorted_tris:
                        if 0 <= ti < len(faces):
                            f = faces[ti]
                            rows.append({
                                "Triangle Index": ti,
                                "Vertex A": f[0],
                                "Vertex B": f[1],
                                "Vertex C": f[2],
                            })
                    st.dataframe(rows, use_container_width=True, height=300)

                    # Copy-friendly
                    st.code(", ".join(str(t) for t in sorted_tris), language=None)

                    if st.button("Clear triangle selection"):
                        st.session_state.selected_tris = set()
                        st.rerun()

    # ------------------------------------------------------------------
    # Anatomical region finder
    # ------------------------------------------------------------------
    with st.expander("Anatomical Region Finder", expanded=False):
        st.subheader("🦴 Anatomical Region Finder")
        st.write(
            "Search vertices by spatial region. The MHR model is in **centimetres**, "
            "Z-up, Y-forward. Use the sliders to define a 3D bounding box and find "
            "all vertices inside it."
        )
        region_col1, region_col2, region_col3 = st.columns(3)
        with region_col1:
            x_range = st.slider("X range (left/right)", float(verts[:, 0].min()), float(verts[:, 0].max()),
                                (float(verts[:, 0].min()), float(verts[:, 0].max())), step=1.0)
        with region_col2:
            y_range = st.slider("Y range (front/back)", float(verts[:, 1].min()), float(verts[:, 1].max()),
                                (float(verts[:, 1].min()), float(verts[:, 1].max())), step=1.0)
        with region_col3:
            z_range = st.slider("Z range (up/down)", float(verts[:, 2].min()), float(verts[:, 2].max()),
                                (float(verts[:, 2].min()), float(verts[:, 2].max())), step=1.0)

        if st.button("🎯 Find vertices in region"):
            mask = (
                (verts[:, 0] >= x_range[0]) & (verts[:, 0] <= x_range[1]) &
                (verts[:, 1] >= y_range[0]) & (verts[:, 1] <= y_range[1]) &
                (verts[:, 2] >= z_range[0]) & (verts[:, 2] <= z_range[1])
            )
            region_verts = set(np.where(mask)[0].tolist())
            st.session_state.selected_verts = region_verts
            st.write(f"Found **{len(region_verts)}** vertices in the selected region.")
            st.rerun()

    # ------------------------------------------------------------------
    # Saved Selections
    # ------------------------------------------------------------------
    with st.expander("💾 Saved Selections", expanded=False):
        save_col, load_col = st.columns(2)

        # ---- Save ----
        with save_col:
            st.markdown("**Save current selection**")
            sel_name = st.text_input(
                "Selection name",
                placeholder="e.g. left_arm_front, head_region_v2",
                key="save_sel_name",
            )
            if st.button("💾 Save selection", disabled=not st.session_state.selected_verts):
                if not sel_name.strip():
                    st.error("Please enter a name for the selection.")
                elif not st.session_state.selected_verts:
                    st.warning("No vertices selected.")
                else:
                    out_path = save_selection(
                        sel_name.strip(),
                        st.session_state.selected_verts,
                        st.session_state.selected_tris or None,
                        dominant_joint,
                    )
                    st.success(f"Saved **{len(st.session_state.selected_verts)}** vertices → `{out_path.relative_to(SCRIPT_DIR)}`")
                    st.rerun()

        # ---- Load / Delete ----
        with load_col:
            st.markdown("**Load a saved selection**")
            saved_files = list_saved_selections()
            if not saved_files:
                st.info("No saved selections yet.")
            else:
                file_labels = []
                for sf in saved_files:
                    try:
                        meta = json.loads(sf.read_text())
                        label = f"{meta.get('name', sf.stem)}  ({meta.get('vertex_count', '?')} verts)"
                    except Exception:
                        label = sf.stem
                    file_labels.append(label)

                chosen_idx = st.selectbox(
                    "Saved selections",
                    range(len(saved_files)),
                    format_func=lambda i: file_labels[i],
                    key="load_sel_idx",
                )
                chosen_path = saved_files[chosen_idx]

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("📂 Load"):
                        data = load_selection(chosen_path)
                        st.session_state.selected_verts = set(data.get("vertices", []))
                        st.session_state.selected_tris = set(data.get("triangles", []))
                        st.rerun()
                with btn_col2:
                    if st.button("➕ Merge into current"):
                        data = load_selection(chosen_path)
                        st.session_state.selected_verts |= set(data.get("vertices", []))
                        st.session_state.selected_tris |= set(data.get("triangles", []))
                        st.rerun()
                with btn_col3:
                    if st.button("🗑️ Delete", type="secondary"):
                        delete_selection(chosen_path)
                        st.rerun()

                # Preview the selected file
                try:
                    preview = load_selection(chosen_path)
                    with st.container():
                        st.caption(f"Created: {preview.get('created', 'unknown')}")
                        region_summary = preview.get("region_summary", {})
                        if region_summary:
                            st.markdown("**Region breakdown:**")
                            summary_rows = [{"Region": r, "Vertices": c} for r, c in region_summary.items()]
                            st.dataframe(summary_rows, use_container_width=True, height=200)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    with st.expander("Export Options", expanded=False):
        st.subheader("💾 Export")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("Export current mesh as OBJ"):
                lines = ["# MHR mesh export\n"]
                for v in verts:
                    lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for f in faces:
                    lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
                obj_data = "".join(lines)
                st.download_button(
                    "⬇ Download OBJ",
                    data=obj_data,
                    file_name="mhr_mesh.obj",
                    mime="text/plain",
                )
        with export_col2:
            if st.session_state.selected_verts:
                sel_data = ", ".join(str(v) for v in sorted(st.session_state.selected_verts))
                st.download_button(
                    "⬇ Download selected vertex indices",
                    data=sel_data,
                    file_name="selected_vertices.txt",
                    mime="text/plain",
                )


if __name__ == "__main__":
    main()
