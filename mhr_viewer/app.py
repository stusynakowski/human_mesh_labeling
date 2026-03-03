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
from pathlib import Path

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
        "r_uparm_rz": -1.2,
        "l_uparm_rz": 1.2,
    },
    "Arms Forward": {
        "r_uparm_ry": -1.2,
        "l_uparm_ry": -1.2,
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


# ===================================================================
# Model loading (cached so it only loads once per session)
# ===================================================================
@st.cache_resource(show_spinner="Loading MHR model …")
def load_model():
    """Load the TorchScript model and mesh topology."""
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
    return scripted_model, faces


@st.cache_data(show_spinner="Computing mesh …")
def compute_mesh(
    _model,  # leading underscore → unhashable arg ignored by st.cache_data
    identity_tuple: tuple,
    pose_tuple: tuple,
    expression_tuple: tuple,
):
    """Run the MHR forward pass and return vertices (N,3) as numpy."""
    identity = torch.tensor([identity_tuple], dtype=torch.float32)
    pose = torch.tensor([pose_tuple], dtype=torch.float32)
    expression = torch.tensor([expression_tuple], dtype=torch.float32)
    with torch.no_grad():
        verts, _ = _model(identity, pose, expression)
    return verts[0].numpy()


# ===================================================================
# Plotly visualisation helpers
# ===================================================================
def build_mesh_figure(
    verts: np.ndarray,
    faces: np.ndarray,
    selected_verts: set | None = None,
    selected_tris: set | None = None,
    show_wireframe: bool = False,
    color_mode: str = "height",
    point_size: int = 3,
):
    """Build an interactive Plotly 3D figure of the mesh."""
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    # ---- colour the mesh ------------------------------------------------
    if color_mode == "height":
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
    hover_text = [f"Vertex {idx}<br>x={x[idx]:.2f}  y={y[idx]:.2f}  z={z[idx]:.2f}" for idx in range(len(x))]

    fig = go.Figure()

    # Main mesh
    fig.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            intensity=intensity,
            colorscale=colorscale,
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
    )

    # Wireframe overlay
    if show_wireframe:
        edge_x, edge_y, edge_z = [], [], []
        for face in faces:
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

    # Vertex scatter (always on top for picking)
    vert_colors = np.full(len(x), "rgba(100,100,255,0.4)", dtype=object)
    vert_sizes = np.full(len(x), point_size, dtype=float)
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
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0, y=-2.0, z=0.5),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=750,
        showlegend=False,
        paper_bgcolor="#1e1e1e",
        font_color="white",
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
    model, faces = load_model()

    # Sidebar → parameter arrays
    identity, pose, expression = sidebar_controls()

    # Top-row display options
    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
    with opt_col1:
        show_wireframe = st.checkbox("Show wireframe", value=False)
    with opt_col2:
        color_mode = st.selectbox("Color by", ["height", "depth", "x-axis", "distance"], index=0)
    with opt_col3:
        point_size = st.slider("Vertex point size", 1, 10, 3)
    with opt_col4:
        st.metric("Vertices", f"{faces.max() + 1:,}")

    # Compute mesh
    verts = compute_mesh(
        model,
        tuple(identity.tolist()),
        tuple(pose.tolist()),
        tuple(expression.tolist()),
    )

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
    # 3D plot
    # ------------------------------------------------------------------
    fig = build_mesh_figure(
        verts,
        faces,
        selected_verts=st.session_state.selected_verts,
        selected_tris=st.session_state.selected_tris,
        show_wireframe=show_wireframe,
        color_mode=color_mode,
        point_size=point_size,
    )

    # Use plotly_events-style click capture via on_select
    selected_point = st.plotly_chart(
        fig,
        use_container_width=True,
        key="mesh_plot",
        on_select="rerun",
    )

    # Process click selection from plotly
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

    # ------------------------------------------------------------------
    # Selection info panel
    # ------------------------------------------------------------------
    if st.session_state.selected_verts or st.session_state.selected_tris:
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
    # Export
    # ------------------------------------------------------------------
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
