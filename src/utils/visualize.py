from pathlib import Path
from copy import deepcopy

import plotly.graph_objects as go

import numpy as np
from ase.atoms import Atoms
from ase.data.colors import jmol_colors
from ase.data import covalent_radii


# update the colors and radii for the dummy atom
jmol_colors[0] = (0.5, 0.5, 0.5)  # X
jmol_colors[25] = (0.5, 1, 0.4)  # Mn
covalent_radii[0] = 0.5


class Visualizer:
    def __init__(
        self,
        atoms_list: list[Atoms] = None,
        atomic_size: float = 0.8,
        opacity: float = 1,
        resolution: int = 19,
        layout_kwargs: dict = None,
    ):
        if atoms_list is None:
            raise ValueError("atoms_list must be provided.")
        self.atoms_list = atoms_list
        self.atomic_size = atomic_size
        self.opacity = opacity
        self.resolution = resolution

        self.fig = None
        self.index = 0
        self.num_frames = len(atoms_list)

        self.default_layout = self._default_layout(layout_kwargs)

    def _create_sphere(self, radius, center, resolution):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        return x, y, z

    def _default_layout(self, layout_kwargs):
        layout = go.Layout(
            width=400,
            height=400,
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-2, y=0, z=0),
                    projection=dict(type="orthographic"),
                ),
                aspectmode="data",
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
        )
        # update layout_kwargs
        if layout_kwargs is not None:
            layout.update(layout_kwargs)
        return layout

    def _make_frame(self, atoms: Atoms, frame_idx: int):
        positions = atoms.get_positions()
        species = atoms.get_chemical_symbols()
        atomic_colors = [jmol_colors[n] for n in atoms.numbers]
        atomic_radii = [covalent_radii[n] * self.atomic_size for n in atoms.numbers]
        x, y, z = zip(*positions)

        fig = go.Figure()
        for xi, yi, zi, color, radius, sp in zip(
            x, y, z, atomic_colors, atomic_radii, species
        ):
            color_rgb = f"rgb{tuple(int(c * 255) for c in color)}"
            sphere_x, sphere_y, sphere_z = self._create_sphere(
                radius, (xi, yi, zi), self.resolution
            )
            fig.add_trace(
                go.Mesh3d(
                    x=sphere_x.flatten(),
                    y=sphere_y.flatten(),
                    z=sphere_z.flatten(),
                    color=color_rgb,
                    opacity=self.opacity,
                    alphahull=0,
                    hoverinfo="text",
                    hovertext=f"{sp}",
                    lighting=dict(
                        ambient=0.70,
                    ),
                    lightposition=dict(x=1000, y=1000, z=0),
                )
            )

        # Dynamically create unit cell lines for the current frame
        a, b, c = atoms.cell
        lines = [
            [[0, 0, 0], a],
            [[0, 0, 0], b],
            [[0, 0, 0], c],
            [a, a + b],
            [a, a + c],
            [b, b + a],
            [b, b + c],
            [c, c + a],
            [c, c + b],
            [a + b, a + b + c],
            [a + c, a + c + b],
            [b + c, b + c + a],
        ]
        line_traces = []
        for line in lines:
            x_values, y_values, z_values = zip(*line)
            line_trace = go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
            line_traces.append(line_trace)
        fig.add_traces(line_traces)

        # frame_layout for view_trajectory
        frame_layout = go.Layout(
            title_text=f"Time = {frame_idx+1}",
            title_x=0.5,
            title_y=0.1,
            title_xanchor="center",
        )
        return go.Frame(
            data=fig.data,
            layout=frame_layout,
        )

    def view(self, index: int = 0):
        frame = self._make_frame(self.atoms_list[index], index)
        self.fig = go.Figure(data=frame.data, layout=self.default_layout)
        return self.fig

    def view_trajectory(self, duration: float = 0.01):
        frames = [
            self._make_frame(atoms, idx) for idx, atoms in enumerate(self.atoms_list)
        ]

        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": duration,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {
                                    "duration": duration,
                                    "easing": "quadratic-in-out",
                                },
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ]

        layout = deepcopy(self.default_layout)
        layout.update(updatemenus=updatemenus)
        self.fig = go.Figure(
            data=frames[0].data,
            layout=layout,
            frames=frames,
        )

        return self.fig

    def save_html(self, save_path: str = "trajectory.html"):
        save_path = Path(save_path)
        if self.fig is None:
            raise ValueError(
                "You must call view_trajectory() before saving the trajectory."
            )
        self.fig.write_html(save_path)
        print(f"Saved trajectory to {save_path}")
