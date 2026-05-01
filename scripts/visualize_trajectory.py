"""Render a window of a PLaTITO trajectory as an animated GIF.

Dependencies (beyond the base PLaTITO install):
    pip install Pillow pulchra
    conda install -c conda-forge pymol-open-source

Usage:
    python scripts/visualize_trajectory.py \
        --coords outputs/my_protein/generated_coords.pt \
        --pdb    /path/to/start.pdb \
        --out    outputs/my_protein/trajectory.gif \
        --stride 5 --fps 10
"""

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import mdtraj as md
import pymol2
import torch
from PIL import Image
from pulchra.pulchra import Pulchra
from pulchra.io.pdb_writer import write_pdb

from platito.utils.amino_acid_vocab import AA_3_TO_1

_DSSP_TO_PYMOL = {"H": "H", "E": "S", "C": "L"}


def rebuild_and_load(pm, xyz_nm, seq, obj, tmpdir, pulchra_obj):
    pdb_tmp = os.path.join(tmpdir, f"{obj}.pdb")
    mol = pulchra_obj.reconstruct_coords(xyz_nm * 10.0, seq)
    write_pdb(pdb_tmp, mol)
    pm.cmd.load(pdb_tmp, obj)
    traj = md.load(pdb_tmp)
    dssp = "".join(md.compute_dssp(traj, simplified=True)[0])
    for resi, ss in enumerate(dssp, start=1):
        pm.cmd.alter(f"{obj} and resi {resi}", "ss=" + repr(_DSSP_TO_PYMOL.get(ss, "L")))
    pm.cmd.rebuild()
    pm.cmd.show("cartoon", obj)
    pm.cmd.set_color("proteina_helix", [0.25, 0.55, 0.85])  # cornflower blue matching overview
    pm.cmd.set_color("proteina_sheet", [0.85, 0.15, 0.15])  # red
    pm.cmd.set_color("proteina_loop",  [0.55, 0.55, 0.55])  # medium gray
    pm.cmd.color("proteina_helix", f"{obj} and ss h")
    pm.cmd.color("proteina_sheet", f"{obj} and ss s")
    pm.cmd.color("proteina_loop",  f"{obj} and not (ss h or ss s)")
    pm.cmd.set("cartoon_transparency", 0.5, f"{obj} and not (ss h or ss s)")


def render_gif(frames_nm, seq, out_path, fps, width, height, tmpdir,
               hold_first_s=0, hold_last_s=0, hold_pre_outro_s=0,
               outro_zoom_frames=0, outro_rotation_frames=0):
    pulchra_obj = Pulchra(
        verbose=False, ca_optimize=False, rebuild_backbone=True,
        rebuild_sidechains=False, optimize_xvol=False, optimize_hbonds=False,
    )
    pil_frames = []
    n_traj = len(frames_nm)
    n_total = n_traj + outro_zoom_frames + outro_rotation_frames

    with pymol2.PyMOL() as pm:
        pm.cmd.bg_color("white")
        pm.cmd.set("cartoon_sampling",      16)
        pm.cmd.set("cartoon_oval_length",   1)
        pm.cmd.set("cartoon_oval_width",    0.1)
        pm.cmd.set("cartoon_fancy_helices", 1)
        pm.cmd.set("cartoon_round_helices", 1)
        pm.cmd.set("cartoon_smooth_loops",  1)
        pm.cmd.set("cartoon_discrete_colors", 1)
        pm.cmd.set("light_count",           5)
        pm.cmd.set("spec_count",            1)
        pm.cmd.set("shininess",             15)
        pm.cmd.set("specular",              0.6)
        pm.cmd.set("ambient",               0)
        pm.cmd.set("direct",                0)
        pm.cmd.set("reflect",               1.5)
        pm.cmd.set("ray_shadow_decay_factor", 0.1)
        pm.cmd.set("ray_shadow_decay_range",  2)
        pm.cmd.set("ray_shadows",           1)
        pm.cmd.set("orthoscopic",           1)
        pm.cmd.set("depth_cue",             0)
        pm.cmd.set("ray_trace_mode",        0)

        # ref_align (first frame) defines the shared coordinate system.
        # ref_view (last/folded frame) is aligned into that system first, then
        # used to orient the camera — so the view angle shows the folded state
        # as it will actually appear in the movie.
        rebuild_and_load(pm, frames_nm[0],  seq, "ref_align", tmpdir, pulchra_obj)
        rebuild_and_load(pm, frames_nm[-1], seq, "ref_view",  tmpdir, pulchra_obj)
        pm.cmd.align("ref_view", "ref_align")
        pm.cmd.disable("ref_align")
        pm.cmd.disable("ref_view")
        pm.cmd.orient("ref_view")
        pm.cmd.zoom("ref_view", 40)
        view = pm.cmd.get_view()

        # --- trajectory frames ---
        for fi, xyz in enumerate(frames_nm):
            rebuild_and_load(pm, xyz, seq, "fr", tmpdir, pulchra_obj)
            pm.cmd.align("fr", "ref_align")
            pm.cmd.set_view(view)
            pm.cmd.hide("lines")
            png = os.path.join(tmpdir, f"f{fi:05d}.png")
            pm.cmd.ray(width, height)
            pm.cmd.png(png, quiet=1)
            pm.cmd.delete("fr")
            pil_frames.append(Image.open(png).convert("RGB"))
            print(f"  rendered {len(pil_frames)}/{n_total}", end="\r", flush=True)

        # --- outro: zoom-in then 360° rotation of the folded state ---
        if outro_zoom_frames > 0 or outro_rotation_frames > 0:
            rebuild_and_load(pm, frames_nm[-1], seq, "outro", tmpdir, pulchra_obj)
            pm.cmd.align("outro", "ref_align")

            # tight_view: same orientation as the trajectory view but zoomed in fully
            pm.cmd.set_view(view)
            pm.cmd.zoom("outro", 0)
            tight_view = list(pm.cmd.get_view())
            traj_view  = list(view)

            # zoom-in transition: linearly interpolate from trajectory view to tight view
            for i in range(outro_zoom_frames):
                t = (i + 1) / outro_zoom_frames
                interp = tuple(traj_view[j] * (1 - t) + tight_view[j] * t for j in range(18))
                pm.cmd.set_view(interp)
                pm.cmd.hide("lines")
                png = os.path.join(tmpdir, f"oz{i:03d}.png")
                pm.cmd.ray(width, height)
                pm.cmd.png(png, quiet=1)
                pil_frames.append(Image.open(png).convert("RGB"))
                print(f"  rendered {len(pil_frames)}/{n_total}", end="\r", flush=True)

            # 360° turntable rotation at tight zoom
            pm.cmd.set_view(tuple(tight_view))
            step = 360.0 / max(outro_rotation_frames, 1)
            for i in range(outro_rotation_frames):
                pm.cmd.turn("y", step)
                pm.cmd.hide("lines")
                png = os.path.join(tmpdir, f"or{i:03d}.png")
                pm.cmd.ray(width, height)
                pm.cmd.png(png, quiet=1)
                pil_frames.append(Image.open(png).convert("RGB"))
                print(f"  rendered {len(pil_frames)}/{n_total}", end="\r", flush=True)

    frame_ms = int(1000 / fps)
    has_outro = outro_zoom_frames > 0 or outro_rotation_frames > 0
    durations = []
    for i in range(len(pil_frames)):
        if i == 0:
            durations.append(max(frame_ms, int(hold_first_s * 1000)))
        elif i == n_traj - 1 and has_outro:
            durations.append(max(frame_ms, int(hold_pre_outro_s * 1000)))
        elif i == len(pil_frames) - 1:
            durations.append(max(frame_ms, int(hold_last_s * 1000)))
        else:
            durations.append(frame_ms)

    print(f"\nSaving → {out_path}")
    pil_frames[0].save(
        out_path, save_all=True, append_images=pil_frames[1:],
        loop=0, duration=durations,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", required=True, help="generated_coords.pt  [1, N, L, 3] nm")
    parser.add_argument("--pdb",    required=True, help="Reference PDB (for sequence)")
    parser.add_argument("--out",    required=True, help="Output path (.gif)")
    parser.add_argument("--start",  type=int, default=0, help="First frame (default 0)")
    parser.add_argument("--end",    type=int, default=None, help="Last frame exclusive (default: end of trajectory)")
    parser.add_argument("--stride", type=int, default=1,  help="Frame stride (default 1)")
    parser.add_argument("--fps",                  type=int,   default=10, help="GIF fps (default 10)")
    parser.add_argument("--hold-first",           type=float, default=0,  help="Seconds to hold the first frame (default 0)")
    parser.add_argument("--hold-last",            type=float, default=0,  help="Seconds to hold the last frame (default 0)")
    parser.add_argument("--hold-pre-outro",       type=float, default=0,  help="Seconds to hold last trajectory frame before outro (default 0)")
    parser.add_argument("--outro-zoom-frames",     type=int,   default=0,  help="Frames to zoom in after trajectory (default 0)")
    parser.add_argument("--outro-rotation-frames", type=int,   default=0,  help="Frames for 360° rotation at end (default 0)")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    coords  = torch.load(args.coords, map_location="cpu", weights_only=False)
    traj_nm = coords.squeeze(0).numpy()  # [N_frames, L, 3]
    L       = traj_nm.shape[1]

    end           = args.end or traj_nm.shape[0]
    frame_indices = list(range(args.start, end, args.stride))
    frames_nm     = traj_nm[frame_indices]

    traj_ref = md.load(args.pdb)
    seq      = "".join(AA_3_TO_1.get(r.name, "X") for r in traj_ref.topology.residues)[:L]

    print(f"Trajectory   : {traj_nm.shape[0]} frames, {L} residues")
    print(f"Sequence     : {seq}")
    print(f"Window       : [{args.start}:{end}:{args.stride}]  →  {len(frame_indices)} frames")
    print(f"Duration     : {len(frame_indices)/args.fps:.1f} s at {args.fps} fps")

    with tempfile.TemporaryDirectory() as tmpdir:
        render_gif(frames_nm, seq, args.out, args.fps, args.width, args.height, tmpdir,
                   hold_first_s=args.hold_first, hold_last_s=args.hold_last,
                   hold_pre_outro_s=args.hold_pre_outro,
                   outro_zoom_frames=args.outro_zoom_frames,
                   outro_rotation_frames=args.outro_rotation_frames)

    meta = {
        "coords":       str(Path(args.coords).resolve()),
        "pdb":          str(Path(args.pdb).resolve()),
        "out":          str(Path(args.out).resolve()),
        "start":        args.start,
        "end":          end,
        "stride":       args.stride,
        "n_frames":     len(frame_indices),
        "fps":          args.fps,
        "duration_s":   round(len(frame_indices) / args.fps + args.hold_first + args.hold_last, 2),
        "hold_first_s": args.hold_first,
        "hold_last_s":  args.hold_last,
        "width":        args.width,
        "height":       args.height,
        "sequence":     seq,
        "created":      datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Metadata     → {meta_path}")
    print(f"\nDone → {args.out}")


if __name__ == "__main__":
    main()
