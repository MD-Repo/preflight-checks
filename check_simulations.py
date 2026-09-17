#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "mdanalysis"]
# ///
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-09-16
Purpose: Check simulation directories of any engine before submitting them

`check_amber.py` came first and reads NetCDF only. That turned out to cover
very little even of AMBER: of AMBER's 115,922 trajectory replicates on record,
4,023 are `.nc` -- about 3.5%. The rest are `.mdc` and `.xtc`, and the archive
as a whole is mostly `.xtc` and `.mdc` from SPONGE, ACEMD and GROMACS. This
reads XTC, TRR, DCD and NetCDF through MDAnalysis, which opens all of them and
needs a topology for none of them.

`check_amber.py` is kept as it is. Nothing here replaces it in place.

  ATOM MISMATCH -- the trajectory's atom count against the topology's.
  Stripping solvent after a simulation and then shipping the ORIGINAL topology
  beside the stripped trajectory produces a pair no tool can read. The topology
  is read on its own rather than paired with the trajectory, because pairing
  them is exactly what fails when the counts disagree.

  EMPTY TRAJECTORY -- a file holding no frames at all. Usually an interrupted
  copy or a job that never wrote output. They carry a valid checksum and a
  correct manifest, so nothing upstream notices.

  BAD UNIT CELL -- frames whose box is not a box: a length or an angle that is
  not finite, a length of zero or less, or an angle outside 0 to 180 degrees.
  Two submissions have arrived this way, one with a box of exact zeros and one
  with values like 1.46e+233. A frame like this can stall the conversion step
  for hours rather than fail it.

  ZERO FRAMES -- frames in which every coordinate of every atom is exactly 0.0,
  the signature of a partially written file. They survive conversion and then
  break the analysis stage.

  BAD COORDINATES -- frames whose coordinates are not positions: NaN, infinity,
  or a number too large to be one. The size test is not redundant. XTC cannot
  store a NaN: writing one saturates its integer encoding and the value reads
  back as 21,474,836 angstroms, finite and just as impossible. Anything that
  has been through an XTC therefore carries the damage as an ordinary-looking
  number.

  NO TIME AXIS -- NetCDF only, and deliberately so. Frame spacing cannot be
  read from a NetCDF with no `time` variable, so MDRepo cannot derive the
  duration; declare `sampling_frequency_ps` in mdrepo-metadata.toml instead.
  This check is NOT extended to the other formats, because it cannot be done
  honestly there: MDAnalysis reports 0, 1, 2, 3 ... ps for a trajectory with no
  time information at all, which is indistinguishable from a real 1 ps/frame
  run. XTC and TRR always store a time per frame, so the question does not
  arise for them. For DCD it arises differently, which is the next check.

  IMPLAUSIBLE TIMESTEP -- DCD only. A DCD carries its spacing in the header,
  and a file written by some tools carries a placeholder there instead, which
  reads back as ~4.9e-5 ps per frame. Anything below 1e-4 ps is reported. This
  is the one check here with no counterpart in check_amber.py.

Directories are found by looking for `mdrepo-metadata.toml`, so both a single
simulation directory and a parent holding many of them work as arguments.

Read-only: never writes into a simulation directory.
"""

import argparse
import os
import sys
import tomllib
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple

try:
    import numpy as np
except ImportError as err:
    sys.exit(
        f"{err}\n\n"
        "This needs numpy, scipy and MDAnalysis. Easiest is:\n\n"
        "    uv run check_simulations.py --help\n\n"
        "which fetches them itself, or install them into a virtual "
        "environment and run this script with its python.\n"
    )

# MDAnalysis is loud on import and noisier still on some readers. None of it
# is the submitter's problem.
warnings.filterwarnings("ignore")

# Frames are read in blocks so a large trajectory never lands in memory whole.
# Only the NetCDF fast path uses this; the MDAnalysis path is frame at a time.
CHUNK_FRAMES = 256

# A coordinate this large is not a position. It is 0.1 mm, where a simulation
# box is a few hundred angstroms at most, and it sits 21x below the value an
# XTC saturates a NaN to.
MAX_ABS_COORD = 1.0e6

# Below this, a DCD's declared spacing is a placeholder rather than a number.
FLOOR_PS = 1.0e-4

NETCDF_SUFFIXES = (".nc", ".netcdf")
MDANALYSIS_SUFFIXES = (".xtc", ".trr", ".dcd", ".nc", ".netcdf")


class Args(NamedTuple):
    """Command-line arguments"""

    dirs: List[str]
    threads: int
    headers_only: bool
    report: Optional[str]
    quiet: bool


class Finding(NamedTuple):
    """One trajectory's verdict"""

    directory: str
    trajectory: str
    topology: str
    traj_atoms: Optional[int]
    top_atoms: Optional[int]
    frames: Optional[int]
    zero_frames: int
    zero_runs: List[Tuple[int, int]]
    bad_cell_frames: int
    bad_cell_runs: List[Tuple[int, int]]
    bad_coord_frames: int
    bad_coord_runs: List[Tuple[int, int]]
    timestep_ps: Optional[float]
    read_error: Optional[str]
    problems: List[str]
    error: Optional[str]


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Check simulation directories before submitting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dirs",
        metavar="DIR",
        nargs="+",
        help="Simulation directories, or a parent containing them",
    )

    parser.add_argument(
        "-j",
        "--threads",
        metavar="int",
        type=int,
        default=4,
        help="Parallel workers",
    )

    parser.add_argument(
        "--headers-only",
        action="store_true",
        help="Skip the frame scans; NetCDF still has its unit cell checked",
    )

    parser.add_argument(
        "-r",
        "--report",
        metavar="FILE",
        help="Write a plain-text report of the problems found",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress per-directory progress",
    )

    args = parser.parse_args()

    if args.threads < 1:
        parser.error(f"--threads must be positive, not {args.threads}")

    return Args(
        dirs=args.dirs,
        threads=args.threads,
        headers_only=args.headers_only,
        report=args.report,
        quiet=args.quiet,
    )


# --------------------------------------------------
def find_dirs(paths: List[str]) -> List[str]:
    """Expand each argument to the simulation directories under it"""

    found = []
    for path in paths:
        if os.path.isfile(os.path.join(path, "mdrepo-metadata.toml")):
            found.append(path)
            continue

        if not os.path.isdir(path):
            print(f'Skipping "{path}": not a directory', file=sys.stderr)
            continue

        for entry in sorted(os.scandir(path), key=lambda e: e.name):
            if entry.is_dir() and os.path.isfile(
                os.path.join(entry.path, "mdrepo-metadata.toml")
            ):
                found.append(entry.path)

    return found


# --------------------------------------------------
def group_runs(indexes: List[int]) -> List[Tuple[int, int]]:
    """Group ascending frame indexes into contiguous runs"""

    grouped: List[Tuple[int, int]] = []
    for idx in indexes:
        if grouped and idx == grouped[-1][1] + 1:
            grouped[-1] = (grouped[-1][0], idx)
        else:
            grouped.append((idx, idx))

    return grouped


# --------------------------------------------------
def scan_cell(lengths, angles) -> List[int]:
    """
    Find frames whose unit cell is not a usable box

    A length or an angle that is not finite, a length that is zero or
    negative, or an angle outside 0 to 180 degrees cannot describe a box.

    The one case that is not a defect is a trajectory with no periodic box at
    all, which some writers record as zero lengths in every frame. That is why
    the all-zero file returns early instead of reporting every frame: the
    signal we are after is a box that disappears partway through a file whose
    other frames have one.
    """

    lengths = np.asarray(lengths, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)

    if lengths.ndim != 2 or angles.ndim != 2:
        return []

    no_box = (lengths == 0).all(axis=1)
    if no_box.all():
        return []

    bad = ~np.isfinite(lengths).all(axis=1)
    bad |= ~np.isfinite(angles).all(axis=1)
    bad |= (lengths <= 0).any(axis=1)
    bad |= (angles <= 0).any(axis=1)
    bad |= (angles >= 180).any(axis=1)

    return [int(i) for i in np.where(bad)[0]]


# --------------------------------------------------
def classify_frame(coords) -> Optional[str]:
    """
    Say what is wrong with one frame's coordinates, if anything

    The three answers never overlap: a frame of exact zeros is finite, and a
    frame holding a value too large to be a position is neither.
    """

    coords = np.asarray(coords)

    if not np.isfinite(coords).all():
        return "nonfinite"
    if not coords.any():
        return "zero"
    if (np.abs(coords) >= MAX_ABS_COORD).any():
        return "huge"

    return None


# --------------------------------------------------
def scan_coordinates(coords) -> Tuple[List[int], List[int]]:
    """
    Scan NetCDF coordinates in blocks, without MDAnalysis

    NetCDF is read directly for the same reason the unit cell is: it is
    faster in blocks than frame at a time, it matches `check_amber.py` and
    the pipeline's own screen exactly, and MDAnalysis cannot read a NetCDF
    frame whose box is zero -- which a non-periodic trajectory legitimately
    has in every frame.

    Returns the all-zero frames and the frames that are not positions.
    """

    zero: List[int] = []
    bad: List[int] = []
    total = coords.shape[0]

    for start in range(0, total, CHUNK_FRAMES):
        block = np.asarray(coords[start : start + CHUNK_FRAMES])
        flat = block.reshape(block.shape[0], -1)

        finite = np.isfinite(flat).all(axis=1)
        for offset in np.where(~finite)[0]:
            bad.append(start + int(offset))
        for offset in np.where(finite & ~flat.any(axis=1))[0]:
            zero.append(start + int(offset))

        big = finite & (
            np.abs(np.where(finite[:, None], flat, 0.0)) >= MAX_ABS_COORD
        ).any(axis=1)
        for offset in np.where(big)[0]:
            bad.append(start + int(offset))

    return zero, sorted(bad)


# --------------------------------------------------
def netcdf_coordinates(path):
    """A NetCDF's zero frames and not-a-position frames"""

    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=True) as ncf:
        return scan_coordinates(ncf.variables["coordinates"])


# --------------------------------------------------
def netcdf_has_time(path: str) -> Optional[bool]:
    """
    Whether a NetCDF declares a time variable

    Read straight from the container rather than through MDAnalysis, which
    reports 0, 1, 2, 3 ... ps for a file with no time information at all --
    indistinguishable from a real 1 ps/frame trajectory. Returns None when the
    file cannot be inspected, which is not the same as "no time axis".
    """

    try:
        from scipy.io import netcdf_file

        with netcdf_file(path, "r", mmap=True) as ncf:
            return "time" in ncf.variables
    except Exception:
        return None


# --------------------------------------------------
def netcdf_cell(path):
    """
    A NetCDF's cell arrays, without touching its coordinates

    Three numbers per frame against tens of thousands, which is what makes the
    unit cell check affordable under --headers-only. Returns None when the
    file has no cell at all.
    """

    try:
        from scipy.io import netcdf_file

        with netcdf_file(path, "r", mmap=True) as ncf:
            if "cell_lengths" not in ncf.variables:
                return None
            return (
                np.array(ncf.variables["cell_lengths"][:]),
                np.array(ncf.variables["cell_angles"][:]),
            )
    except Exception:
        return None


# --------------------------------------------------
def count_atoms(path: str) -> int:
    """Atom count of a topology or a trajectory, read on its own"""

    import MDAnalysis as mda

    return len(mda.Universe(path).atoms)


# --------------------------------------------------
def netcdf_shape(path: str) -> Tuple[int, int]:
    """A NetCDF's atom count and frame count, from its dimensions"""

    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=True) as ncf:
        coords = ncf.variables["coordinates"]
        return int(coords.shape[1]), int(coords.shape[0])


# --------------------------------------------------
def check_netcdf(path: str, blank: dict, headers_only: bool) -> List[str]:
    """
    Check a NetCDF without MDAnalysis, and say what is wrong with it

    Deliberately not routed through MDAnalysis, for three reasons: reading in
    blocks is faster than frame at a time, it is the same code path as
    `check_amber.py` and the pipeline's own screen so the three cannot
    disagree, and MDAnalysis cannot read a NetCDF frame whose box is zero --
    which a non-periodic trajectory legitimately has in every frame.
    """

    problems: List[str] = []

    traj_atoms, frames = netcdf_shape(path)
    blank.update(traj_atoms=traj_atoms, frames=frames)

    if frames == 0:
        return ["EMPTY TRAJECTORY"]

    if netcdf_has_time(path) is False:
        problems.append("NO TIME AXIS")

    cell = netcdf_cell(path)
    bad_cell = scan_cell(*cell) if cell is not None else []

    zero_idx: List[int] = []
    bad_coord_idx: List[int] = []
    if not headers_only:
        zero_idx, bad_coord_idx = netcdf_coordinates(path)

    blank.update(
        bad_cell_frames=len(bad_cell),
        bad_cell_runs=group_runs(bad_cell),
        zero_frames=len(zero_idx),
        zero_runs=group_runs(zero_idx),
        bad_coord_frames=len(bad_coord_idx),
        bad_coord_runs=group_runs(bad_coord_idx),
    )

    if bad_cell:
        problems.append("BAD UNIT CELL")
    if zero_idx:
        problems.append("ZERO FRAMES")
    if bad_coord_idx:
        problems.append("BAD COORDINATES")

    return problems


# --------------------------------------------------
def check_via_mdanalysis(
    path: str, blank: dict, headers_only: bool, is_dcd: bool
) -> List[str]:
    """
    Check an XTC, TRR or DCD frame by frame

    None of the three needs a topology to open, so nothing here depends on
    the structure or topology file being readable.
    """

    import MDAnalysis as mda

    universe = mda.Universe(path)
    traj_atoms = len(universe.atoms)
    frames = len(universe.trajectory)
    blank.update(traj_atoms=traj_atoms, frames=frames)

    if frames == 0:
        return ["EMPTY TRAJECTORY"]

    problems: List[str] = []

    if is_dcd:
        # MDAnalysis derives this from the DCD header, so a placeholder there
        # arrives as an absurd number rather than as a missing one.
        step = float(universe.trajectory.dt)
        blank.update(timestep_ps=step)
        if 0 < step < FLOOR_PS:
            problems.append("IMPLAUSIBLE TIMESTEP")

    if headers_only:
        # Every remaining check needs the frames, and for these formats even
        # the box does: a frame has to be decompressed to read it.
        return problems

    zero_idx: List[int] = []
    bad_coord_idx: List[int] = []
    lengths: List[List[float]] = []
    angles: List[List[float]] = []

    try:
        for step_ in universe.trajectory:
            verdict = classify_frame(step_.positions)
            if verdict == "zero":
                zero_idx.append(step_.frame)
            elif verdict is not None:
                bad_coord_idx.append(step_.frame)

            box = step_.dimensions
            if box is None:
                lengths.append([0.0, 0.0, 0.0])
                angles.append([0.0, 0.0, 0.0])
            else:
                lengths.append([float(v) for v in box[:3]])
                angles.append([float(v) for v in box[3:6]])
    except Exception as err:
        # Keep what was found before the reader gave up. A file that stops a
        # reader partway is usually stopped by a defect we can already name,
        # and replacing those findings with "unreadable" would tell the
        # submitter less than we know.
        problems.append("FRAME SCAN INCOMPLETE")
        blank.update(read_error=f"{type(err).__name__}: {err}")

    bad_cell = scan_cell(lengths, angles) if lengths else []

    blank.update(
        bad_cell_frames=len(bad_cell),
        bad_cell_runs=group_runs(bad_cell),
        zero_frames=len(zero_idx),
        zero_runs=group_runs(zero_idx),
        bad_coord_frames=len(bad_coord_idx),
        bad_coord_runs=group_runs(bad_coord_idx),
    )

    if bad_cell:
        problems.append("BAD UNIT CELL")
    if zero_idx:
        problems.append("ZERO FRAMES")
    if bad_coord_idx:
        problems.append("BAD COORDINATES")

    return problems


# --------------------------------------------------
def check_trajectory(
    directory: str,
    trajectory: str,
    topology: str,
    top_atoms: Optional[int],
    headers_only: bool,
) -> Finding:
    """Check one trajectory against an already-read topology"""

    blank = dict(
        directory=directory,
        trajectory=trajectory,
        topology=topology,
        traj_atoms=None,
        top_atoms=top_atoms,
        frames=None,
        zero_frames=0,
        zero_runs=[],
        bad_cell_frames=0,
        bad_cell_runs=[],
        bad_coord_frames=0,
        bad_coord_runs=[],
        timestep_ps=None,
        read_error=None,
        problems=[],
    )

    path = os.path.join(directory, trajectory)
    lowered = trajectory.lower()

    if not lowered.endswith(MDANALYSIS_SUFFIXES):
        return Finding(
            **blank, error=f"no reader for this format ({trajectory})"
        )

    if not os.path.isfile(path):
        return Finding(**blank, error="file is missing")

    if os.path.getsize(path) == 0:
        blank.update(frames=0, problems=["EMPTY TRAJECTORY"])
        return Finding(**blank, error=None)

    try:
        if lowered.endswith(NETCDF_SUFFIXES):
            problems = check_netcdf(path, blank, headers_only)
        else:
            problems = check_via_mdanalysis(
                path, blank, headers_only, lowered.endswith(".dcd")
            )
    except Exception as err:
        return Finding(**blank, error=f"unreadable trajectory: {err}")

    if (
        top_atoms is not None
        and blank["traj_atoms"] is not None
        and blank["traj_atoms"] != top_atoms
        and "EMPTY TRAJECTORY" not in problems
    ):
        problems.insert(0, "ATOM MISMATCH")

    blank.update(problems=problems)
    return Finding(**blank, error=None)


# --------------------------------------------------
def check_dir(directory: str, headers_only: bool) -> List[Finding]:
    """
    Check one simulation directory

    Returns one Finding per declared trajectory. A directory may declare
    several, and each is checked against the same topology -- reporting only
    the first would hide every later defect.
    """

    blank = dict(
        directory=directory,
        trajectory="",
        topology="",
        traj_atoms=None,
        top_atoms=None,
        frames=None,
        zero_frames=0,
        zero_runs=[],
        bad_cell_frames=0,
        bad_cell_runs=[],
        bad_coord_frames=0,
        bad_coord_runs=[],
        timestep_ps=None,
        read_error=None,
        problems=[],
    )

    try:
        with open(os.path.join(directory, "mdrepo-metadata.toml"), "rb") as fh:
            meta = tomllib.load(fh)
    except Exception as err:
        return [Finding(**blank, error=f"unreadable metadata: {err}")]

    trajectories = meta.get("trajectory_file_names") or []
    topology = meta.get("topology_file_name") or ""

    if not trajectories:
        return [Finding(**blank, error="metadata names no trajectory")]

    # A topology we cannot read is not fatal: every frame check below still
    # works without it, and only ATOM MISMATCH is lost. Saying so beats
    # refusing to look at the trajectory at all.
    top_atoms: Optional[int] = None
    if topology:
        try:
            top_atoms = count_atoms(os.path.join(directory, topology))
        except Exception:
            top_atoms = None

    return [
        check_trajectory(directory, name, topology, top_atoms, headers_only)
        for name in trajectories
    ]


# --------------------------------------------------
def describe(finding: Finding) -> str:
    """One-line verdict for the console"""

    if finding.error:
        return f"ERROR {finding.error}"

    if not finding.problems:
        return f"ok {finding.traj_atoms} atoms, {finding.frames} frames"

    parts = []
    for problem in finding.problems:
        if problem == "ATOM MISMATCH":
            parts.append(
                f"ATOM MISMATCH trajectory {finding.traj_atoms} "
                f"vs topology {finding.top_atoms}"
            )
        elif problem == "ZERO FRAMES":
            parts.append(
                f"ZERO FRAMES {finding.zero_frames} of {finding.frames} "
                f"(frames {format_runs(finding.zero_runs)})"
            )
        elif problem == "BAD UNIT CELL":
            parts.append(
                f"BAD UNIT CELL {finding.bad_cell_frames} of {finding.frames} "
                f"(frames {format_runs(finding.bad_cell_runs)})"
            )
        elif problem == "BAD COORDINATES":
            parts.append(
                f"BAD COORDINATES {finding.bad_coord_frames} of "
                f"{finding.frames} (frames "
                f"{format_runs(finding.bad_coord_runs)})"
            )
        elif problem == "IMPLAUSIBLE TIMESTEP":
            parts.append(f"IMPLAUSIBLE TIMESTEP {finding.timestep_ps:g} ps")
        elif problem == "FRAME SCAN INCOMPLETE":
            parts.append(f"FRAME SCAN INCOMPLETE ({finding.read_error})")
        else:
            parts.append(problem)

    return "; ".join(parts)


# --------------------------------------------------
def sims(count: int) -> str:
    """Pluralize a simulation count"""

    return f"{count} simulation" + ("" if count == 1 else "s")


# --------------------------------------------------
def format_runs(runs: List[Tuple[int, int]]) -> str:
    """Render frame runs as compact ranges"""

    shown = [
        f"{start}-{end}" if start != end else f"{start}"
        for start, end in runs[:4]
    ]
    if len(runs) > 4:
        shown.append(f"and {len(runs) - 4} more")

    return ", ".join(shown)


# --------------------------------------------------
def build_report(findings: List[Finding], headers_only: bool) -> str:
    """
    Build a plain-text report of the problems found

    Keyed on the file names from the metadata rather than on directory names,
    so the report identifies each problem by something the person who prepared
    the data will recognise. The healthy count leads, so a handful of problems
    in a large submission does not read as a wholesale rejection.
    """

    def having(problem):
        return sorted(
            [f for f in findings if problem in f.problems],
            key=lambda f: f.trajectory,
        )

    mismatches = having("ATOM MISMATCH")
    empties = having("EMPTY TRAJECTORY")
    bad_cell = having("BAD UNIT CELL")
    zeros = having("ZERO FRAMES")
    bad_coord = having("BAD COORDINATES")
    no_time = having("NO TIME AXIS")
    bad_step = having("IMPLAUSIBLE TIMESTEP")
    incomplete = having("FRAME SCAN INCOMPLETE")
    errors = [f for f in findings if f.error]
    clean = [f for f in findings if not f.problems and not f.error]

    n_dirs = len({f.directory for f in findings})
    trajectories = (
        "" if len(findings) == n_dirs else f" ({len(findings)} trajectories)"
    )
    out = [
        f"Checked {n_dirs} simulation directories{trajectories}: "
        f"{len(clean)} are fine, {len(findings) - len(clean)} need attention."
    ]

    section = 0

    if mismatches:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORY AND TOPOLOGY DISAGREE ON ATOM COUNT "
            f"({sims(len(mismatches))})",
            "",
            "   The trajectory looks solvent-stripped while the topology",
            "   beside it is the unstripped one, so the pair cannot be read.",
            "   Each of these needs the topology matching its trajectory, or",
            "   the unstripped trajectory.",
            "",
        ]
        out += [
            f"   {f.trajectory}: trajectory has {f.traj_atoms} atoms, but "
            f"{f.topology} declares {f.top_atoms}"
            for f in mismatches
        ]

    if empties:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORY FILES HOLD NO FRAMES ({sims(len(empties))})",
            "",
            "   No data was ever written to these. The checksums match what",
            "   was uploaded, so the files themselves transferred correctly --",
            "   the copies at the source are empty.",
            "",
        ]
        out += [f"   {f.trajectory}" for f in empties]

    if bad_cell:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES CONTAIN FRAMES WITH NO USABLE BOX "
            f"({sims(len(bad_cell))})",
            "",
            "   Some frames record a unit cell that cannot describe a box:",
            "   a length or angle that is not a finite number, a length of",
            "   zero or less, or an angle outside 0 to 180 degrees. The frames",
            "   on either side are usually ordinary, which is what makes this",
            "   easy to miss. A frame like this can stall the conversion step",
            "   for hours instead of failing it, so these must be rewritten or",
            "   removed before the data is submitted.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.bad_cell_frames} of {f.frames} frames "
            f"(frames {format_runs(f.bad_cell_runs)})"
            for f in bad_cell
        ]

    if zeros:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES CONTAIN ALL-ZERO FRAMES "
            f"({sims(len(zeros))})",
            "",
            "   Some frames hold 0.0 for every coordinate of every atom --",
            "   the signature of a partially written file. The affected frames",
            "   cannot be recovered; these need rewriting, or those frames",
            "   removed.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.zero_frames} of {f.frames} frames "
            f"(frames {format_runs(f.zero_runs)})"
            for f in zeros
        ]

    if bad_coord:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES CONTAIN COORDINATES THAT ARE NOT "
            f"POSITIONS ({sims(len(bad_coord))})",
            "",
            "   Some frames hold coordinates that are NaN, infinity, or a",
            "   number far too large to place an atom. This is what",
            "   uninitialised memory looks like when it is written to a file",
            "   as if it were data. The affected frames cannot be recovered;",
            "   these need rewriting, or those frames removed.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.bad_coord_frames} of {f.frames} frames "
            f"(frames {format_runs(f.bad_coord_runs)})"
            for f in bad_coord
        ]

    if no_time:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES CARRY NO TIME AXIS "
            f"({sims(len(no_time))})",
            "",
            "   These NetCDF files have no `time` variable, so the frame",
            "   spacing cannot be read from them. Either rewrite them with",
            "   time information, or add `sampling_frequency_ps` to",
            "   mdrepo-metadata.toml (output frequency x integration timestep).",
            "",
        ]
        out += [f"   {f.trajectory}" for f in no_time]

    if bad_step:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES DECLARE AN IMPOSSIBLE FRAME SPACING "
            f"({sims(len(bad_step))})",
            "",
            "   The spacing in these DCD headers is far below any real",
            "   sampling interval, which usually means the header carries a",
            "   placeholder rather than the value. Add",
            "   `sampling_frequency_ps` to mdrepo-metadata.toml, or rewrite",
            "   the files with the spacing recorded.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.timestep_ps:g} ps per frame"
            for f in bad_step
        ]

    if incomplete:
        section += 1
        out += [
            "",
            f"{section}. THE READER COULD NOT GET THROUGH THESE FILES "
            f"({sims(len(incomplete))})",
            "",
            "   Reading stopped partway, so the frame checks above are",
            "   incomplete for these files and there may be more wrong with",
            "   them than is listed. This usually follows from a defect",
            "   already reported above: a frame with no usable box stops most",
            "   readers dead. Fix what is listed and run this again.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.read_error}" for f in incomplete
        ]

    if errors:
        out += ["", f"NOT CHECKED ({len(errors)} trajectories)", ""]
        out += [
            f"   {os.path.basename(f.directory)}"
            + (f"/{f.trajectory}" if f.trajectory else "")
            + f": {f.error}"
            for f in sorted(errors, key=lambda f: f.directory)
        ]

    if headers_only:
        out += [
            "",
            "Note: run without --headers-only to also scan the frames, which",
            "is what finds all-zero frames and coordinates that are not",
            "positions. Only NetCDF had its unit cell checked in this run.",
        ]

    return "\n".join(out)


# --------------------------------------------------
def main() -> None:
    """Make a jazz noise here"""

    args = get_args()
    dirs = find_dirs(args.dirs)

    if not dirs:
        sys.exit("No simulation directories found")

    print(
        f"Checking {len(dirs)} directory(s) with {args.threads} thread(s)"
        + (" (headers only)" if args.headers_only else ""),
        file=sys.stderr,
    )

    findings: List[Finding] = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [
            pool.submit(check_dir, directory, args.headers_only)
            for directory in dirs
        ]

        for future in futures:
            for finding in future.result():
                findings.append(finding)
                if not args.quiet and (finding.problems or finding.error):
                    print(
                        f"{os.path.basename(finding.directory)}: "
                        f"{describe(finding)}",
                        file=sys.stderr,
                        flush=True,
                    )

    findings.sort(key=lambda f: (f.directory, f.trajectory))

    counts: Dict[str, int] = {}
    for finding in findings:
        for problem in finding.problems:
            counts[problem] = counts.get(problem, 0) + 1
        if finding.error:
            counts["NOT CHECKED"] = counts.get("NOT CHECKED", 0) + 1

    ok = len([f for f in findings if not f.problems and not f.error])

    print("", file=sys.stderr)
    print(
        f"{len(dirs)} directory(s), {len(findings)} trajectory(s): {ok} clean",
        file=sys.stderr,
    )
    for problem, count in sorted(counts.items()):
        print(f"  {problem}: {count}", file=sys.stderr)

    if args.report:
        with open(args.report, "wt") as out_fh:
            out_fh.write(build_report(findings, args.headers_only))
            out_fh.write("\n")
        print(f"\nReport written to '{args.report}'", file=sys.stderr)

    sys.exit(1 if counts else 0)


# --------------------------------------------------
if __name__ == "__main__":
    main()
