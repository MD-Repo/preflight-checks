#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""
Author : Ken Youens-Clark <kyclark@arizona.edu>
Date   : 2026-08-12
Purpose: Check AMBER simulation directories against their own topology before
         submitting them

Each check below catches a defect that is cheap to find here and expensive to
find later: MDRepo's pipeline only discovers these after the whole submission
has been uploaded and converted, and the error it reports then usually names a
missing output file rather than the cause.

  ATOM MISMATCH -- the trajectory's `atom` dimension against the topology's
  NATOM. Stripping solvent after a simulation and then shipping the ORIGINAL
  topology beside the stripped trajectory produces a pair no tool can read: an
  ~850-atom trajectory against an ~11,000-atom topology. Both numbers are
  header reads, so this costs milliseconds per directory and needs none of the
  trajectory body (--headers-only).

  EMPTY TRAJECTORY -- a file too small to hold even one frame. These are
  usually the result of an interrupted copy or a job that never wrote output.
  They carry a valid checksum and a correct manifest, so nothing upstream of
  this notices.

  BAD UNIT CELL -- frames whose box is not a box: a length or an angle that is
  not a finite number, a length that is zero or negative, or an angle outside
  0 to 180 degrees. Two submissions have arrived this way, one carrying a box
  of exact zeros and one carrying values like 1.46e+233. A frame like this can
  hang the conversion step for hours rather than fail it. The cell arrays are
  three numbers per frame against tens of thousands of coordinates, so this
  runs even under --headers-only -- though NetCDF stores each frame's records
  together, so reading them still walks the whole file rather than just its
  header.

  It is not a substitute for BAD COORDINATES below. One file of the second
  submission carries damaged frames under a box identical to its neighbours',
  and only the coordinate scan finds it.

  ZERO FRAMES -- frames in which every coordinate of every atom is exactly 0.0,
  the signature of a partially written file. They read as valid NetCDF and
  survive conversion, then break the analysis stage: fitting an all-zero frame
  produces coordinates that overflow the XTC integer encoding, and the RMSD /
  RMSF step rejects the result. Finding these needs the coordinate data, so
  unlike the checks above it is not free and not available before upload.

  BAD COORDINATES -- frames holding coordinates that are not positions: NaN,
  infinity, or a number so large it cannot describe an atom. These come from a
  writer that put uninitialised memory into the file instead of data. The size
  test is not redundant: XTC cannot store NaN, so a damaged frame written
  through one comes back as a finite 21,474,836 angstroms, and a test for
  finiteness alone would pass it. They read as valid NetCDF, survive conversion, and
  reach the analysis stage, where they either stall it or produce numbers that
  mean nothing. Reading them needs the coordinate data, so this rides along
  with the zero-frame scan and costs nothing on top of it.

  NO TIME AXIS -- a NetCDF trajectory with no `time` variable. Frame spacing
  cannot be read from such a file, so MDRepo cannot derive the duration or the
  sampling frequency. Declare `sampling_frequency_ps` in mdrepo-metadata.toml
  when this is reported, or the directory will be rejected.

Directories are found by looking for `mdrepo-metadata.toml`, so both a single
simulation directory and a parent holding many of them work as arguments.

Read-only: never writes into a simulation directory.
"""

import argparse
import os
import re
import sys
import tomllib
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple

try:
    import numpy as np
    from scipy.io import netcdf_file
except ImportError as err:
    sys.exit(
        f"{err}\n\n"
        "This needs numpy and scipy, which are not installed. From the "
        "directory holding this script:\n\n"
        "    python3 -m venv .venv\n"
        "    ./.venv/bin/pip install -r requirements.txt\n"
        "    ./.venv/bin/python check_amber.py --help\n"
    )

# scipy warns on closing an mmap'd NetCDF while array views onto it still
# exist. That is exactly how this reads coordinates -- in chunks, copying each
# chunk -- so the warning describes intended use and would otherwise fire once
# per trajectory.
warnings.filterwarnings(
    "ignore",
    message="Cannot close a netcdf_file opened with mmap=True",
    category=RuntimeWarning,
)

# Frames are read in blocks so a large trajectory never lands in memory whole.
CHUNK_FRAMES = 256

# A coordinate this large is not a position. It is 0.1 mm, where a simulation
# box is a few hundred angstroms at most.
#
# The check matters because XTC cannot store NaN: writing one saturates the
# 32-bit integer encoding, and the value reads back as +/-21,474,836 A -- a
# finite, non-zero number that a finiteness test passes. That saturation is the
# signature the RMSD/RMSF ceiling sees after conversion, and it is the only
# trace a damaged frame leaves in an XTC. Without this, screening an XTC would
# find nothing.
MAX_ABS_COORD = 1.0e6


class Args(NamedTuple):
    """Command-line arguments"""

    dirs: List[str]
    threads: int
    headers_only: bool
    report: Optional[str]
    quiet: bool


class Finding(NamedTuple):
    """One directory's verdict"""

    directory: str
    trajectory: str
    topology: str
    traj_atoms: Optional[int]
    top_atoms: Optional[int]
    traj_size: Optional[int]
    frames: Optional[int]
    zero_frames: int
    zero_runs: List[Tuple[int, int]]
    bad_cell_frames: int
    bad_cell_runs: List[Tuple[int, int]]
    bad_coord_frames: int
    bad_coord_runs: List[Tuple[int, int]]
    has_time: Optional[bool]
    problems: List[str]
    error: Optional[str]


# --------------------------------------------------
def get_args() -> Args:
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Check AMBER simulation directories before submitting",
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
        help="Skip the coordinate scans; reads headers and the unit cell only",
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
def read_prmtop_natom(path: str) -> int:
    """
    Read NATOM from an AMBER prmtop

    NATOM is the first field of the POINTERS block, which follows a %FLAG
    POINTERS line and its %FORMAT line. Only the header is read.
    """

    with open(path, "rt", errors="replace") as fh:
        in_pointers = False
        for line in fh:
            if line.startswith("%FLAG"):
                in_pointers = line.split()[1:2] == ["POINTERS"]
                continue

            if in_pointers:
                if line.startswith("%FORMAT"):
                    continue
                fields = re.findall(r"\d+", line)
                if fields:
                    return int(fields[0])

    raise ValueError("no POINTERS block")


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
def scan_coordinates(coords) -> Tuple[List[int], List[int], List[int]]:
    """
    Find all-zero frames and frames holding values that are not finite

    Both answers come out of one pass, because the coordinates are the only
    expensive thing this program reads and there is no reason to read them
    twice. The two results never overlap: a frame of exact zeros is finite.
    """

    zero: List[int] = []
    nonfinite: List[int] = []
    huge: List[int] = []
    total = coords.shape[0]

    for start in range(0, total, CHUNK_FRAMES):
        block = np.asarray(coords[start : start + CHUNK_FRAMES])
        flat = block.reshape(block.shape[0], -1)

        finite = np.isfinite(flat).all(axis=1)
        for offset in np.where(~finite)[0]:
            nonfinite.append(start + int(offset))
        for offset in np.where(finite & ~flat.any(axis=1))[0]:
            zero.append(start + int(offset))

        big = finite & (np.abs(np.where(finite[:, None], flat, 0.0))
                        >= MAX_ABS_COORD).any(axis=1)
        for offset in np.where(big)[0]:
            huge.append(start + int(offset))

    return zero, nonfinite, huge


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
        traj_size=None,
        frames=None,
        zero_frames=0,
        zero_runs=[],
        bad_cell_frames=0,
        bad_cell_runs=[],
        bad_coord_frames=0,
        bad_coord_runs=[],
        has_time=None,
        problems=[],
    )

    try:
        with open(os.path.join(directory, "mdrepo-metadata.toml"), "rb") as fh:
            meta = tomllib.load(fh)
    except Exception as err:
        return [Finding(**blank, error=f"unreadable metadata: {err}")]

    trajectories = meta.get("trajectory_file_names") or []
    topology = meta.get("topology_file_name") or ""

    if not trajectories or not topology:
        blank.update(topology=topology)
        return [
            Finding(**blank, error="metadata names no trajectory or topology")
        ]

    try:
        top_atoms = read_prmtop_natom(os.path.join(directory, topology))
    except Exception as err:
        blank.update(topology=topology)
        return [Finding(**blank, error=f"unreadable topology: {err}")]

    return [
        check_trajectory(directory, name, topology, top_atoms, headers_only)
        for name in trajectories
    ]


# --------------------------------------------------
def check_trajectory(
    directory: str,
    trajectory: str,
    topology: str,
    top_atoms: int,
    headers_only: bool,
) -> Finding:
    """Check one trajectory against an already-read topology"""

    blank = dict(
        directory=directory,
        trajectory=trajectory,
        topology=topology,
        traj_atoms=None,
        top_atoms=top_atoms,
        traj_size=None,
        frames=None,
        zero_frames=0,
        zero_runs=[],
        bad_cell_frames=0,
        bad_cell_runs=[],
        bad_coord_frames=0,
        bad_coord_runs=[],
        has_time=None,
        problems=[],
    )

    if not trajectory.endswith(".nc"):
        return Finding(**blank, error=f"not a NetCDF trajectory ({trajectory})")

    # A trajectory too small to hold one frame of coordinates never reaches the
    # NetCDF reader in a useful state -- the parse fails on the header, because
    # the data section was never written. Catch it on size first so it reports
    # as the defect it is rather than as a parse error.
    traj_path = os.path.join(directory, trajectory)
    try:
        traj_size = os.path.getsize(traj_path)
    except OSError as err:
        return Finding(**blank, error=f"unreadable trajectory: {err}")

    blank.update(traj_size=traj_size)

    if traj_size < top_atoms * 3 * 4:
        blank.update(frames=0, problems=["EMPTY TRAJECTORY"])
        return Finding(**blank, error=None)

    problems: List[str] = []

    try:
        with netcdf_file(traj_path, "r", mmap=True) as ncf:
            traj_atoms = int(ncf.dimensions["atom"])
            coords = ncf.variables["coordinates"]
            frames = int(coords.shape[0])
            has_time = "time" in ncf.variables

            blank.update(
                traj_atoms=traj_atoms, frames=frames, has_time=has_time
            )

            if traj_atoms != top_atoms:
                problems.append("ATOM MISMATCH")

            if not has_time:
                problems.append("NO TIME AXIS")

            # The cell arrays are three numbers per frame against tens of
            # thousands of coordinates, so this runs whatever the atom counts
            # say and whatever --headers-only says. It is the check that costs
            # least and, on the two submissions that prompted it, the only one
            # that fires.
            bad_cell: List[int] = []
            if "cell_lengths" in ncf.variables and "cell_angles" in ncf.variables:
                bad_cell = scan_cell(
                    ncf.variables["cell_lengths"][:],
                    ncf.variables["cell_angles"][:],
                )
                if bad_cell:
                    problems.append("BAD UNIT CELL")

            # Only scan coordinates when the pair is coherent: a mismatched
            # pair cannot be processed anyway, and the scan is the only
            # expensive thing here.
            zero_idx: List[int] = []
            nonfinite_idx: List[int] = []
            if not headers_only and traj_atoms == top_atoms:
                zero_idx, unfinite, huge = scan_coordinates(coords)
                # One defect with two faces. A coordinate of 3.4e38 written
                # into an XTC comes back as 21,474,836 -- finite, and just as
                # impossible -- so reporting them apart would tell a submitter
                # that two different things went wrong with one frame.
                nonfinite_idx = sorted(set(unfinite) | set(huge))
                if zero_idx:
                    problems.append("ZERO FRAMES")
                if nonfinite_idx:
                    problems.append("BAD COORDINATES")

            blank.update(
                zero_frames=len(zero_idx),
                zero_runs=group_runs(zero_idx),
                bad_cell_frames=len(bad_cell),
                bad_cell_runs=group_runs(bad_cell),
                bad_coord_frames=len(nonfinite_idx),
                bad_coord_runs=group_runs(nonfinite_idx),
            )
    except Exception as err:
        return Finding(**blank, error=f"unreadable trajectory: {err}")

    blank.update(problems=problems)
    return Finding(**blank, error=None)


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
                f"ZERO FRAMES {finding.zero_frames} of {finding.frames}"
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
        elif problem == "EMPTY TRAJECTORY":
            parts.append(f"EMPTY TRAJECTORY {finding.traj_size} bytes")
        else:
            parts.append(problem)

    return "; ".join(parts)


# --------------------------------------------------
def sims(count: int) -> str:
    """Pluralize a simulation count"""

    return f"{count} simulation" + ("" if count == 1 else "s")


# --------------------------------------------------
def format_runs(runs: List[Tuple[int, int]]) -> str:
    """Render zero-frame runs as compact ranges"""

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

    mismatches = [f for f in findings if "ATOM MISMATCH" in f.problems]
    empties = [f for f in findings if "EMPTY TRAJECTORY" in f.problems]
    zeros = [f for f in findings if "ZERO FRAMES" in f.problems]
    bad_cell = [f for f in findings if "BAD UNIT CELL" in f.problems]
    bad_coord = [f for f in findings if "BAD COORDINATES" in f.problems]
    no_time = [f for f in findings if "NO TIME AXIS" in f.problems]
    errors = [f for f in findings if f.error]
    clean = [f for f in findings if not f.problems and not f.error]

    n_dirs = len({f.directory for f in findings})
    trajectories = (
        ""
        if len(findings) == n_dirs
        else f" ({len(findings)} trajectories)"
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
            for f in sorted(mismatches, key=lambda f: f.trajectory)
        ]

    if empties:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORY FILES ARE EMPTY ({sims(len(empties))})",
            "",
            "   These files are too small to hold a single frame, so no data",
            "   was ever written to them. The checksums match what was",
            "   uploaded, so the files themselves transferred correctly -- the",
            "   copies at the source are empty.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.traj_size} bytes, no frames"
            for f in sorted(empties, key=lambda f: f.trajectory)
        ]

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
            for f in sorted(bad_cell, key=lambda f: f.trajectory)
        ]

    if zeros:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES CONTAIN ALL-ZERO FRAMES "
            f"({sims(len(zeros))})",
            "",
            "   These are valid NetCDF files with the right atom count, but",
            "   some frames hold 0.0 for every coordinate of every atom --",
            "   the signature of a partially written file. The affected frames",
            "   cannot be recovered; these need rewriting, or those frames",
            "   removed.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.zero_frames} of {f.frames} frames are "
            f"all-zero (frames {format_runs(f.zero_runs)})"
            for f in sorted(zeros, key=lambda f: f.trajectory)
        ]

    if bad_coord:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORIES CONTAIN COORDINATES THAT ARE NOT "
            f"NUMBERS ({sims(len(bad_coord))})",
            "",
            "   Some frames hold coordinates that are NaN or infinity rather",
            "   than a position. This is what uninitialised memory looks like",
            "   when it is written to a file as if it were data. The affected",
            "   frames cannot be recovered; these need rewriting, or those",
            "   frames removed.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.bad_coord_frames} of {f.frames} frames "
            f"(frames {format_runs(f.bad_coord_runs)})"
            for f in sorted(bad_coord, key=lambda f: f.trajectory)
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
        out += [
            f"   {f.trajectory}"
            for f in sorted(no_time, key=lambda f: f.trajectory)
        ]

    if errors:
        out += ["", f"NOT CHECKED ({len(errors)} directories)", ""]
        out += [
            f"   {os.path.basename(f.directory)}: {f.error}"
            for f in sorted(errors, key=lambda f: f.directory)
        ]

    if headers_only:
        out += [
            "",
            "Note: run without --headers-only to also scan the coordinates,",
            "which is what finds all-zero frames and coordinates that are not",
            "numbers. The unit cell was checked either way.",
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

    findings.sort(key=lambda f: f.directory)

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
