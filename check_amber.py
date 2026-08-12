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

  ZERO FRAMES -- frames in which every coordinate of every atom is exactly 0.0,
  the signature of a partially written file. They read as valid NetCDF and
  survive conversion, then break the analysis stage: fitting an all-zero frame
  produces coordinates that overflow the XTC integer encoding, and the RMSD /
  RMSF step rejects the result. Finding these needs the coordinate data, so
  unlike the checks above it is not free and not available before upload.

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
        help="Atom counts only; skips the zero-frame scan, reads no coordinates",
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
def scan_zero_frames(coords) -> Tuple[int, List[Tuple[int, int]]]:
    """Count all-zero coordinate frames, grouped into contiguous runs"""

    zero_idx = []
    total = coords.shape[0]

    for start in range(0, total, CHUNK_FRAMES):
        block = np.asarray(coords[start : start + CHUNK_FRAMES])
        flat = block.reshape(block.shape[0], -1)
        for offset in np.where(~flat.any(axis=1))[0]:
            zero_idx.append(start + int(offset))

    runs: List[Tuple[int, int]] = []
    for idx in zero_idx:
        if runs and idx == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], idx)
        else:
            runs.append((idx, idx))

    return len(zero_idx), runs


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

            # Only scan coordinates when the pair is coherent: a mismatched
            # pair cannot be processed anyway, and the scan is the only
            # expensive thing here.
            zero_frames, zero_runs = 0, []
            if not headers_only and traj_atoms == top_atoms:
                zero_frames, zero_runs = scan_zero_frames(coords)
                if zero_frames:
                    problems.append("ZERO FRAMES")

            blank.update(zero_frames=zero_frames, zero_runs=zero_runs)
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
            "Note: run without --headers-only to also scan for zero frames.",
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
