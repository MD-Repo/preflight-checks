#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "mdanalysis>=2.10.0"]
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

  FRAME JUMPS -- frames in which an atom has moved MAX_JUMP (5,000 A) or more
  from the frame before, or twice the box diagonal if that is larger. The
  Dissociation Dynamic Database sent last frames that move whole protein
  chains by one constant offset of 1e4-1e7 A; five of twelve never reach the
  BAD COORDINATES size limit. Wrapping in a periodic box moves an atom at most
  one diagonal, which is why the limit grows with the box. A step where time
  jumps or starts again is where joined runs meet, and is not tested, unless
  the block after it is a single frame. The same rule as the pipeline's
  screen_trajectory.py.

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

  TOPOLOGY MISMATCH -- the first, middle or last frame does not fit the
  topology's bonds: BAD_BOND_SHARE (5%) or more of the bonds are stretched
  past any real bond length. A trajectory can have the right atom count and
  still not belong to its topology -- one from another run or another strip
  step, or atoms in another order. Then the "bonded" atoms sit Angstroms to
  tens of Angstroms apart, while a correct pairing breaks almost no bonds, so
  the bounds can be loose. Distances use the minimum image when the frame has
  a box. The bonds come from the topology (.psf, .prmtop, .tpr, a readable
  GROMACS .top), or are guessed from the topology's own coordinates for .pdb
  and .gro -- never from the trajectory, which would be circular. Bonds whose
  atoms are not a recognisable element by name and mass (coarse-grained
  beads, virtual sites) are left out, and when too few are left the check is
  skipped and says why.

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

# An atom that moves this far between consecutive frames has not moved: the
# frame is damaged. The limit is this, or twice the frame's own box diagonal if
# that is larger -- a wrapped trajectory legitimately moves an atom up to one
# diagonal when it leaves one corner and comes back through the opposite one.
# The same rule, and the same numbers, as MDRepo's screen_trajectory.py.
MAX_JUMP = 5.0e3

# Two frames are consecutive in time when their spacing is within this share
# of the file's regular step. float32 times wander in the last digits.
STEP_TOLERANCE = 0.01

# Below this, a DCD's declared spacing is a placeholder rather than a number.
FLOOR_PS = 1.0e-4

NETCDF_SUFFIXES = (".nc", ".netcdf")
MDANALYSIS_SUFFIXES = (".xtc", ".trr", ".dcd", ".nc", ".netcdf")

# TOPOLOGY MISMATCH. A bond is stretched when it is longer than BOND_STRETCH
# times the sum of its atoms' covalent radii, and never at less than
# BOND_FLOOR: CHARMM's TIP3P carries an H-H "bond" of 1.51 A, which the
# radii alone would flag in every water. A frame fails when BAD_BOND_SHARE of
# its checked bonds are stretched. A wrong pairing breaks most bonds by
# Angstroms, a right one almost none, so none of these needs to be tight.
BOND_STRETCH = 1.5
BOND_FLOOR = 2.0
BAD_BOND_SHARE = 0.05

# Below this many checkable bonds, or below this share of the topology's
# bonds, the check says it was skipped rather than give a verdict.
MIN_BONDS = 10
MIN_CHECKABLE_SHARE = 0.5

# Guessing bonds from a .pdb's or .gro's own coordinates: two atoms are bonded
# when closer than the sum of their covalent radii plus this, the usual rule.
GUESS_SLACK = 0.45

# Topology formats that carry coordinates but usually no (or only some) bonds.
GUESS_SUFFIXES = (".pdb", ".ent", ".gro")

# Covalent radius (A, Cordero et al. 2008) and standard atomic mass of the
# elements bonded in biomolecular systems. An atom is given an element only
# when its name starts with the symbol and, when the topology has masses, its
# mass fits that element (see element_fits).
ELEMENTS: Dict[str, Tuple[float, float]] = {
    "H": (0.31, 1.008),
    "LI": (1.28, 6.94),
    "B": (0.84, 10.81),
    "C": (0.76, 12.011),
    "N": (0.71, 14.007),
    "O": (0.66, 15.999),
    "F": (0.57, 18.998),
    "NA": (1.66, 22.990),
    "MG": (1.41, 24.305),
    "AL": (1.21, 26.982),
    "SI": (1.11, 28.085),
    "P": (1.07, 30.974),
    "S": (1.05, 32.06),
    "CL": (1.02, 35.45),
    "K": (2.03, 39.098),
    "CA": (1.76, 40.078),
    "MN": (1.39, 54.938),
    "FE": (1.32, 55.845),
    "CO": (1.26, 58.933),
    "NI": (1.24, 58.693),
    "CU": (1.32, 63.546),
    "ZN": (1.22, 65.38),
    "SE": (1.20, 78.971),
    "BR": (1.20, 79.904),
    "I": (1.39, 126.904),
}


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
    jump_frames: int
    jump_runs: List[Tuple[int, int]]
    time_breaks: List[int]
    timestep_ps: Optional[float]
    read_error: Optional[str]
    problems: List[str]
    error: Optional[str]
    # TOPOLOGY MISMATCH: (frame, stretched, checked) for each frame tested,
    # the worst bonds of the worst frame, and why the check did not run.
    bond_frames: List[Tuple[int, int, int]] = []
    bond_worst: List[str] = []
    bond_skipped: Optional[str] = None


class BondSet(NamedTuple):
    """A topology's checkable bonds, with what is needed to report them"""

    pairs: np.ndarray  # (n, 2) atom indexes
    limits: np.ndarray  # (n,) the length past which each one is stretched
    labels: List[str]  # one per atom: "LYS 12 NZ"
    source: str  # "topology" or "guessed"


class Topology(NamedTuple):
    """What the topology file says, read once per directory"""

    atoms: Optional[int]
    bonds: Optional[BondSet]
    bond_note: Optional[str]  # why there is no BondSet


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
# FRAME JUMPS. The functions from here to classify_frame are copied from
# MDRepo's screen_trajectory.py and must stay identical to it.


# --------------------------------------------------
def box_diagonal(lengths, angles) -> Optional[float]:
    """
    The longest distance a wrap can move an atom in one box, or None

    Wrapping moves an atom by a sum of box vectors, at most one of each, so the
    longest move is the longest of a+b+c, a+b-c, a-b+c and -a+b+c. For a
    rectangular box that is the plain diagonal. No box -- zero, negative or
    non-finite lengths, or angles that make no cell -- gives None.
    """

    lengths = np.asarray(lengths, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if lengths.shape != (3,) or angles.shape != (3,):
        return None
    if not (np.isfinite(lengths).all() and np.isfinite(angles).all()):
        return None
    if (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any():
        return None

    alpha, beta, gamma = np.radians(angles)
    a_len, b_len, c_len = lengths
    a = np.array([a_len, 0.0, 0.0])
    b = np.array([b_len * np.cos(gamma), b_len * np.sin(gamma), 0.0])
    cx = c_len * np.cos(beta)
    cy = c_len * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c_len**2 - cx**2 - cy**2
    if not np.isfinite(cz_sq) or cz_sq <= 0:
        return None
    c = np.array([cx, cy, np.sqrt(cz_sq)])

    return float(max(np.linalg.norm(v) for v in (a + b + c, a + b - c, a - b + c, -a + b + c)))


# --------------------------------------------------
def jump_limit(diagonal: Optional[float]) -> float:
    """MAX_JUMP, or twice the box diagonal when there is a box that large"""

    if diagonal is None or not np.isfinite(diagonal) or diagonal <= 0:
        return MAX_JUMP

    return max(MAX_JUMP, 2.0 * float(diagonal))


# --------------------------------------------------
def regular_step(times) -> Optional[float]:
    """
    The file's usual spacing between frames, or None if time tells us nothing

    The median, not the most common value: float32 times differ in the last
    digits, so an exact count can split one real step into several. A file
    with no times, or whose times do not advance, gives None.
    """

    if times is None or len(times) < 2:
        return None

    step = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    if not np.isfinite(step) or step <= 0:
        return None

    return step


# --------------------------------------------------
def find_jumps(moves, times, diagonals) -> Tuple[List[int], List[int]]:
    """
    Frames whose atoms moved too far, and the time boundaries that excused one

    `moves[i]` is the largest one-atom move from frame i-1 to frame i (0 for
    frame 0). `diagonals[i]` is frame i's box diagonal, or None.

    A trajectory may be several runs joined, and between two runs the system
    can be anywhere, so a jump is not tested across a boundary in time: a step
    longer than the regular one, or one that goes back (a clock restart). Only
    a boundary into a block of two or more frames counts. Otherwise a frame
    damaged in its time as well as its coordinates -- a garbage time on a
    damaged last frame -- would excuse itself.

    Without usable times nothing can show a boundary, so every step is tested.
    A wrong refusal is loud and gets fixed; a missed defect is silent.
    """

    frames = len(moves)
    step = regular_step(times)

    irregular = [False] * frames
    if step is not None:
        spacing = np.diff(np.asarray(times, dtype=np.float64))
        for i in range(1, frames):
            irregular[i] = not abs(spacing[i - 1] - step) <= STEP_TOLERANCE * step

    boundaries = [
        i
        for i in range(1, frames)
        if irregular[i] and i + 1 < frames and not irregular[i + 1]
    ]
    excused = set(boundaries)

    jumps = [
        i
        for i in range(1, frames)
        if i not in excused
        and moves[i] >= jump_limit(None if diagonals is None else diagonals[i])
    ]

    return jumps, boundaries


# --------------------------------------------------
def largest_move(frame, before) -> float:
    """
    The largest one-atom move between two frames

    Squared distances, and one square root at the end. A non-finite coordinate
    is reported by BAD COORDINATES, so it does not count here.
    """

    with np.errstate(invalid="ignore", over="ignore"):
        # float32 is plenty: the question is 5,000 A against a few hundred.
        delta = np.subtract(frame, before, dtype=np.float32)
        squared = np.einsum("ij,ij->i", delta, delta)
    finite = squared[np.isfinite(squared)]

    return float(np.sqrt(finite.max())) if finite.size else 0.0


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
def scan_coordinates(coords, moves=None) -> Tuple[List[int], List[int]]:
    """
    Scan NetCDF coordinates in blocks, without MDAnalysis

    NetCDF is read directly for the same reason the unit cell is: it is
    faster in blocks than frame at a time, it matches `check_amber.py` and
    the pipeline's own screen exactly, and MDAnalysis cannot read a NetCDF
    frame whose box is zero -- which a non-periodic trajectory legitimately
    has in every frame.

    Returns the all-zero frames and the frames that are not positions. Given a
    list as `moves`, the same pass appends each frame's largest one-atom move
    from the frame before, for FRAME JUMPS; the last frame of each block is
    carried into the next, so a jump at a block's first frame is measured too.
    """

    zero: List[int] = []
    bad: List[int] = []
    total = coords.shape[0]
    before = None

    for start in range(0, total, CHUNK_FRAMES):
        block = np.asarray(coords[start : start + CHUNK_FRAMES])
        flat = block.reshape(block.shape[0], -1)

        if moves is not None:
            for frame in block:
                moves.append(0.0 if before is None else largest_move(frame, before))
                before = frame

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
def netcdf_coordinates(path, moves=None):
    """A NetCDF's zero frames and not-a-position frames (and moves, if asked)"""

    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=True) as ncf:
        return scan_coordinates(ncf.variables["coordinates"], moves)


# --------------------------------------------------
def netcdf_times(path):
    """A NetCDF's frame times, or None when it has no time variable"""

    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=True) as ncf:
        if "time" not in ncf.variables:
            return None
        return np.array(ncf.variables["time"][:], dtype=np.float64)


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
# TOPOLOGY MISMATCH


# --------------------------------------------------
def load_topology(path: str):
    """
    Open a topology on its own, as an MDAnalysis Universe

    MDAnalysis reads ".top" as AMBER's format. A GROMACS .top that fails that
    way is tried again as one, with its own directory for any #include.
    """

    import MDAnalysis as mda

    try:
        return mda.Universe(path)
    except Exception:
        if not path.lower().endswith(".top"):
            raise

    return mda.Universe(
        path, topology_format="ITP", include_dir=os.path.dirname(path) or "."
    )


# --------------------------------------------------
def element_fits(element: str, mass: Optional[float]) -> bool:
    """
    Whether a topology mass can belong to an element

    Hydrogen mass repartitioning moves about 2 u onto each hydrogen from the
    heavy atom it is bonded to, so a hydrogen may weigh up to about 4 u and a
    heavy atom up to three hydrogens' worth less than standard. A virtual site
    weighs 0, and a coarse-grained bead (72 u in Martini) fits no element its
    name starts with. With no mass to go on, the name decides.
    """

    if mass is None or not np.isfinite(mass):
        return True

    if element == "H":
        return 0.9 <= mass <= 4.2

    standard = ELEMENTS[element][1]
    return standard - 6.5 <= mass <= standard + 0.6


# --------------------------------------------------
def assign_element(
    name: str, mass: Optional[float], declared: Optional[str]
) -> Optional[str]:
    """
    One atom's element, or None when it is not a recognisable one

    The topology's own element comes first, then the one- and two-letter
    symbols the name starts with (leading digits dropped: "1HB"), the first
    whose mass fits. One letter before two, so "CA" is an alpha carbon unless
    it weighs 40 u.
    """

    candidates: List[str] = []

    if declared is not None:
        symbol = str(declared).strip().upper()
        if symbol in ELEMENTS:
            candidates.append(symbol)

    letters = ""
    for char in str(name).lstrip("0123456789").upper():
        if not char.isalpha():
            break
        letters += char

    for symbol in (letters[:1], letters[:2]):
        if symbol in ELEMENTS and symbol not in candidates:
            candidates.append(symbol)

    for symbol in candidates:
        if element_fits(symbol, mass):
            return symbol

    return None


# --------------------------------------------------
def atom_elements(names, masses, declared) -> List[Optional[str]]:
    """Every atom's element (see assign_element), cached on what decides it"""

    cache: Dict[tuple, Optional[str]] = {}
    found: List[Optional[str]] = []

    for i, name in enumerate(names):
        mass = None if masses is None else float(masses[i])
        symbol = None if declared is None else declared[i]
        key = (name, None if mass is None else round(mass, 2), symbol)
        if key not in cache:
            cache[key] = assign_element(name, mass, symbol)
        found.append(cache[key])

    return found


# --------------------------------------------------
def usable_box(dims) -> Optional[np.ndarray]:
    """A frame's box for minimum-image distances, or None if it has none"""

    if dims is None:
        return None

    dims = np.asarray(dims, dtype=np.float64)
    if dims.shape != (6,) or box_diagonal(dims[:3], dims[3:6]) is None:
        return None

    return dims.astype(np.float32)


# --------------------------------------------------
def guess_pairs(
    coords, elements: List[Optional[str]], residues, box
) -> np.ndarray:
    """
    Bonds guessed from a structure's own coordinates

    Two atoms of recognisable elements in the same residue are bonded when
    they are closer than the sum of their covalent radii plus GUESS_SLACK.
    Within a residue only, so an ion sitting against a water is not bonded to
    it and then "stretched" when it moves off; the links between residues are
    few, and losing them costs the check nothing. Atoms closer than 0.1 A are
    not bonded, so a file of placeholder coordinates (all at the origin)
    gives no bonds rather than every pair.
    """

    from MDAnalysis.lib.distances import self_capped_distance

    known = np.array([i for i, e in enumerate(elements) if e is not None])
    if known.size < 2:
        return np.empty((0, 2), dtype=np.int64)

    radii = np.array([ELEMENTS[elements[i]][0] for i in known])
    pairs, dist = self_capped_distance(
        np.asarray(coords, dtype=np.float32)[known],
        max_cutoff=2.0 * float(radii.max()) + GUESS_SLACK,
        min_cutoff=0.1,
        box=box,
    )
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)

    keep = dist < radii[pairs[:, 0]] + radii[pairs[:, 1]] + GUESS_SLACK
    pairs = known[pairs[keep]].astype(np.int64)
    residues = np.asarray(residues)

    return pairs[residues[pairs[:, 0]] == residues[pairs[:, 1]]]


# --------------------------------------------------
def bond_limits(
    pairs: np.ndarray, elements: List[Optional[str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The bonds that can be judged, and the length past which each is stretched

    A bond is judged only when both its atoms are recognisable elements.
    """

    radius = np.array(
        [np.nan if e is None else ELEMENTS[e][0] for e in elements]
    )
    if len(pairs) == 0:
        return pairs, np.empty(0)

    summed = radius[pairs[:, 0]] + radius[pairs[:, 1]]
    keep = np.isfinite(summed)

    return pairs[keep], np.maximum(BOND_FLOOR, BOND_STRETCH * summed[keep])


class SkipBonds(Exception):
    """The topology's bonds cannot be judged, for the reason given"""


# --------------------------------------------------
def read_topology(path: str) -> Topology:
    """
    The topology's atom count and the bonds TOPOLOGY MISMATCH will test

    Never fails: a topology that cannot be read, or one whose bonds cannot be
    judged, comes back with a note saying why, and the other checks go on.
    """

    try:
        universe = load_topology(path)
    except Exception:
        return Topology(None, None, "the topology could not be read")

    atoms = universe.atoms
    natoms = len(atoms)

    try:
        return Topology(natoms, topology_bonds(path, universe), None)
    except SkipBonds as skip:
        return Topology(natoms, None, str(skip))
    except Exception as err:
        return Topology(natoms, None, f"its bonds could not be read ({err})")


# --------------------------------------------------
def topology_bonds(path: str, universe) -> BondSet:
    """The BondSet for read_topology, or SkipBonds saying why there is none"""

    atoms = universe.atoms
    if not hasattr(atoms, "names"):
        raise SkipBonds("the topology names no atoms")

    elements = atom_elements(
        atoms.names,
        atoms.masses if hasattr(atoms, "masses") else None,
        atoms.elements if hasattr(atoms, "elements") else None,
    )

    if path.lower().endswith(GUESS_SUFFIXES):
        try:
            coords = atoms.positions
        except Exception:
            raise SkipBonds("the topology has no coordinates to guess bonds from")
        pairs = guess_pairs(
            coords, elements, atoms.resindices, usable_box(universe.dimensions)
        )
        source = "guessed"
        if len(pairs) == 0:
            raise SkipBonds("no bonds could be guessed from the topology")
    else:
        if not hasattr(universe, "bonds") or len(universe.bonds) == 0:
            raise SkipBonds("the topology declares no bonds")
        pairs = np.asarray(universe.bonds.to_indices(), dtype=np.int64)
        source = "topology"

    total = len(pairs)
    pairs, limits = bond_limits(pairs, elements)
    if len(pairs) < MIN_BONDS or len(pairs) < MIN_CHECKABLE_SHARE * total:
        raise SkipBonds(
            f"only {len(pairs)} of {total} bonds join recognisable elements "
            "(coarse-grained, or virtual sites?)"
        )

    labels = [
        f"{resname} {resid} {name}"
        for resname, resid, name in zip(
            atoms.resnames if hasattr(atoms, "resnames") else [""] * len(atoms),
            atoms.resids if hasattr(atoms, "resids") else [""] * len(atoms),
            atoms.names,
        )
    ]

    return BondSet(pairs=pairs, limits=limits, labels=labels, source=source)


# --------------------------------------------------
def sample_frames(frames: int) -> List[int]:
    """The frames TOPOLOGY MISMATCH tests: first, middle and last"""

    return sorted({0, frames // 2, frames - 1}) if frames > 0 else []


# --------------------------------------------------
def check_bonds(
    bonds: BondSet, samples: dict, skip: set
) -> Tuple[List[Tuple[int, int, int]], List[str], bool]:
    """
    Measure the topology's bonds in each sampled frame

    `samples` maps a frame number to its (positions, box). Frames in `skip`
    are already reported as not holding positions and are not tested again.
    Returns (frame, stretched, checked) per frame tested, the worst bonds of
    the frame with the largest share stretched, and whether any frame
    reached BAD_BOND_SHARE.
    """

    from MDAnalysis.lib.distances import calc_bonds

    stats: List[Tuple[int, int, int]] = []
    worst: List[str] = []
    worst_share = 0.0
    mismatch = False

    for frame in sorted(samples):
        if frame in skip:
            continue

        positions, dims = samples[frame]
        positions = np.asarray(positions, dtype=np.float32)
        lengths = calc_bonds(
            positions[bonds.pairs[:, 0]],
            positions[bonds.pairs[:, 1]],
            box=usable_box(dims),
        )
        stretched = lengths > bonds.limits
        count = int(stretched.sum())
        stats.append((frame, count, len(lengths)))

        share = count / len(lengths)
        mismatch |= share >= BAD_BOND_SHARE
        if count and share > worst_share:
            worst_share = share
            ratio = np.where(stretched, lengths / bonds.limits, 0.0)
            worst = [
                f"frame {frame}: {bonds.labels[i]} - {bonds.labels[j]} "
                f"{length:.1f} A"
                for i, j, length in (
                    (*bonds.pairs[k], lengths[k])
                    for k in np.argsort(-ratio)[: min(5, count)]
                )
            ]

    return stats, worst, mismatch


# --------------------------------------------------
def netcdf_shape(path: str) -> Tuple[int, int]:
    """A NetCDF's atom count and frame count, from its dimensions"""

    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=True) as ncf:
        coords = ncf.variables["coordinates"]
        return int(coords.shape[1]), int(coords.shape[0])


# --------------------------------------------------
def check_netcdf(
    path: str, blank: dict, headers_only: bool, samples: Optional[dict] = None
) -> List[str]:
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
    jump_idx: List[int] = []
    breaks: List[int] = []
    if not headers_only:
        moves: List[float] = []
        zero_idx, bad_coord_idx = netcdf_coordinates(path, moves)
        diagonals = None
        if cell is not None:
            diagonals = [box_diagonal(ln, an) for ln, an in zip(*cell)]
        jump_idx, breaks = find_jumps(moves, netcdf_times(path), diagonals)

        if samples is not None:
            from scipy.io import netcdf_file

            with netcdf_file(path, "r", mmap=True) as ncf:
                coords = ncf.variables["coordinates"]
                for frame in sample_frames(frames):
                    box = None
                    if cell is not None:
                        box = np.concatenate([cell[0][frame], cell[1][frame]])
                    samples[frame] = (np.array(coords[frame]), box)

    blank.update(
        jump_frames=len(jump_idx),
        jump_runs=group_runs(jump_idx),
        time_breaks=breaks,
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
    if jump_idx:
        problems.append("FRAME JUMPS")

    return problems


# --------------------------------------------------
def check_via_mdanalysis(
    path: str,
    blank: dict,
    headers_only: bool,
    is_dcd: bool,
    samples: Optional[dict] = None,
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
    moves: List[float] = []
    times: List[float] = []
    diagonals: List[Optional[float]] = []
    before = None
    # TOPOLOGY MISMATCH's frames are kept as the scan passes them, so the
    # check costs no second read of the file.
    wanted = set(sample_frames(frames)) if samples is not None else set()

    try:
        for step_ in universe.trajectory:
            positions = step_.positions
            verdict = classify_frame(positions)

            # DCD stores only a start and an interval, so its times always
            # read back regular and every step of one is tested.
            times.append(float(step_.time))
            moves.append(0.0 if before is None else largest_move(positions, before))
            before = positions.copy()
            diagonals.append(
                None
                if step_.dimensions is None
                else box_diagonal(step_.dimensions[:3], step_.dimensions[3:6])
            )

            if step_.frame in wanted:
                samples[step_.frame] = (
                    positions.copy(),
                    None if step_.dimensions is None else step_.dimensions.copy(),
                )

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
    jump_idx, breaks = find_jumps(moves, times, diagonals)

    blank.update(
        jump_frames=len(jump_idx),
        jump_runs=group_runs(jump_idx),
        time_breaks=breaks,
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
    if jump_idx:
        problems.append("FRAME JUMPS")

    return problems


# --------------------------------------------------
def check_trajectory(
    directory: str,
    trajectory: str,
    topology: str,
    top_atoms: Optional[int],
    headers_only: bool,
    bonds: Optional[BondSet] = None,
    bond_note: Optional[str] = None,
) -> Finding:
    """
    Check one trajectory against an already-read topology

    `bonds` are the topology's bonds for TOPOLOGY MISMATCH; without them,
    `bond_note` says why, and is reported as the reason the check was skipped.
    """

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
        jump_frames=0,
        jump_runs=[],
        time_breaks=[],
        timestep_ps=None,
        read_error=None,
        problems=[],
        bond_frames=[],
        bond_worst=[],
        bond_skipped=None,
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

    samples = {} if bonds is not None and not headers_only else None

    try:
        if lowered.endswith(NETCDF_SUFFIXES):
            problems = check_netcdf(path, blank, headers_only, samples)
        else:
            problems = check_via_mdanalysis(
                path, blank, headers_only, lowered.endswith(".dcd"), samples
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

    if not headers_only and not (
        {"ATOM MISMATCH", "EMPTY TRAJECTORY"} & set(problems)
    ):
        if bonds is None:
            blank.update(bond_skipped=bond_note)
        elif blank["traj_atoms"] != len(bonds.labels):
            blank.update(bond_skipped="the trajectory's atom count is unknown")
        else:
            skip = {
                frame
                for start, end in blank["zero_runs"] + blank["bad_coord_runs"]
                for frame in range(start, end + 1)
            }
            try:
                stats, worst, mismatch = check_bonds(bonds, samples, skip)
            except Exception as err:
                blank.update(bond_skipped=f"bonds could not be measured ({err})")
            else:
                blank.update(bond_frames=stats, bond_worst=worst)
                if not stats:
                    blank.update(bond_skipped="no frame holds positions to test")
                if mismatch:
                    problems.append("TOPOLOGY MISMATCH")

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
        jump_frames=0,
        jump_runs=[],
        time_breaks=[],
        timestep_ps=None,
        read_error=None,
        problems=[],
        bond_frames=[],
        bond_worst=[],
        bond_skipped=None,
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
    # works without it, and only ATOM MISMATCH and TOPOLOGY MISMATCH are
    # lost. Saying so beats refusing to look at the trajectory at all. Under
    # --headers-only the bonds are not needed, so only the count is read.
    top = Topology(None, None, "no topology is named in the metadata")
    if topology:
        top_path = os.path.join(directory, topology)
        if headers_only:
            try:
                top = Topology(count_atoms(top_path), None, None)
            except Exception:
                pass
        else:
            top = read_topology(top_path)

    return [
        check_trajectory(
            directory,
            name,
            topology,
            top.atoms,
            headers_only,
            top.bonds,
            top.bond_note,
        )
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
        elif problem == "FRAME JUMPS":
            parts.append(
                f"FRAME JUMPS {finding.jump_frames} of {finding.frames} "
                f"(frames {format_runs(finding.jump_runs)})"
            )
        elif problem == "IMPLAUSIBLE TIMESTEP":
            parts.append(f"IMPLAUSIBLE TIMESTEP {finding.timestep_ps:g} ps")
        elif problem == "FRAME SCAN INCOMPLETE":
            parts.append(f"FRAME SCAN INCOMPLETE ({finding.read_error})")
        elif problem == "TOPOLOGY MISMATCH":
            parts.append(f"TOPOLOGY MISMATCH {stretched_bonds(finding)}")
        else:
            parts.append(problem)

    return "; ".join(parts)


# --------------------------------------------------
def stretched_bonds(finding: Finding) -> str:
    """Bonds stretched per frame tested: "frame 0 812 of 3,365 bonds, ..." """

    return ", ".join(
        f"frame {frame} {stretched:,} of {checked:,} bonds stretched"
        for frame, stretched, checked in finding.bond_frames
    )


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
    jumps = having("FRAME JUMPS")
    no_time = having("NO TIME AXIS")
    bad_step = having("IMPLAUSIBLE TIMESTEP")
    incomplete = having("FRAME SCAN INCOMPLETE")
    mismatched = having("TOPOLOGY MISMATCH")
    bond_skips = sorted(
        [f for f in findings if f.bond_skipped and not f.error],
        key=lambda f: f.trajectory,
    )
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

    if jumps:
        section += 1
        out += [
            "",
            f"{section}. ATOMS JUMP IMPOSSIBLY FAR BETWEEN FRAMES "
            f"({sims(len(jumps))})",
            "",
            f"   In some frames atoms have moved {MAX_JUMP:,.0f} angstroms or more",
            "   (or twice the box diagonal, if that is larger) since the frame",
            "   before -- often a whole molecule, moved as one piece, in the",
            "   last frame of a run. No simulation moves an atom that far in",
            "   one frame, and wrapping in a periodic box cannot either. These",
            "   frames need removing or rewriting. If the file joins separate",
            "   runs, the jump is not counted where the time axis shows the",
            "   join (a gap, or time starting again), so check that each run's",
            "   times were kept.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.jump_frames} of {f.frames} frames "
            f"(frames {format_runs(f.jump_runs)})"
            for f in jumps
        ]

    if mismatched:
        section += 1
        out += [
            "",
            f"{section}. TRAJECTORY FRAMES DO NOT FIT THE TOPOLOGY'S BONDS "
            f"({sims(len(mismatched))})",
            "",
            "   In the first, middle or last frame, "
            f"{BAD_BOND_SHARE:.0%} or more of the topology's",
            "   bonds are stretched past any real bond length.",
            "   The atom count agrees, but the coordinates do not belong to this",
            "   topology: usually a topology from another run or another strip",
            "   step, or the atoms in another order. Each of these needs the",
            "   topology the trajectory was written with. If the system is",
            "   coarse-grained or otherwise unusual, say so when submitting.",
            "",
        ]
        for f in mismatched:
            out.append(f"   {f.trajectory} against {f.topology}: {stretched_bonds(f)}")
            out += [f"      {line}" for line in f.bond_worst]

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

    if bond_skips:
        out += [
            "",
            f"BOND CHECK SKIPPED ({len(bond_skips)} trajectories)",
            "",
            "   These were not tested against their topology's bonds, so a",
            "   mismatched pair among them would not be noticed. Nothing is",
            "   wrong with them on that account.",
            "",
        ]
        out += [
            f"   {f.trajectory}: {f.bond_skipped}" for f in bond_skips
        ]

    if headers_only:
        out += [
            "",
            "Note: run without --headers-only to also scan the frames, which",
            "is what finds all-zero frames, coordinates that are not",
            "positions, frame jumps, and frames that do not fit the topology.",
            "Only NetCDF had its unit cell checked in this run.",
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

    # Not a problem, so it does not set the exit status, but it is said.
    skipped = len([f for f in findings if f.bond_skipped and not f.error])
    if skipped:
        print(
            f"  (bond check skipped for {skipped}; the report says why)",
            file=sys.stderr,
        )

    if args.report:
        with open(args.report, "wt") as out_fh:
            out_fh.write(build_report(findings, args.headers_only))
            out_fh.write("\n")
        print(f"\nReport written to '{args.report}'", file=sys.stderr)

    sys.exit(1 if counts else 0)


# --------------------------------------------------
if __name__ == "__main__":
    main()
