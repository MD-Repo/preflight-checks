"""Tests for check_simulations.py

Run with:

    uv run --with pytest --with mdanalysis pytest tests/

Same corpus as `test_check_amber.py`, plus the formats the older script
cannot read. Where a case exists in both files it asserts the same verdict,
because the two must not disagree about whether a submission is sound.

The cases are the defects that have actually arrived. One submission sent a
block of frames written as exact zeros. A second sent frames of uninitialised
memory: coordinates to the largest number a 32-bit float holds, some not
finite, sometimes under a unit cell of the same garbage and sometimes under a
box identical to its neighbours'.

Two behaviours here are worth reading before changing anything.

`test_a_missing_time_axis_is_only_claimed_for_netcdf` pins the reason
NO TIME AXIS is not extended to the other formats: MDAnalysis reports
0, 1, 2, 3 ... ps for a trajectory carrying no time information at all, which
cannot be told apart from a real 1 ps/frame run. The check reads the NetCDF
container directly instead, and stays quiet elsewhere rather than guessing.

`test_the_xtc_saturation_value_is_caught` pins the reason the coordinate check
tests magnitude and not just finiteness: an XTC cannot store a NaN, and
saturates it to a finite 21,474,836 angstroms.
"""

import os
import sys

import numpy as np
import pytest
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_simulations as c  # noqa: E402

ATOMS = 40
FRAMES = 6


# --------------------------------------------------
def write_nc(path, coords, lengths, angles, with_time=True, times=None):
    """
    Write a minimal AMBER-convention NetCDF trajectory

    `times` overrides the default 7 ps spacing; `with_time=False` leaves the
    time variable out.
    """

    out = netcdf_file(str(path), "w", version=2)
    out.Conventions = "AMBER"
    out.ConventionVersion = "1.0"
    out.createDimension("frame", None)
    out.createDimension("spatial", 3)
    out.createDimension("atom", coords.shape[1])
    out.createDimension("cell_spatial", 3)
    out.createDimension("cell_angular", 3)
    out.createDimension("label", 5)

    var = out.createVariable("spatial", "c", ("spatial",))
    var[:] = np.array(list("xyz"), dtype="c")
    var = out.createVariable("cell_spatial", "c", ("cell_spatial",))
    var[:] = np.array(list("abc"), dtype="c")
    var = out.createVariable("cell_angular", "c", ("cell_angular", "label"))
    var[:] = np.array([list("alpha"), list("beta "), list("gamma")], dtype="c")

    if with_time:
        var = out.createVariable("time", "f", ("frame",))
        var.units = "picosecond"
        if times is None:
            var[:] = np.arange(coords.shape[0], dtype=np.float32) * 7.0
        else:
            var[:] = np.asarray(times, dtype=np.float32)

    var = out.createVariable("coordinates", "f", ("frame", "atom", "spatial"))
    var.units = "angstrom"
    var[:] = coords

    var = out.createVariable("cell_lengths", "d", ("frame", "cell_spatial"))
    var.units = "angstrom"
    var[:] = lengths

    var = out.createVariable("cell_angles", "d", ("frame", "cell_angular"))
    var.units = "degree"
    var[:] = angles

    out.close()
    return str(path)


# --------------------------------------------------
def write_via_mdanalysis(path, coords, dimensions, times=None):
    """
    Write a trajectory in whatever format the extension names

    `times` sets each frame's time in ps. XTC and TRR store it as given; DCD
    stores only a start and an interval, so it reads back as regular whatever
    is passed.
    """

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mda.Universe.empty(coords.shape[1], trajectory=True)
    universe.load_new(coords, format=MemoryReader, dimensions=dimensions)
    with mda.Writer(str(path), coords.shape[1]) as writer:
        for step in universe.trajectory:
            if times is not None:
                step.time = times[step.frame]
            writer.write(universe.atoms)

    return str(path)


# --------------------------------------------------
def healthy():
    """Coordinates, lengths and angles of a trajectory with nothing wrong"""

    rng = np.random.default_rng(2339)
    coords = rng.uniform(-40, 40, (FRAMES, ATOMS, 3)).astype(np.float32)
    lengths = np.full((FRAMES, 3), 75.09, dtype=np.float64)
    angles = np.full((FRAMES, 3), 109.471219, dtype=np.float64)
    return coords, lengths, angles


# --------------------------------------------------
def boxes(n, lengths=(75.09, 75.09, 75.09), angles=(109.471219,) * 3):
    return np.tile(
        np.array(list(lengths) + list(angles), dtype=np.float32), (n, 1)
    )


# --------------------------------------------------
def write_pdb_topology(path, natom):
    """A PDB with a known atom count, readable by MDAnalysis"""

    with open(path, "wt") as out:
        for i in range(natom):
            out.write(
                f"ATOM  {i + 1:>5}  CA  ALA A{i + 1:>4}    "
                f"{0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00  0.00           C\n"
            )
        out.write("END\n")

    return str(path)


# --------------------------------------------------
def make_dir(tmp_path, name, trajectory_writer, top_atoms=ATOMS):
    """One simulation directory laid out the way a submission is"""

    trajectory_writer(tmp_path / name)
    write_pdb_topology(tmp_path / "top.pdb", top_atoms)
    (tmp_path / "mdrepo-metadata.toml").write_text(
        f'trajectory_file_names = ["{name}"]\n'
        'topology_file_name = "top.pdb"\n'
        'structure_file_name = "top.pdb"\n'
    )
    return str(tmp_path)


# --------------------------------------------------
def only(directory, headers_only=False):
    """The one finding for a directory declaring one trajectory"""

    findings = c.check_dir(directory, headers_only)
    assert len(findings) == 1
    assert findings[0].error is None, findings[0].error
    return findings[0]


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
def test_a_healthy_trajectory_is_clean_in_every_format(tmp_path, ext):
    coords, lengths, angles = healthy()

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles)
        return write_via_mdanalysis(path, coords, boxes(FRAMES))

    finding = only(make_dir(tmp_path, f"traj.{ext}", write))

    assert finding.problems == []
    assert finding.frames == FRAMES
    assert finding.traj_atoms == ATOMS


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
def test_a_broken_box_is_caught_in_every_format(tmp_path, ext):
    coords, lengths, angles = healthy()
    lengths[2] = 0.0
    angles[2] = 0.0

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles)
        dims = boxes(FRAMES)
        dims[2] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return write_via_mdanalysis(path, coords, dims)

    finding = only(make_dir(tmp_path, f"traj.{ext}", write))

    assert "BAD UNIT CELL" in finding.problems
    assert finding.bad_cell_runs == [(2, 2)]



# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "trr", "dcd"])
def test_non_finite_coordinates_are_caught(tmp_path, ext):
    """
    XTC is excluded on purpose: it cannot hold a NaN

    What it does instead is the next test.
    """

    coords, lengths, angles = healthy()
    coords[4, 3] = np.nan

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles)
        return write_via_mdanalysis(path, coords, boxes(FRAMES))

    finding = only(make_dir(tmp_path, f"traj.{ext}", write))

    assert "BAD COORDINATES" in finding.problems
    assert finding.bad_coord_runs == [(4, 4)]


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "trr", "dcd"])
def test_the_xtc_saturation_value_is_caught(tmp_path, ext):
    """
    21,474,836 angstroms is finite, non-zero, and impossible

    It is what an XTC turns a NaN into, and every trajectory MDRepo writes is
    an XTC. A finiteness test alone would pass it.

    XTC is not in the parameters, and the reason is worth recording: asking
    MDAnalysis's XTC writer to store this exact value aborts the process --
    "Internal overflow compressing coordinates", then a corrupted heap. The
    value is at the limit of the format's integer encoding, which is why it
    is the saturation value in the first place. A real XTC carrying it comes
    from cpptraj, not from a test fixture. The rule itself is exercised on
    XTC by the test below.
    """

    coords, lengths, angles = healthy()
    coords[1, 9] = 2.1474836e7

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles)
        return write_via_mdanalysis(path, coords, boxes(FRAMES))

    finding = only(make_dir(tmp_path, f"traj.{ext}", write))

    assert "BAD COORDINATES" in finding.problems
    assert finding.bad_coord_runs == [(1, 1)]


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
def test_an_impossible_magnitude_is_caught_in_every_format(tmp_path, ext):
    """
    The magnitude rule itself, on a value every writer can store

    5e6 angstroms is 0.5 mm: five times over the limit, and well inside what
    an XTC can encode, so this covers the format the saturation test cannot.
    """

    coords, lengths, angles = healthy()
    coords[2, 11] = 5.0e6

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles)
        return write_via_mdanalysis(path, coords, boxes(FRAMES))

    finding = only(make_dir(tmp_path, f"traj.{ext}", write))

    assert "BAD COORDINATES" in finding.problems
    assert finding.bad_coord_runs == [(2, 2)]


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
def test_all_zero_frames_are_caught_in_every_format(tmp_path, ext):
    coords, lengths, angles = healthy()
    coords[2:4] = 0.0

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles)
        return write_via_mdanalysis(path, coords, boxes(FRAMES))

    finding = only(make_dir(tmp_path, f"traj.{ext}", write))

    assert "ZERO FRAMES" in finding.problems
    assert finding.zero_runs == [(2, 3)]


# --------------------------------------------------
def test_atom_mismatch_is_caught_through_the_topology(tmp_path):
    """The stripped-trajectory-with-unstripped-topology case"""

    coords, lengths, angles = healthy()

    finding = only(
        make_dir(
            tmp_path,
            "traj.nc",
            lambda p: write_nc(p, coords, lengths, angles),
            top_atoms=ATOMS + 11,
        )
    )

    assert "ATOM MISMATCH" in finding.problems
    assert finding.traj_atoms == ATOMS
    assert finding.top_atoms == ATOMS + 11


# --------------------------------------------------
def test_a_missing_time_axis_is_only_claimed_for_netcdf(tmp_path):
    """
    The reason NO TIME AXIS stops at NetCDF

    MDAnalysis reports 0, 1, 2, 3 ... ps for a trajectory with no time
    information at all, which is indistinguishable from a real 1 ps/frame
    run. So the check reads the NetCDF container directly, and says nothing
    about formats where it cannot tell. Asserting the silence keeps anyone
    from "improving" this into a guess.
    """

    coords, lengths, angles = healthy()

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    netcdf = only(
        make_dir(
            tmp_path / "a",
            "traj.nc",
            lambda p: write_nc(p, coords, lengths, angles, with_time=False),
        )
    )
    assert "NO TIME AXIS" in netcdf.problems

    xtc = only(
        make_dir(
            tmp_path / "b",
            "traj.xtc",
            lambda p: write_via_mdanalysis(p, coords, boxes(FRAMES)),
        )
    )
    assert "NO TIME AXIS" not in xtc.problems


# --------------------------------------------------
def test_a_netcdf_that_has_a_time_axis_is_not_reported(tmp_path):
    coords, lengths, angles = healthy()

    finding = only(
        make_dir(
            tmp_path,
            "traj.nc",
            lambda p: write_nc(p, coords, lengths, angles, with_time=True),
        )
    )

    assert "NO TIME AXIS" not in finding.problems


# --------------------------------------------------
def test_an_empty_file_is_reported_not_crashed_on(tmp_path):
    finding = only(
        make_dir(tmp_path, "traj.nc", lambda p: open(p, "wb").close())
    )

    assert finding.problems == ["EMPTY TRAJECTORY"]
    assert finding.frames == 0


# --------------------------------------------------
def test_the_unit_cell_is_checked_for_netcdf_under_headers_only(tmp_path):
    """
    The cheap path survives --headers-only, as it does in check_amber.py

    Three numbers per frame against tens of thousands of coordinates.
    """

    coords, lengths, angles = healthy()
    lengths[1] = [1.46e233, -1.04e-306, -1.60e-203]

    directory = make_dir(
        tmp_path, "traj.nc", lambda p: write_nc(p, coords, lengths, angles)
    )
    finding = only(directory, headers_only=True)

    assert finding.problems == ["BAD UNIT CELL"]
    assert finding.bad_cell_runs == [(1, 1)]


# --------------------------------------------------
def test_headers_only_does_not_scan_frames(tmp_path):
    coords, lengths, angles = healthy()
    coords[3, 7] = np.inf

    directory = make_dir(
        tmp_path, "traj.nc", lambda p: write_nc(p, coords, lengths, angles)
    )

    assert only(directory, headers_only=True).problems == []
    assert "BAD COORDINATES" in only(directory).problems


# --------------------------------------------------
def test_a_trajectory_with_no_periodic_box_is_not_a_defect(tmp_path):
    coords, lengths, angles = healthy()
    lengths[:] = 0.0
    angles[:] = 0.0

    finding = only(
        make_dir(
            tmp_path, "traj.nc", lambda p: write_nc(p, coords, lengths, angles)
        )
    )

    assert finding.problems == []


# --------------------------------------------------
def test_an_unreadable_topology_loses_only_the_atom_check(tmp_path):
    """
    A topology we cannot parse must not stop us looking at the frames

    Only ATOM MISMATCH depends on it, and refusing the whole trajectory over
    it would hide the defects that actually cost us tickets.
    """

    coords, lengths, angles = healthy()
    coords[2] = 0.0
    write_nc(tmp_path / "traj.nc", coords, lengths, angles)
    (tmp_path / "top.parm7").write_text("not a topology anyone can read\n")
    (tmp_path / "mdrepo-metadata.toml").write_text(
        'trajectory_file_names = ["traj.nc"]\n'
        'topology_file_name = "top.parm7"\n'
    )

    finding = only(str(tmp_path))

    assert finding.top_atoms is None
    assert "ATOM MISMATCH" not in finding.problems
    assert "ZERO FRAMES" in finding.problems


# --------------------------------------------------
def test_a_format_with_no_reader_is_reported(tmp_path):
    write_pdb_topology(tmp_path / "top.pdb", ATOMS)
    (tmp_path / "traj.dtr").write_bytes(b"a desmond trajectory")
    (tmp_path / "mdrepo-metadata.toml").write_text(
        'trajectory_file_names = ["traj.dtr"]\n'
        'topology_file_name = "top.pdb"\n'
    )

    findings = c.check_dir(str(tmp_path), False)

    assert len(findings) == 1
    assert findings[0].error is not None
    assert "no reader" in findings[0].error


# --------------------------------------------------
def test_every_declared_trajectory_is_checked(tmp_path):
    """Reporting only the first would hide every later defect"""

    coords, lengths, angles = healthy()
    write_nc(tmp_path / "good.nc", coords, lengths, angles)
    bad = coords.copy()
    bad[5] = 0.0
    write_nc(tmp_path / "bad.nc", bad, lengths, angles)
    write_pdb_topology(tmp_path / "top.pdb", ATOMS)
    (tmp_path / "mdrepo-metadata.toml").write_text(
        'trajectory_file_names = ["good.nc", "bad.nc"]\n'
        'topology_file_name = "top.pdb"\n'
    )

    findings = c.check_dir(str(tmp_path), False)

    assert len(findings) == 2
    assert findings[0].problems == []
    assert "ZERO FRAMES" in findings[1].problems


# --------------------------------------------------
def test_the_rules_match_check_amber(tmp_path):
    """
    The two scripts must not disagree about the same bytes

    check_amber.py is kept for historical reasons and still runs on NetCDF,
    so a submitter could reasonably run either. Same file, same verdict.
    """

    import check_amber

    coords, lengths, angles = healthy()
    coords[1, 4] = np.nan
    coords[3] = 0.0
    lengths[5] = 0.0
    angles[5] = 0.0

    write_nc(tmp_path / "traj.nc", coords, lengths, angles)
    write_pdb_topology(tmp_path / "top.pdb", ATOMS)

    assert c.scan_cell(lengths, angles) == check_amber.scan_cell(
        lengths, angles
    )

    zero, nonfinite, huge = check_amber.scan_coordinates(coords)
    mine = [
        i
        for i in range(coords.shape[0])
        if c.classify_frame(coords[i]) is not None
    ]

    assert sorted(set(zero) | set(nonfinite) | set(huge)) == mine


# --------------------------------------------------
def test_runs_are_grouped():
    assert c.group_runs([]) == []
    assert c.group_runs([4]) == [(4, 4)]
    assert c.group_runs([1, 2, 3, 9, 10, 40]) == [(1, 3), (9, 10), (40, 40)]


# --------------------------------------------------
def test_frame_classification_is_exclusive():
    assert c.classify_frame(np.zeros((ATOMS, 3))) == "zero"
    assert c.classify_frame(np.full((ATOMS, 3), np.nan)) == "nonfinite"
    assert c.classify_frame(np.full((ATOMS, 3), 2.1474836e7)) == "huge"
    assert c.classify_frame(np.full((ATOMS, 3), 950.0)) is None


# ==================================================
# FRAME JUMPS
#
# The same rule as MDRepo's screen_trajectory.py (simulation-processing
# 7884c2e), tested on the same shapes. A frame is flagged when an atom moves
# MAX_JUMP (5,000 A) or more from the frame before -- or twice the box
# diagonal, if larger -- unless the step is a boundary in time into a block of
# two or more frames.
# ==================================================

CHAIN = 20  # the first CHAIN atoms play the protein chain the damage moves


# --------------------------------------------------
def steady(frames=FRAMES, atoms=ATOMS, box=50.0):
    """A trajectory whose atoms stay put apart from thermal noise"""

    rng = np.random.default_rng(88497)
    start = rng.uniform(5.0, box - 5.0, (atoms, 3))
    noise = rng.normal(0.0, 0.3, (frames, atoms, 3))
    coords = (start + noise).astype(np.float32)
    lengths = np.full((frames, 3), box, dtype=np.float64)
    angles = np.full((frames, 3), 90.0, dtype=np.float64)
    return coords, lengths, angles


# --------------------------------------------------
def offset(distance):
    """One constant vector of the given length, like the offsets measured"""

    v = np.array([37144.0, 81529.0, 233850.0])
    return (v / np.linalg.norm(v) * distance).astype(np.float32)


# --------------------------------------------------
def writer(ext, coords, lengths, angles, times=None, with_time=True):
    """A trajectory writer for make_dir, in the format ext names"""

    def write(path):
        if ext == "nc":
            return write_nc(path, coords, lengths, angles, with_time, times)
        box = lengths[0][0]
        dims = boxes(len(coords), lengths=(box, box, box), angles=(90.0,) * 3)
        return write_via_mdanalysis(path, coords, dims, times)

    return write


# --------------------------------------------------
def test_the_jump_rule_matches_the_pipeline():
    """The numbers and the decisions screen_trajectory.py makes"""

    assert c.MAX_JUMP == 5.0e3
    assert c.STEP_TOLERANCE == 0.01
    assert c.jump_limit(None) == c.MAX_JUMP
    assert c.jump_limit(273.0) == c.MAX_JUMP
    assert c.jump_limit(5196.2) == pytest.approx(2 * 5196.2)
    assert c.box_diagonal([158.0] * 3, [90.0] * 3) == pytest.approx(158.0 * 3**0.5)
    assert c.box_diagonal([0.0] * 3, [90.0] * 3) is None

    big = 250_425.0
    assert c.find_jumps([0, 4, 4, 4, 4, big], [0, 1, 2, 3, 4, 5], None) == ([5], [])
    # a gap into a block of three frames excuses the jump
    assert c.find_jumps(
        [0, 4, 4, big, 4, 4], [0, 10, 20, 5000, 5010, 5020], None
    ) == ([], [3])
    # a clock restart does too
    assert c.find_jumps([0, 4, 4, big, 4, 4], [0, 10, 20, 0, 10, 20], None) == (
        [],
        [3],
    )
    # a one-frame block after a gap is not excused
    assert c.find_jumps([0, 4, 4, 4, 4, big], [0, 1, 2, 3, 4, 9.9e9], None)[0] == [5]
    # no usable times: every step tested
    assert c.find_jumps([0, 4, 4, big, 4, 4], None, None) == ([3], [])
    assert c.find_jumps([0, 4, 4, big, 4, 4], [0.0] * 6, None) == ([3], [])


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
@pytest.mark.parametrize("distance", [15_425.0, 250_425.0])
def test_a_ddd_damaged_last_frame_is_caught_in_every_format(tmp_path, ext, distance):
    """
    The last frame moves the first chain by one vector, well under 1e6 A

    15,425 A is the smallest median offset measured (3fa3 Pro_lig30),
    250,425 A the 6mlh one. BAD COORDINATES misses both.
    """

    coords, lengths, angles = steady()
    coords[FRAMES - 1, :CHAIN] += offset(distance)

    finding = only(make_dir(tmp_path, f"traj.{ext}", writer(ext, coords, lengths, angles)))

    assert "FRAME JUMPS" in finding.problems
    assert "BAD COORDINATES" not in finding.problems
    assert finding.jump_runs == [(FRAMES - 1, FRAMES - 1)]
    assert "FRAME JUMPS 1 of 6 (frames 5)" in c.describe(finding)


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc"])
def test_a_wrapped_trajectory_is_clean(tmp_path, ext):
    """MDR00088497: a 158 A box, atoms crossing it by a whole diagonal"""

    box = 158.0
    coords, lengths, angles = steady(box=box)
    for frame in range(1, FRAMES, 2):
        coords[frame, 0] += box

    finding = only(make_dir(tmp_path, f"traj.{ext}", writer(ext, coords, lengths, angles)))

    assert finding.problems == []


# --------------------------------------------------
def test_an_unwrapped_trajectory_far_from_the_origin_is_clean(tmp_path):
    """Past 20,000 A from the origin, moving 5 A a frame"""

    coords, lengths, angles = steady()
    for frame in range(FRAMES):
        coords[frame] += np.float32(19_990.0 + 5.0 * frame)

    finding = only(make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles)))

    assert finding.problems == []


# --------------------------------------------------
def test_a_huge_box_raises_the_limit_but_not_without_bound(tmp_path):
    box = 3000.0
    coords, lengths, angles = steady(box=box)
    coords[3, 0] += box

    finding = only(make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles)))
    assert finding.problems == []

    coords[FRAMES - 1, :CHAIN] += offset(20_000.0)
    finding = only(make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles)))
    assert "FRAME JUMPS" in finding.problems


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr"])
@pytest.mark.parametrize(
    "times",
    [[0.0, 10.0, 20.0, 5000.0, 5010.0, 5020.0], [0.0, 10.0, 20.0, 0.0, 10.0, 20.0]],
    ids=["gap", "clock-restart"],
)
def test_joined_runs_are_clean_and_the_join_is_recorded(tmp_path, ext, times):
    coords, lengths, angles = steady()
    coords[3:, :CHAIN] += offset(20_000.0)

    finding = only(
        make_dir(tmp_path, f"traj.{ext}", writer(ext, coords, lengths, angles, times))
    )

    assert finding.problems == []
    assert finding.time_breaks == [3]


# --------------------------------------------------
def test_dcd_cannot_carry_a_join_so_its_jumps_are_flagged(tmp_path):
    coords, lengths, angles = steady()
    coords[3:, :CHAIN] += offset(20_000.0)
    times = [0.0, 10.0, 20.0, 5000.0, 5010.0, 5020.0]

    finding = only(
        make_dir(tmp_path, "traj.dcd", writer("dcd", coords, lengths, angles, times))
    )

    assert "FRAME JUMPS" in finding.problems
    assert finding.jump_runs == [(3, 3)]


# --------------------------------------------------
@pytest.mark.parametrize("last_time", [9.9e9, 0.0], ids=["garbage-time", "zero-time"])
def test_a_damaged_last_frame_with_a_damaged_time_is_still_flagged(tmp_path, last_time):
    coords, lengths, angles = steady()
    coords[FRAMES - 1, :CHAIN] += offset(250_425.0)
    times = [0.0, 1.0, 2.0, 3.0, 4.0, last_time]

    finding = only(
        make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles, times))
    )

    assert "FRAME JUMPS" in finding.problems
    assert finding.jump_runs == [(FRAMES - 1, FRAMES - 1)]


# --------------------------------------------------
def test_a_netcdf_with_no_time_axis_has_every_step_tested(tmp_path):
    """Contributor 289's MDR00048669 has no time variable"""

    coords, lengths, angles = steady()
    coords[3:, :CHAIN] += offset(20_000.0)

    finding = only(
        make_dir(
            tmp_path, "traj.nc", writer("nc", coords, lengths, angles, with_time=False)
        )
    )

    assert "NO TIME AXIS" in finding.problems
    assert "FRAME JUMPS" in finding.problems
    assert finding.jump_runs == [(3, 3)]


# --------------------------------------------------
def test_a_jump_across_a_read_chunk_is_caught(tmp_path):
    """NetCDF is read CHUNK_FRAMES at a time; the boundary must be compared"""

    frames = c.CHUNK_FRAMES + 10
    coords, lengths, angles = steady(frames=frames)
    coords[c.CHUNK_FRAMES :, :CHAIN] += offset(20_000.0)

    finding = only(make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles)))

    assert finding.jump_runs == [(c.CHUNK_FRAMES, c.CHUNK_FRAMES)]


# --------------------------------------------------
def test_headers_only_does_not_look_for_jumps(tmp_path):
    coords, lengths, angles = steady()
    coords[FRAMES - 1, :CHAIN] += offset(250_425.0)

    finding = only(
        make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles)),
        headers_only=True,
    )

    assert "FRAME JUMPS" not in finding.problems


# --------------------------------------------------
def test_the_report_explains_frame_jumps(tmp_path):
    coords, lengths, angles = steady()
    coords[FRAMES - 1, :CHAIN] += offset(250_425.0)

    finding = only(make_dir(tmp_path, "traj.nc", writer("nc", coords, lengths, angles)))
    report = c.build_report([finding], headers_only=False)

    assert "ATOMS JUMP IMPOSSIBLY FAR BETWEEN FRAMES (1 simulation)" in report
    assert "traj.nc: 1 of 6 frames (frames 5)" in report


# --------------------------------------------------
# TOPOLOGY MISMATCH
#
# A small "protein": RESIDUES residues of N, CA, C, O with real bond lengths,
# each residue BACKBONE_GAP apart so no bond is guessed between them. Its
# guessed bonds are N-CA, CA-C and C-O in each residue: 30 bonds, 40 atoms.

RESIDUES = 10
BACKBONE = [
    ("N", (0.0, 0.0, 0.0)),
    ("CA", (1.46, 0.0, 0.0)),
    ("C", (2.21, 1.30, 0.0)),
    ("O", (2.21, 2.53, 0.0)),
]


# --------------------------------------------------
def protein():
    """Names, residue numbers and coordinates of the small protein"""

    names, resids, coords = [], [], []
    for res in range(RESIDUES):
        origin = np.array([8.0 + 10.0 * (res % 4), 8.0 + 10.0 * (res // 4), 20.0])
        for name, xyz in BACKBONE:
            names.append(name)
            resids.append(res + 1)
            coords.append(origin + np.array(xyz))

    return names, resids, np.array(coords, dtype=np.float32)


# --------------------------------------------------
def write_pdb(path, names, resids, coords, resname="ALA", box=None):
    """A PDB with real coordinates, so bonds can be guessed from it"""

    with open(path, "wt") as out:
        if box is not None:
            out.write(
                f"CRYST1{box:9.3f}{box:9.3f}{box:9.3f}"
                f"{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1\n"
            )
        for i, (name, resid, xyz) in enumerate(zip(names, resids, coords)):
            out.write(
                f"ATOM  {i + 1:>5} {name:<4} {resname} A{resid:>4}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
                f"          {name[0]:>2}\n"
            )
        out.write("END\n")

    return str(path)


# --------------------------------------------------
def write_psf(path, atoms, bonds):
    """
    A minimal CHARMM PSF: `atoms` as (resname, resid, name, mass), `bonds` as
    1-based pairs. The bonds come from the file, not from any coordinates.
    """

    lines = ["PSF", "", "       1 !NTITLE", " REMARKS test", ""]
    lines.append(f"{len(atoms):8d} !NATOM")
    for i, (resname, resid, name, mass) in enumerate(atoms):
        lines.append(
            f"{i + 1:8d} A    {resid:<4d} {resname:<4} {name:<4} {name:<4}"
            f" {0.0:10.6f}    {mass:10.4f}           0"
        )
    lines += ["", f"{len(bonds):8d} !NBOND: bonds"]
    for start in range(0, len(bonds), 4):
        lines.append(
            "".join(f"{a:8d}{b:8d}" for a, b in bonds[start : start + 4])
        )
    for section in ("!NTHETA: angles", "!NPHI: dihedrals", "!NIMPHI: impropers"):
        lines += ["", f"       0 {section}"]
    lines.append("")

    (path).write_text("\n".join(lines) + "\n")
    return str(path)


# --------------------------------------------------
def frames_of(coords, frames=FRAMES, seed=4207):
    """A trajectory of the structure with thermal noise, as float32"""

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.03, (frames,) + coords.shape)
    return (coords[None, :, :] + noise).astype(np.float32)


# --------------------------------------------------
def bond_dir(tmp_path, traj, topology_writer, ext="xtc", box=60.0, top="top.pdb"):
    """A directory with `traj` under `topology_writer`'s topology"""

    lengths = np.full((len(traj), 3), box, dtype=np.float64)
    angles = np.full((len(traj), 3), 90.0, dtype=np.float64)
    name = f"traj.{ext}"
    writer(ext, traj, lengths, angles)(tmp_path / name)
    topology_writer(tmp_path / top)
    (tmp_path / "mdrepo-metadata.toml").write_text(
        f'trajectory_file_names = ["{name}"]\ntopology_file_name = "{top}"\n'
    )
    return str(tmp_path)


# --------------------------------------------------
def shuffled(coords, seed=11):
    """The same atoms in another order: the same count, the wrong pairing"""

    return coords[:, np.random.default_rng(seed).permutation(coords.shape[1]), :]


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
def test_a_trajectory_that_fits_its_topology_is_clean(tmp_path, ext):
    names, resids, coords = protein()
    finding = only(
        bond_dir(tmp_path, frames_of(coords), lambda p: write_pdb(p, names, resids, coords), ext)
    )

    assert finding.problems == []
    assert finding.bond_skipped is None
    assert [f for f, _, _ in finding.bond_frames] == [0, FRAMES // 2, FRAMES - 1]
    assert all(stretched == 0 and checked == 30 for _, stretched, checked in finding.bond_frames)


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc", "trr", "dcd"])
def test_atoms_in_another_order_are_a_topology_mismatch(tmp_path, ext):
    names, resids, coords = protein()
    finding = only(
        bond_dir(tmp_path, shuffled(frames_of(coords)), lambda p: write_pdb(p, names, resids, coords), ext)
    )

    assert finding.problems == ["TOPOLOGY MISMATCH"]
    assert finding.bond_worst
    assert "TOPOLOGY MISMATCH frame 0" in c.describe(finding)


# --------------------------------------------------
@pytest.mark.parametrize("frame", [FRAMES // 2, FRAMES - 1])
def test_a_mismatch_in_the_middle_or_last_frame_alone_is_caught(tmp_path, frame):
    names, resids, coords = protein()
    traj = frames_of(coords)
    traj[frame] = shuffled(traj)[frame]

    finding = only(
        bond_dir(tmp_path, traj, lambda p: write_pdb(p, names, resids, coords))
    )

    assert "TOPOLOGY MISMATCH" in finding.problems
    assert [(f, n > 0) for f, n, _ in finding.bond_frames] == [
        (f, f == frame) for f in (0, FRAMES // 2, FRAMES - 1)
    ]
    assert finding.bond_worst[0].startswith(f"frame {frame}:")


# --------------------------------------------------
def test_a_few_stretched_bonds_are_noise_not_a_mismatch(tmp_path):
    """One bond of 30 is 3%, under BAD_BOND_SHARE: reported in the counts only"""

    names, resids, coords = protein()
    traj = frames_of(coords)
    traj[:, 3] += np.float32([0.0, 4.0, 0.0])  # residue 1's O, off its C

    finding = only(
        bond_dir(tmp_path, traj, lambda p: write_pdb(p, names, resids, coords))
    )

    assert finding.problems == []
    assert all(n == 1 for _, n, _ in finding.bond_frames)


# --------------------------------------------------
@pytest.mark.parametrize("ext", ["nc", "xtc"])
def test_a_molecule_split_by_the_box_uses_the_minimum_image(tmp_path, ext):
    """
    Wrapping puts half of every residue at the far side of the box

    The raw distance across the split is ~58 A; through the box it is the
    real bond. Every residue is split, so without the minimum image all of
    them would fail.
    """

    names, resids, coords = protein()
    box = 60.0
    coords = coords.copy()
    coords[:, 0] -= coords[1::4, 0].repeat(4) - 0.7  # straddle x = 0
    traj = frames_of(coords) % box

    finding = only(
        bond_dir(tmp_path, traj, lambda p: write_pdb(p, names, resids, coords), ext, box=box)
    )

    assert finding.problems == []
    assert all(n == 0 for _, n, _ in finding.bond_frames)


# --------------------------------------------------
def test_the_declared_bonds_of_a_psf_are_used_and_tip3p_is_clean(tmp_path):
    """
    CHARMM's TIP3P declares an H1-H2 bond of 1.51 A, past the hydrogens'
    radii: BOND_FLOOR is what keeps every water from failing
    """

    atoms, bonds, coords = [], [], []
    for w in range(12):
        base = len(atoms) + 1
        atoms += [("TIP3", w + 1, "OH2", 15.9994), ("TIP3", w + 1, "H1", 1.008), ("TIP3", w + 1, "H2", 1.008)]
        bonds += [(base, base + 1), (base, base + 2), (base + 1, base + 2)]
        origin = np.array([5.0 + 6.0 * (w % 4), 5.0 + 6.0 * (w // 4), 10.0])
        coords += [origin, origin + [0.9572, 0.0, 0.0], origin + [-0.2400, 0.9266, 0.0]]
    coords = np.array(coords, dtype=np.float32)

    finding = only(
        bond_dir(tmp_path, frames_of(coords), lambda p: write_psf(p, atoms, bonds), top="top.psf")
    )
    assert finding.problems == []
    assert finding.bond_frames[0] == (0, 0, 36)

    mismatch = tmp_path / "mismatch"
    mismatch.mkdir()
    finding = only(
        bond_dir(mismatch, shuffled(frames_of(coords)), lambda p: write_psf(p, atoms, bonds), top="top.psf")
    )
    assert finding.problems == ["TOPOLOGY MISMATCH"]


# --------------------------------------------------
def test_a_coarse_grained_topology_is_skipped_and_says_why(tmp_path):
    """Martini beads weigh 72 u: "BB" is not boron, and nothing is judged"""

    atoms = [("ALA", i // 2 + 1, "BB" if i % 2 == 0 else "SC1", 72.0) for i in range(40)]
    bonds = [(i, i + 1) for i in range(1, 40)]
    coords = np.array([[5.0 + 4.7 * (i % 10), 5.0 + 4.7 * (i // 10), 10.0] for i in range(40)], dtype=np.float32)

    finding = only(
        bond_dir(tmp_path, shuffled(frames_of(coords)), lambda p: write_psf(p, atoms, bonds), top="top.psf")
    )

    assert finding.problems == []
    assert "only 0 of 39 bonds join recognisable elements" in finding.bond_skipped
    report = c.build_report([finding], headers_only=False)
    assert "BOND CHECK SKIPPED (1 trajectories)" in report


# --------------------------------------------------
def test_a_structure_with_no_bonds_to_guess_is_skipped(tmp_path):
    """The original fixture: every atom at the origin, so nothing is bonded"""

    coords, lengths, angles = healthy()
    finding = only(
        make_dir(tmp_path, "traj.nc", lambda p: write_nc(p, coords, lengths, angles))
    )

    assert finding.problems == []
    assert finding.bond_skipped == "no bonds could be guessed from the topology"


# --------------------------------------------------
def test_an_unreadable_topology_skips_the_bond_check(tmp_path):
    coords, lengths, angles = healthy()
    write_nc(tmp_path / "traj.nc", coords, lengths, angles)
    (tmp_path / "top.parm7").write_text("not a topology anyone can read\n")
    (tmp_path / "mdrepo-metadata.toml").write_text(
        'trajectory_file_names = ["traj.nc"]\ntopology_file_name = "top.parm7"\n'
    )

    finding = only(str(tmp_path))

    assert finding.problems == []
    assert finding.bond_skipped == "the topology could not be read"


# --------------------------------------------------
def test_frames_already_reported_are_not_tested_again(tmp_path):
    """An all-zero last frame is ZERO FRAMES, not also a mismatch"""

    names, resids, coords = protein()
    traj = frames_of(coords)
    traj[FRAMES - 1] = 0.0

    finding = only(
        bond_dir(tmp_path, traj, lambda p: write_pdb(p, names, resids, coords), "nc")
    )

    assert finding.problems == ["ZERO FRAMES"]
    assert [f for f, _, _ in finding.bond_frames] == [0, FRAMES // 2]


# --------------------------------------------------
def test_an_atom_mismatch_is_not_also_a_topology_mismatch(tmp_path):
    names, resids, coords = protein()
    finding = only(
        bond_dir(tmp_path, frames_of(coords[:-4]), lambda p: write_pdb(p, names, resids, coords))
    )

    assert finding.problems == ["ATOM MISMATCH"]
    assert finding.bond_frames == []
    assert finding.bond_skipped is None


# --------------------------------------------------
def test_headers_only_does_not_test_bonds(tmp_path):
    names, resids, coords = protein()
    finding = only(
        bond_dir(tmp_path, shuffled(frames_of(coords)), lambda p: write_pdb(p, names, resids, coords), "nc"),
        headers_only=True,
    )

    assert finding.problems == []
    assert finding.bond_frames == []
    assert finding.bond_skipped is None


# --------------------------------------------------
@pytest.mark.parametrize(
    "name, mass, declared, expected",
    [
        ("CA", 12.011, None, "C"),  # alpha carbon
        ("CA", 40.08, None, "CA"),  # calcium
        ("CLA", 35.45, None, "CL"),  # CHARMM chloride
        ("1HB", 1.008, None, "H"),  # leading digit
        ("HB1", 3.024, None, "H"),  # repartitioned hydrogen
        ("CB", 5.96, None, "C"),  # a methyl carbon after repartitioning
        ("FE", 55.85, None, "FE"),
        ("OW", None, None, "O"),  # no masses: the name decides
        ("MW", 0.0, None, None),  # TIP4P virtual site
        ("LP1", 0.0, None, None),  # lone pair
        ("BB", 72.0, None, None),  # Martini bead
        ("X1", 32.06, "S", "S"),  # the topology's element wins
    ],
)
def test_elements_come_from_name_and_mass(name, mass, declared, expected):
    assert c.assign_element(name, mass, declared) == expected


# --------------------------------------------------
def test_the_report_explains_a_topology_mismatch(tmp_path):
    names, resids, coords = protein()
    finding = only(
        bond_dir(tmp_path, shuffled(frames_of(coords)), lambda p: write_pdb(p, names, resids, coords))
    )
    report = c.build_report([finding], headers_only=False)

    assert "TRAJECTORY FRAMES DO NOT FIT THE TOPOLOGY'S BONDS (1 simulation)" in report
    assert "traj.xtc against top.pdb: frame 0 " in report
    assert "      frame " in report  # the worst bonds, indented under it


# --------------------------------------------------
# Clean controls and shuffled pairs from real engines, using MDAnalysis's own
# test files: CHARMM PSF + DCD, AMBER prmtop + NetCDF, GROMACS TPR + XTC, and
# a .gro and a .pdb whose bonds are guessed. Skipped without MDAnalysisTests:
#
#     uv run --with pytest --with mdanalysis --with mdanalysistests pytest tests/


ENGINES = [
    ("PSF", "DCD", "traj.dcd"),
    ("PRMncdf", "NCDF", "traj.nc"),
    ("TPR", "XTC", "traj.xtc"),
    ("GRO", "XTC", "traj.xtc"),
    ("PDB_small", "DCD", "traj.dcd"),
]


# --------------------------------------------------
@pytest.mark.parametrize("top_name, traj_name, out", ENGINES)
@pytest.mark.parametrize("shuffle", [False, True], ids=["clean", "shuffled"])
def test_real_engines(tmp_path, top_name, traj_name, out, shuffle):
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    import shutil

    import MDAnalysis as mda

    top = getattr(datafiles, top_name)
    universe = mda.Universe(top, getattr(datafiles, traj_name))
    order = np.arange(len(universe.atoms))
    if shuffle:
        third = len(order) // 3
        order[:third] = np.random.default_rng(1).permutation(order[:third])

    with mda.Writer(str(tmp_path / out), len(universe.atoms)) as writer_:
        for _ in universe.trajectory[:5]:
            writer_.write(universe.atoms[order])
    shutil.copy(top, tmp_path / os.path.basename(top))
    (tmp_path / "mdrepo-metadata.toml").write_text(
        f'trajectory_file_names = ["{out}"]\n'
        f'topology_file_name = "{os.path.basename(top)}"\n'
    )

    finding = only(str(tmp_path))

    if shuffle:
        assert finding.problems == ["TOPOLOGY MISMATCH"]
    else:
        assert finding.problems == []
        assert finding.bond_skipped is None
        assert all(n == 0 for _, n, _ in finding.bond_frames)
