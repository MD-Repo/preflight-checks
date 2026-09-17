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
def write_nc(path, coords, lengths, angles, with_time=True):
    """Write a minimal AMBER-convention NetCDF trajectory"""

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
        var[:] = np.arange(coords.shape[0], dtype=np.float32) * 7.0

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
def write_via_mdanalysis(path, coords, dimensions):
    """Write a trajectory in whatever format the extension names"""

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    universe = mda.Universe.empty(coords.shape[1], trajectory=True)
    universe.load_new(coords, format=MemoryReader, dimensions=dimensions)
    with mda.Writer(str(path), coords.shape[1]) as writer:
        for _ in universe.trajectory:
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
