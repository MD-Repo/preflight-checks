"""Tests for check_amber.py

Run with:

    uv run --with pytest pytest tests/

The cases are the defects that have actually arrived, not invented ones.

One submission sent a block of frames written as exact zeros -- coordinates,
box lengths, box angles and time all 0.0. A second, five days later, sent
frames of uninitialised memory: coordinates running to the largest number a
32-bit float holds, some of them not finite, under a unit cell of the same
kind of garbage.

The case that decides the shape of the checks is the one file of that second
submission that everyone believed was clean. Its two damaged frames carry an
ordinary box -- the same lengths and angles as their neighbours -- and every
damaged atom in them is solvent. A check that reads only the unit cell passes
it. The converted trajectory made from it holds a frame that crashes the
reader. That is `test_a_valid_box_does_not_excuse_bad_coordinates`, and it is
why the coordinate scan exists alongside the cheap one.

These fixtures are the shared corpus named in the module docstring: the same
cases are tested against `screen_trajectory.py` in MDRepo's own pipeline,
which carries the same two rules. A submitter told their data is fine here and
then rejected after uploading is worse off than with no tool at all, so the two
must agree.
"""

import os
import sys

import numpy as np
import pytest
from scipy.io import netcdf_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_amber as c  # noqa: E402

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
        var[:] = np.arange(coords.shape[0], dtype=np.float32)

    var = out.createVariable(
        "coordinates", "f", ("frame", "atom", "spatial")
    )
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
def write_prmtop(path, natom):
    """Write enough of an AMBER prmtop for the atom count to be read"""

    with open(path, "wt") as out:
        out.write("%VERSION  VERSION_STAMP = V0001.000\n")
        out.write("%FLAG TITLE\n%FORMAT(20a4)\ntest\n")
        out.write("%FLAG POINTERS\n%FORMAT(10I8)\n")
        out.write(f"{natom:>8}" + f"{0:>8}" * 9 + "\n")

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
def make_dir(tmp_path, coords, lengths, angles, with_time=True):
    """One simulation directory laid out the way a submission is"""

    write_nc(tmp_path / "traj.nc", coords, lengths, angles, with_time)
    write_prmtop(tmp_path / "top.prmtop", ATOMS)
    (tmp_path / "mdrepo-metadata.toml").write_text(
        'trajectory_file_names = ["traj.nc"]\n'
        'topology_file_name = "top.prmtop"\n'
        'structure_file_name = "top.prmtop"\n'
    )
    return str(tmp_path)


# --------------------------------------------------
def problems_of(directory, headers_only=False):
    """Every problem reported for the one trajectory in a directory"""

    findings = c.check_dir(directory, headers_only)
    assert len(findings) == 1
    assert findings[0].error is None, findings[0].error
    return findings[0]


# --------------------------------------------------
def test_a_healthy_directory_is_clean(tmp_path):
    directory = make_dir(tmp_path, *healthy())

    assert problems_of(directory).problems == []


# --------------------------------------------------
def test_the_all_zero_frame_case(tmp_path):
    coords, lengths, angles = healthy()
    coords[2:4] = 0.0
    lengths[2:4] = 0.0
    angles[2:4] = 0.0

    finding = problems_of(make_dir(tmp_path, coords, lengths, angles))

    assert "ZERO FRAMES" in finding.problems
    assert "BAD UNIT CELL" in finding.problems
    assert finding.zero_frames == 2
    assert finding.zero_runs == [(2, 3)]
    assert finding.bad_cell_runs == [(2, 3)]


# --------------------------------------------------
def test_the_uninitialised_memory_case(tmp_path):
    coords, lengths, angles = healthy()
    coords[4, :5] = np.nan
    coords[4, 5:10] = 3.3e38
    lengths[4] = [1.46e233, -1.04e-306, -1.60e-203]
    angles[4] = [1.67e214, -3.00e-79, 1.80e-5]

    finding = problems_of(make_dir(tmp_path, coords, lengths, angles))

    assert "BAD COORDINATES" in finding.problems
    assert "BAD UNIT CELL" in finding.problems
    assert finding.bad_coord_runs == [(4, 4)]
    assert "ZERO FRAMES" not in finding.problems


# --------------------------------------------------
def test_a_valid_box_does_not_excuse_bad_coordinates(tmp_path):
    """The file everyone believed was clean"""

    coords, lengths, angles = healthy()
    coords[3, 7] = np.inf
    directory = make_dir(tmp_path, coords, lengths, angles)

    assert problems_of(directory, headers_only=True).problems == []

    finding = problems_of(directory)
    assert finding.problems == ["BAD COORDINATES"]
    assert finding.bad_coord_frames == 1


# --------------------------------------------------
def test_the_cell_is_checked_even_under_headers_only(tmp_path):
    """
    The cheap check is the one that fires on the submissions we have seen,
    so it must not be the one a hurried run skips.
    """

    coords, lengths, angles = healthy()
    lengths[1] = [1.46e233, -1.04e-306, -1.60e-203]
    directory = make_dir(tmp_path, coords, lengths, angles)

    finding = problems_of(directory, headers_only=True)

    assert finding.problems == ["BAD UNIT CELL"]
    assert finding.bad_cell_runs == [(1, 1)]


# --------------------------------------------------
def test_a_trajectory_with_no_periodic_box_is_not_a_defect(tmp_path):
    coords, lengths, angles = healthy()
    lengths[:] = 0.0
    angles[:] = 0.0

    assert problems_of(make_dir(tmp_path, coords, lengths, angles)).problems == []


# --------------------------------------------------
@pytest.mark.parametrize(
    "lengths_row,angles_row",
    [
        ([0.0, 75.0, 75.0], [109.5, 109.5, 109.5]),
        ([-75.0, 75.0, 75.0], [109.5, 109.5, 109.5]),
        ([np.nan, 75.0, 75.0], [109.5, 109.5, 109.5]),
        ([75.0, 75.0, 75.0], [0.0, 109.5, 109.5]),
        ([75.0, 75.0, 75.0], [180.0, 109.5, 109.5]),
        ([75.0, 75.0, 75.0], [109.5, np.inf, 109.5]),
        ([75.0, 75.0, 75.0], [6.2e116, 6.2e116, 6.2e116]),
    ],
)
def test_each_way_a_cell_can_be_wrong(lengths_row, angles_row):
    """
    The last row is a real frame that a "positive and finite" test lets
    through: 6.2e116 is both positive and finite. The angle gives it away.
    """

    _, lengths, angles = healthy()
    lengths[1] = lengths_row
    angles[1] = angles_row

    assert c.scan_cell(lengths, angles) == [1]


# --------------------------------------------------
def test_an_ordinary_cell_is_left_alone():
    _, lengths, angles = healthy()

    assert c.scan_cell(lengths, angles) == []


# --------------------------------------------------
def test_the_three_coordinate_answers_do_not_overlap():
    coords = np.zeros((5, ATOMS, 3), dtype=np.float32)
    coords[1] = 1.5
    coords[2] = np.nan
    coords[4] = 1.5
    coords[4, 0, 0] = 2.1474836e7

    zero, nonfinite, huge = c.scan_coordinates(coords)

    assert zero == [0, 3]
    assert nonfinite == [2]
    assert huge == [4]
    assert not set(zero) & set(nonfinite) & set(huge)


# --------------------------------------------------
def test_the_xtc_saturation_value_is_caught(tmp_path):
    """
    A NaN written through an XTC comes back as 21,474,836 angstroms

    Finite, non-zero, and just as impossible. A finiteness test alone passes
    it, which would make screening anything that has been through an XTC
    pointless.
    """

    coords, lengths, angles = healthy()
    coords[2, 4] = 2.1474836e7

    finding = problems_of(make_dir(tmp_path, coords, lengths, angles))

    assert finding.problems == ["BAD COORDINATES"]
    assert finding.bad_coord_runs == [(2, 2)]


# --------------------------------------------------
def test_an_ordinary_large_box_is_not_too_large():
    """The limit is 0.1 mm; a big solvated system is a few hundred angstroms"""

    coords = np.full((3, ATOMS, 3), 950.0, dtype=np.float32)

    zero, nonfinite, huge = c.scan_coordinates(coords)

    assert (zero, nonfinite, huge) == ([], [], [])


# --------------------------------------------------
def test_runs_are_grouped():
    assert c.group_runs([]) == []
    assert c.group_runs([4]) == [(4, 4)]
    assert c.group_runs([1, 2, 3, 9, 10, 40]) == [(1, 3), (9, 10), (40, 40)]


# --------------------------------------------------
def test_the_report_names_both_new_defects(tmp_path):
    """
    A submitter acts on the report, not on the console lines, so the report
    has to carry the frame numbers for every defect it found.
    """

    coords, lengths, angles = healthy()
    coords[4, :5] = np.nan
    lengths[1] = 0.0

    findings = c.check_dir(make_dir(tmp_path, coords, lengths, angles), False)
    report = c.build_report(findings, headers_only=False)

    assert "NO USABLE BOX" in report
    assert "NOT\nNUMBERS" in report or "NOT NUMBERS" in report
    assert "frames 1" in report
    assert "frames 4" in report


# --------------------------------------------------
def test_the_existing_checks_still_work(tmp_path):
    """The four that were here first, so this change cannot quietly drop one"""

    coords, lengths, angles = healthy()
    write_nc(tmp_path / "traj.nc", coords, lengths, angles, with_time=False)
    write_prmtop(tmp_path / "top.prmtop", ATOMS + 1)
    (tmp_path / "mdrepo-metadata.toml").write_text(
        'trajectory_file_names = ["traj.nc"]\n'
        'topology_file_name = "top.prmtop"\n'
    )

    finding = problems_of(str(tmp_path))

    assert "ATOM MISMATCH" in finding.problems
    assert "NO TIME AXIS" in finding.problems
