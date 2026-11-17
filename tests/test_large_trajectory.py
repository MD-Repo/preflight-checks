"""--sampled-frames on a real, large trajectory

Skipped unless both of these name files on the machine running the tests:

    PREFLIGHT_LARGE_TRAJECTORY=/path/to/big.xtc
    PREFLIGHT_LARGE_TOPOLOGY=/path/to/its.tpr
    uv run --no-project --with-requirements requirements.txt --with pytest \
        pytest -s tests/test_large_trajectory.py

No large file is kept in the repository, and CI sets neither variable. The
files are linked into a temporary directory and checked there, so the
offsets index MDAnalysis saves beside an XTC or TRR lands in that directory
and not beside the original. Expect it to take as long as one full read of
the file: about 4 minutes for a 13.7 GB XTC.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_simulations as c  # noqa: E402

TRAJECTORY = os.environ.get("PREFLIGHT_LARGE_TRAJECTORY")
TOPOLOGY = os.environ.get("PREFLIGHT_LARGE_TOPOLOGY")

pytestmark = pytest.mark.skipif(
    not (TRAJECTORY and TOPOLOGY),
    reason="set PREFLIGHT_LARGE_TRAJECTORY and PREFLIGHT_LARGE_TOPOLOGY",
)


# --------------------------------------------------
def evict(path):
    """Drop a file from the page cache, so what is read comes off the disk"""

    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(os.path.realpath(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# --------------------------------------------------
def disk_reads():
    """Bytes this process has read from the disk, or None off Linux"""

    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


# --------------------------------------------------
def timed(directory, sampled):
    reads, start = disk_reads(), time.perf_counter()
    findings = c.check_dir(directory, False, sampled)
    seconds = time.perf_counter() - start
    read = None if reads is None else disk_reads() - reads

    assert len(findings) == 1
    assert findings[0].error is None, findings[0].error
    return findings[0], seconds, read


# --------------------------------------------------
def test_sampled_frames_on_a_large_trajectory(tmp_path):
    traj = os.path.basename(TRAJECTORY)
    top = os.path.basename(TOPOLOGY)
    os.symlink(os.path.abspath(TRAJECTORY), tmp_path / traj)
    os.symlink(os.path.abspath(TOPOLOGY), tmp_path / top)
    (tmp_path / "mdrepo-metadata.toml").write_text(
        f'trajectory_file_names = ["{traj}"]\ntopology_file_name = "{top}"\n'
    )
    size = os.path.getsize(TRAJECTORY)

    evict(TRAJECTORY)
    sampled, sampled_s, sampled_read = timed(str(tmp_path), True)
    evict(TRAJECTORY)
    full, full_s, _ = timed(str(tmp_path), False)

    print(
        f"\n{traj}: {size / 1e9:.2f} GB, {full.frames} frames, "
        f"{full.traj_atoms} atoms"
        f"\n  sampled: {sampled_s:7.1f} s"
        + ("" if sampled_read is None else f", {sampled_read / 1e9:.2f} GB read")
        + f"\n  full:    {full_s:7.1f} s"
        f"\n  bonds:   {sampled.bond_frames}"
        + (f" (skipped: {sampled.bond_skipped})" if sampled.bond_skipped else "")
    )

    # The same verdict on what --sampled-frames is for.
    watched = {"ATOM MISMATCH", "EMPTY TRAJECTORY", "TOPOLOGY MISMATCH"}
    assert sampled.frames == full.frames
    assert sampled.traj_atoms == full.traj_atoms
    assert watched & set(sampled.problems) == watched & set(full.problems)
    assert sampled.bond_frames == full.bond_frames
    assert sampled.bond_skipped == full.bond_skipped

    # Without reading the file: finding the frames walks their headers, and
    # nothing between the three is decompressed.
    if sampled_read is not None:
        assert sampled_read < 0.1 * size
    assert sampled_s < 0.5 * full_s
