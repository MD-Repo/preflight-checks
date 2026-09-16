# MDRepo Preflight Checks

Check data before submission.

These tools look for problems that MDRepo's pipeline can only detect *after* a
submission has been uploaded and converted — at which point the error it
reports usually names a missing output file rather than the cause. Running them
first turns a slow, confusing failure into a fast, specific one.

## check_amber.py

Checks AMBER simulation directories against their own topology.

### Install

Needs Python 3.11 or newer (for `tomllib`), plus `numpy` and `scipy`.

**With `uv` (easiest — no environment to manage).** `uv` reads the dependencies
declared at the top of the script and fetches them itself the first time you run
it. Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS and Linux
```

then clone and run:

```bash
git clone https://github.com/MD-Repo/preflight-checks
cd preflight-checks
uv run check_amber.py --headers-only path/to/simulations
```

**With a virtual environment**, if you would rather not install `uv`:

```bash
git clone https://github.com/MD-Repo/preflight-checks
cd preflight-checks
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python check_amber.py --headers-only path/to/simulations
```

If you already have numpy and scipy available — through conda, or a system
Python that has them — skip both and run `./check_amber.py` directly.

### Usage

Substitute whichever of the three invocations above you set up; the examples
here use `uv run`.

```bash
# One directory, or a parent holding many
uv run check_amber.py path/to/simulations

# Fast pass: atom counts only, reads no trajectory data
uv run check_amber.py --headers-only path/to/simulations

# Write a plain-text report of the problems found
uv run check_amber.py -r problems.txt path/to/simulations
```

Directories are found by looking for `mdrepo-metadata.toml`, and every
trajectory named in `trajectory_file_names` is checked against the topology in
`topology_file_name`.

### What it finds

| check | what it means | needs trajectory data? |
|---|---|---|
| `ATOM MISMATCH` | The trajectory and the topology declare different atom counts — usually a solvent-stripped trajectory shipped beside the original, unstripped topology. No tool can read the pair. | no |
| `EMPTY TRAJECTORY` | The file is too small to hold even one frame. Usually an interrupted copy or a job that never wrote output. | no |
| `BAD UNIT CELL` | Some frames record a box that is not a box: a length or angle that is not a finite number, a length of zero or less, or an angle outside 0 to 180 degrees. A frame like this can stall the conversion step for hours rather than fail it. | no |
| `ZERO FRAMES` | Some frames hold `0.0` for every coordinate of every atom — a partially written file. These read as valid NetCDF and survive conversion, then break the analysis stage. | yes |
| `BAD COORDINATES` | Some frames hold coordinates that are not positions — NaN, infinity, or 1e6 Å and up. This is what uninitialised memory looks like written to a file as if it were data. These survive conversion, and the box on such a frame is often perfectly ordinary, so nothing cheaper finds them. The size test matters because XTC cannot store a NaN: written through one, the same damage reads back as a finite 21,474,836 Å. | yes |
| `NO TIME AXIS` | The NetCDF has no `time` variable, so frame spacing cannot be read from it. Either rewrite with time information, or declare `sampling_frequency_ps` in `mdrepo-metadata.toml`. | no |

`--headers-only` skips the two checks that read coordinates. It still checks
the unit cell, because the cell arrays are three numbers per frame against tens
of thousands of coordinates — cheap, though not as cheap as a true header read,
since NetCDF keeps each frame's records together and reading them walks the
whole file.

Run without `--headers-only` when you can. `BAD UNIT CELL` and `BAD
COORDINATES` overlap on most damaged frames but neither covers the other: a
frame can hold coordinates that are not numbers under a box identical to its
neighbours', and only the coordinate scan will see it.

Nothing is ever modified: the checks only read.

Exit status is 0 when everything is clean and 1 when anything was flagged, so it
can gate a submission script.

### Tests

The checks are covered by a corpus of synthetic trajectories built from the
defects that have actually arrived. MDRepo's own pipeline carries the same two
frame checks in `screen_trajectory.py` and is tested against the same cases, so
that this tool and the pipeline cannot drift apart and disagree about whether a
submission is sound.

```bash
uv run --with pytest pytest tests/
```
