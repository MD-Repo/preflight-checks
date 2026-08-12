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
| `ZERO FRAMES` | Some frames hold `0.0` for every coordinate of every atom — a partially written file. These read as valid NetCDF and survive conversion, then break the analysis stage. | yes |
| `NO TIME AXIS` | The NetCDF has no `time` variable, so frame spacing cannot be read from it. Either rewrite with time information, or declare `sampling_frequency_ps` in `mdrepo-metadata.toml`. | no |

Everything except `ZERO FRAMES` is a header read, which is why `--headers-only`
is fast enough to run over a whole submission in seconds. The zero-frame scan
reads coordinates and is roughly disk-speed.

Nothing is ever modified: the checks only read.

Exit status is 0 when everything is clean and 1 when anything was flagged, so it
can gate a submission script.
