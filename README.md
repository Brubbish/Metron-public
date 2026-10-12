# Metron

Automated build repair for historical C/C++ projects. Given a source tree and a Debian release, Metron runs the build in Docker, diagnoses failures, and applies environment-level repairs until the build succeeds.

## Prerequisites

- Python 3.11+
- Docker installed and the daemon running (`docker info` works)
- An LLM API key

### Docker images

Metron runs each build in a container from an image named `theauto/debian:<codename>`, where `<codename>` is the Debian release matching the project's era (e.g. `stretch`, `buster`, `bullseye`, `bookworm`, `trixie`). These images are not published on a registry — build them locally with the scripts in [`docker/`](docker/README.md) (recommended: they install a common build-dependency set and pin the Debian repositories used in the experiments):

```bash
bash docker/build_all_images.sh        # or: bash docker/build_one_image.sh bookworm
```

As a minimal fallback you can tag the official Debian images instead; they work but lack the pinned repository profile:

```bash
for tag in stretch buster bullseye bookworm trixie; do
  docker pull "debian:$tag"
  docker tag "debian:$tag" "theauto/debian:$tag"
done
```

The project directory is mounted into the container at `/work`.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your LLM API key
```

## Usage

### Single project

```bash
python -m core.agent_build_main <folderpath> <sysuse> [--max-turns N]
```

- `<folderpath>`: local path to the checked-out project
- `<sysuse>`: Debian codename for the build environment (e.g. `bookworm`)

Example:

```bash
python -m core.agent_build_main /path/to/helloworld bookworm
```

Exit codes: `0` success, `1` build failure / no plan, `2` timeout, `5` provider error, `6` no command, `7` network failure.

### Batch

```bash
python -m batch.run_agent_compile_batch \
  --input-json <records.json> \
  --output-root <output-dir> \
  [--jobs N] [--max-turns N] [--sysuse <codename>]
```

The input JSON is a list of records with `owner`/`repo`/`commit` fields. Results are appended to `<output-dir>/records.jsonl`.

## License

MIT.
