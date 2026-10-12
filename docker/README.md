Generates and builds the `theauto/debian:<release>` Docker images used by
Metron, for ten Debian releases (etch through trixie). The per-release build
contexts are checked in under `generated/<release>/` and can be regenerated.

Main files:

- `debian_release_matrix.py`
  - Loads and validates the shared repository profile.
- `debian_repository_profiles.json`
  - Single source of truth for base images, Debian repositories, components, and pockets.
- `generate_debian_images.py`
  - Generates per-release build contexts under `docker/generated/<release>/`.
- `build_one_image.sh`
  - Builds a single release image.
- `build_all_images.sh`
  - Builds all generated release images.

Image policy:

- `etch` through `bullseye` use `debian/eol:<release>`
- `bookworm` and `trixie` use `debian:<release>`
- Every image probes apt candidates inside the target release image, then installs the first available package for each common build dependency
- `/etc/apt/sources.list` is generated from the same profile as the offline package metadata
- `/etc/theauto-repository-profile.json` and image labels record the profile hash
- No `apt clean` / `apt-get clean`

Regenerating a build context (optional — the checked-in contexts are ready to
build). Resolved package-list manifests under `metadata_manifests/` are
optional; without them pass `--allow-declared-only` to use the declared
repositories/components of the profile:

```bash
python3 generate_debian_images.py --release bookworm --allow-declared-only
```

Typical usage (set the proxy variables only if your network requires one):

```bash
HTTP_PROXY="http://your-proxy:port" \
HTTPS_PROXY="http://your-proxy:port" \
bash build_one_image.sh bookworm
```

Use a suffix while a batch is active, for example `repo-v2`, so validation does
not replace `theauto/debian:bookworm` in place.
