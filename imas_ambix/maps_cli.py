"""Publish and fetch machine map bundles through GHCR.

A machine's map bundle is the directory ``maps/<machine>/``, published as a
single OCI artifact of type :data:`ARTIFACT_TYPE`.  Each machine has its own
registry package, so each carries its own release line: a tag holds a version
alone and never a machine name.

The version state is read from the registry's tag list, not from git.  The map
directories are gitignored, so a git tag would carry no map content, and both
packages take their own version from git tags, so a map tag in git would become
a package version.  One function lists the package's tags and every version
query reads that one list.

Versions are semantic versions with release candidates.  The package starts
from a ``v0.0.0`` baseline, so a package with no semver tag -- a new one, or an
existing one carrying only legacy date tags -- can be released and abandoned
without a tag to bump from.

``release`` also merges the installer's private ``oras pull`` into ``pull``, so
the registry, the authentication form and the artifact type have one owner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import click

from imas_ambix.data.paths import GHCR_OWNER, MAPS_DIR, package_for_machine

REGISTRY = f"ghcr.io/{GHCR_OWNER}"
ARTIFACT_TYPE = "application/vnd.imas-alambic.bundle.v1"
BUNDLE_NAME = "bundle.json"
BASELINE_VERSION = "v0.0.0"
EXCLUDED_TOP_LEVEL = ("source", "superseded")

ANNOTATION_MACHINE = "io.imas-ambix.machine"
ANNOTATION_COMMIT = "io.imas-ambix.git-commit"
ANNOTATION_TREE_DIGEST = "io.imas-ambix.tree-digest"
ANNOTATION_VERSION = "org.opencontainers.image.version"
ANNOTATION_DESCRIPTION = "org.opencontainers.image.description"

_SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-rc(\d+))?$")


def _package(machine: str) -> tuple[str, str]:
    """The registry and package name carrying ``machine``'s bundle."""
    try:
        return REGISTRY, package_for_machine(machine)
    except KeyError:
        raise click.ClickException(
            f"No registry package is published for machine '{machine}'."
        ) from None


def require_oras() -> None:
    """Refuse when the ``oras`` client is not on ``PATH``."""
    if not shutil.which("oras"):
        raise click.ClickException(
            "oras not found in PATH. Install from: "
            "https://github.com/oras-project/oras/releases"
        )


def ghcr_token() -> str | None:
    """Return the token to authenticate GHCR with, or None for cached creds."""
    for variable in ("GHCR_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(variable)
        if value:
            return value
    return None


def login_to_ghcr(token: str | None) -> None:
    """Log ``oras`` in with ``token``; a gh CLI token is passed as user ``token``."""
    if not token:
        return
    result = subprocess.run(
        ["oras", "login", "ghcr.io", "-u", "token", "--password-stdin"],
        input=token,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"GHCR login failed: {result.stderr}")


def list_registry_tags(registry: str, token: str | None, pkg_name: str) -> list[str]:
    """List every tag the package carries.

    A package the registry reports as not found has no tags rather than raising,
    which is the state a never-released machine is in, so its release line
    starts from the baseline.
    """
    login_to_ghcr(token)
    result = subprocess.run(
        ["oras", "repo", "tags", f"{registry}/{pkg_name}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "not found" in result.stderr.lower():
            return []
        raise click.ClickException(f"Failed to list tags: {result.stderr}")
    return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]


def fetch_tag_annotations(
    registry: str, tags: list[str], pkg_name: str
) -> dict[str, dict[str, str]]:
    """Return each tag's whole manifest annotation map.

    The whole map is returned rather than only the description, because both
    the unchanged-tree skip and ``status`` read the released tree's digest from
    it.  A tag whose manifest cannot be read yields an empty map.
    """
    annotations: dict[str, dict[str, str]] = {}
    for tag in tags:
        result = subprocess.run(
            ["oras", "manifest", "fetch", f"{registry}/{pkg_name}:{tag}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            annotations[tag] = {}
            continue
        try:
            annotations[tag] = json.loads(result.stdout).get("annotations", {})
        except ValueError:
            annotations[tag] = {}
    return annotations


def parse_version(tag: str) -> tuple[int, int, int, int | None]:
    """Parse ``vM.m.p`` and ``vM.m.p-rcN`` into its four components."""
    match = _SEMVER_RE.match(tag)
    if not match:
        raise click.ClickException(f"Cannot parse version: {tag}")
    rc = int(match[4]) if match[4] else None
    return int(match[1]), int(match[2]), int(match[3]), rc


def format_tag(major: str | int, minor: int, patch: int, rc: int | None) -> str:
    """Format version components as a tag (``v1.0.0`` or ``v1.0.0-rc1``)."""
    base = f"v{major}.{minor}.{patch}"
    return f"{base}-rc{rc}" if rc is not None else base


def version_key(tag: str) -> tuple[int, int, int, int, int]:
    """Sort key ordering semver tags newest first, a candidate below its release.

    The fourth component ranks a final release (1) above a candidate (0) for
    the same base version, and the fifth ranks the candidate number, so
    ``v0.2.0`` sorts above ``v0.2.0-rc2`` above ``v0.2.0-rc1``.
    """
    major, minor, patch, rc = parse_version(tag)
    return (major, minor, patch, 1 if rc is None else 0, rc or 0)


def semver_tags(raw_tags: list[str]) -> list[str]:
    """Filter to ``v``-prefixed semver tags, newest first.

    The registry returns tags in its own order, so without this sort "the
    latest tag" is undefined and a version could step backwards or collide.
    Legacy tags such as a date are discarded, never read as a version.
    """
    kept = [tag for tag in raw_tags if _SEMVER_RE.match(tag)]
    return sorted(kept, key=version_key, reverse=True)


def latest_tag(tags: list[str]) -> str:
    """The newest semver tag, or the ``v0.0.0`` baseline when there is none."""
    return tags[0] if tags else BASELINE_VERSION


def latest_stable_tag(tags: list[str]) -> str:
    """The newest non-candidate tag, or the ``v0.0.0`` baseline when there is none."""
    for tag in tags:
        if "-rc" not in tag:
            return tag
    return BASELINE_VERSION


def tag_exists(tags: list[str], tag: str) -> bool:
    """Whether the package already carries ``tag``."""
    return tag in tags


def apply_bump(major: int, minor: int, patch: int, bump: str) -> tuple[int, int, int]:
    """Apply a major, minor or patch bump to base version components."""
    if bump == "major":
        return major + 1, 0, 0
    if bump == "minor":
        return major, minor + 1, 0
    if bump == "patch":
        return major, minor, patch + 1
    raise click.ClickException(f"Invalid bump type: {bump}")


def compute_next_version(bump: str | None, *, final: bool, tags: list[str]) -> str:
    """Compute the next tag from the package's semver tag list.

    Transitions:

      stable + --bump             new candidate series (v0.1.0 + minor -> v0.2.0-rc1)
      stable + --bump + --final   direct release (v0.1.0 + patch -> v0.1.1)
      stable + no bump            patch candidate (v0.1.0 -> v0.1.1-rc1), so an
                                  automatic call -- which carries no flags --
                                  never fails on state
      candidate + no bump         increment the candidate (rc1 -> rc2)
      candidate + --final         finalize (v0.2.0-rc1 -> v0.2.0)
      candidate + --bump          abandon the candidate, bump from the latest
                                  stable (baseline when there is none)

    An RC-collision or a candidate the package already carries advances to the
    next free number, and ``--final`` never carries a candidate suffix.
    """
    latest = latest_tag(tags)
    major, minor, patch, current_rc = parse_version(latest)

    if current_rc is None:
        if bump is None and final:
            raise click.ClickException(
                f"Not in candidate mode (latest: {latest}). "
                "Use --bump to start a release candidate series."
            )
        new_major, new_minor, new_patch = apply_bump(
            major, minor, patch, bump or "patch"
        )
        return _first_free_tag(tags, new_major, new_minor, new_patch, final)

    if bump is None and final:
        return format_tag(major, minor, patch, None)

    if bump is None:
        return _first_free_tag(
            tags, major, minor, patch, final=False, start=current_rc + 1
        )

    s_major, s_minor, s_patch, _ = parse_version(latest_stable_tag(tags))
    new_major, new_minor, new_patch = apply_bump(s_major, s_minor, s_patch, bump)
    return _first_free_tag(tags, new_major, new_minor, new_patch, final)


def _first_free_tag(
    tags: list[str],
    major: int,
    minor: int,
    patch: int,
    final: bool,
    start: int = 1,
) -> str:
    """The candidate number that does not collide with an existing tag."""
    rc: int | None = None if final else start
    tag = format_tag(major, minor, patch, rc)
    if rc is not None:
        while tag_exists(tags, tag):
            rc += 1
            tag = format_tag(major, minor, patch, rc)
    return tag


def released_paths(root: Path) -> list[Path]:
    """Files the published tree holds, relative to ``root`` and sorted.

    The development inputs under ``source/`` and ``superseded/`` are left out,
    so a facility install receives only the converted store and its descriptor.
    """
    rels: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        rels.append(rel)
    return rels


def _bundle_without_version(path: Path) -> bytes:
    """The bundle descriptor with its ``version`` field removed, canonicalised."""
    data = json.loads(path.read_text())
    data.pop("version", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def tree_digest(root: Path) -> str:
    """A sha256 over the published files' paths and contents.

    The bundle's ``version`` field is left out, so writing a released version
    into it does not change the digest and an unchanged tree stays recognisable
    against the latest release.
    """
    digest = hashlib.sha256()
    for rel in released_paths(root):
        digest.update(rel.as_posix().encode())
        digest.update(b"\0")
        blob = (
            _bundle_without_version(root / rel)
            if rel.as_posix() == BUNDLE_NAME
            else (root / rel).read_bytes()
        )
        digest.update(hashlib.sha256(blob).digest())
    return digest.hexdigest()


def stage_bundle(root: Path, dest: Path, version: str) -> None:
    """Copy the published files into ``dest``, writing ``version`` into the bundle."""
    for rel in released_paths(root):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    _write_bundle_version(dest, version)


def _write_bundle_version(root: Path, version: str) -> None:
    """Write the released version into a bundle descriptor under ``root``."""
    bundle = root / BUNDLE_NAME
    data = json.loads(bundle.read_text())
    data["version"] = version
    bundle.write_text(json.dumps(data, indent=2) + "\n")


def write_released_version(machine_dir: Path, version: str) -> None:
    """Write the released version into the local bundle descriptor."""
    _write_bundle_version(machine_dir, version)


def push_tree(
    ref: str, staging: Path, annotations: dict[str, str], token: str | None
) -> None:
    """Push a staged tree to ``ref`` as one artifact, one annotation per key."""
    login_to_ghcr(token)
    command = [
        "oras",
        "push",
        ref,
        f"./:{ARTIFACT_TYPE}",
        "--artifact-type",
        ARTIFACT_TYPE,
    ]
    for key, value in annotations.items():
        command.extend(["--annotation", f"{key}={value}"])
    result = subprocess.run(command, cwd=staging, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"oras push failed: {result.stderr}")


def pull_tree(ref: str, dest: Path, token: str | None) -> None:
    """Fetch the artifact at ``ref`` into ``dest``, creating it if needed."""
    login_to_ghcr(token)
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["oras", "pull", ref, "-o", str(dest)], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise click.ClickException(f"oras pull failed: {result.stderr}")




def git_commit() -> str:
    """The current HEAD commit, or an empty string when unresolvable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _ref(registry: str, pkg_name: str, tag: str) -> str:
    return f"{registry}/{pkg_name}:{tag}"


def _format_replaced(root: Path) -> list[str]:
    """The files a pull into ``root`` would replace, as relative paths."""
    return [rel.as_posix() for rel in released_paths(root)] if root.exists() else []


@click.group(name="maps")
def maps() -> None:
    """Publish and fetch machine map bundles through GHCR."""


@maps.command()
@click.argument("machine")
@click.option(
    "--bump",
    type=click.Choice(["major", "minor", "patch"]),
    default=None,
    help=(
        "Start a new candidate series. Omitted on a stable release, "
        "a patch candidate is started."
    ),
)
@click.option(
    "--final",
    is_flag=True,
    help="Promote the current candidate to a stable release, or skip the candidate.",
)
@click.option(
    "-m", "--message", default=None, help="Release message (push description)."
)
def release(machine: str, bump: str | None, final: bool, message: str | None) -> None:
    """Release ``maps/<machine>/`` as the next version of its registry package.

    The version is computed from the package's tags; the tree is pushed only
    when its digest differs from the latest release, so an automatic call after
    a landing that changed nothing costs no push.  No git tag is created: both
    packages take their version from git tags, so a map tag in git would become
    a package version.
    """
    require_oras()
    registry, pkg_name = _package(machine)
    token = ghcr_token()
    machine_dir = MAPS_DIR / machine
    if not machine_dir.is_dir():
        raise click.ClickException(f"No map directory at {machine_dir}.")

    tags = semver_tags(list_registry_tags(registry, token, pkg_name))
    latest = latest_tag(tags)
    latest_annotations = (
        fetch_tag_annotations(registry, [latest], pkg_name).get(latest, {})
        if latest != BASELINE_VERSION
        else {}
    )
    digest = tree_digest(machine_dir)
    if digest == latest_annotations.get(ANNOTATION_TREE_DIGEST):
        click.echo(
            f"{machine}: nothing to release -- the tree digest matches {latest}."
        )
        return

    version = compute_next_version(bump, final=final, tags=tags)
    write_released_version(machine_dir, version)

    annotations = {
        ANNOTATION_VERSION: version,
        ANNOTATION_MACHINE: machine,
        ANNOTATION_COMMIT: git_commit(),
        ANNOTATION_TREE_DIGEST: digest,
    }
    if message:
        annotations[ANNOTATION_DESCRIPTION] = message

    with tempfile.TemporaryDirectory(prefix="ambix-maps-release-") as staging_dir:
        staging = Path(staging_dir)
        stage_bundle(machine_dir, staging, version)
        push_tree(_ref(registry, pkg_name, version), staging, annotations, token)
    click.echo(f"{machine}: released {version} ({registry}/{pkg_name}).")
    click.echo(f"  tree digest: {digest}")


@maps.command()
@click.argument("machine")
@click.option(
    "--version", "version", default=None, help="Tag to fetch (default: latest stable)."
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to fetch into (default: maps/<machine>/).",
)
@click.option(
    "--force", is_flag=True, help="Overwrite a local tree holding unreleased edits."
)
def pull(machine: str, version: str | None, dest: Path | None, force: bool) -> None:
    """Fetch a released bundle into ``maps/<machine>/`` or a given directory.

    A local tree whose digest matches no released tag holds map edits that
    exist nowhere else, so it is refused unless ``--force`` is given; the files
    a forced pull replaces are printed first.
    """
    require_oras()
    registry, pkg_name = _package(machine)
    token = ghcr_token()
    tags = semver_tags(list_registry_tags(registry, token, pkg_name))

    wanted = version or latest_stable_tag(tags)
    if wanted == BASELINE_VERSION:
        raise click.ClickException(
            f"{registry}/{pkg_name} carries no released tag to pull."
        )
    if not tag_exists(tags, wanted):
        raise click.ClickException(
            f"{registry}/{pkg_name} carries no tag {wanted}."
        )

    dest = Path(dest) if dest is not None else MAPS_DIR / machine
    if dest.exists() and any(dest.iterdir()):
        released = {
            annotations.get(ANNOTATION_TREE_DIGEST)
            for annotations in fetch_tag_annotations(registry, tags, pkg_name).values()
        }
        local = tree_digest(dest)
        if force:
            click.echo(f"replacing {len(_format_replaced(dest))} file(s) under {dest}:")
            for rel in _format_replaced(dest):
                click.echo(f"  {rel}")
        elif local not in released:
            raise click.ClickException(
                f"{dest} matches no released tag and holds unreleased map edits. "
                "Re-run with --force to overwrite."
            )

    pull_tree(_ref(registry, pkg_name, wanted), dest, token)
    click.echo(f"{machine}: fetched {wanted} into {dest}.")


@maps.command()
@click.argument("machine")
def status(machine: str) -> None:
    """Show the release state of each machine: its tags and its local tree."""
    registry, pkg_name = _package(machine)
    token = ghcr_token()
    tags = semver_tags(list_registry_tags(registry, token, pkg_name))
    latest = latest_tag(tags)

    if latest == BASELINE_VERSION:
        click.echo(f"{machine} ({registry}/{pkg_name})")
        click.echo("  Release state: none (no semver tag)")
        return

    annotations = fetch_tag_annotations(registry, tags, pkg_name)
    is_candidate = "-rc" in latest
    candidates = [t for t in tags if "-rc" in t]
    stable = latest_stable_tag(tags)
    click.echo(f"{machine} ({registry}/{pkg_name})")
    click.echo(f"  Release state:   {'candidate' if is_candidate else 'stable'}")
    click.echo(f"  Latest stable:   {stable if stable != BASELINE_VERSION else 'none'}")
    click.echo(f"  Latest candidate: {candidates[0] if candidates else 'none'}")
    click.echo(f"  Latest tag:      {latest}")

    machine_dir = MAPS_DIR / machine
    if not machine_dir.is_dir():
        click.echo(f"  Local tree:      absent ({machine_dir})")
        return
    local = tree_digest(machine_dir)
    matching = [
        tag
        for tag in tags
        if annotations.get(tag, {}).get(ANNOTATION_TREE_DIGEST) == local
    ]
    if matching:
        click.echo(f"  Local tree:      matches {matching[0]}")
    else:
        click.echo("  Local tree:      matches no released tag")
