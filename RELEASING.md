# Releasing

This project publishes to PyPI from GitHub Actions using PyPI Trusted
Publishing. The release workflow builds distributions on every manual run and
publishes only when a GitHub Release is published.

## One-time setup

1. Configure a PyPI Trusted Publisher for the existing `densratio` project:
   - Owner: `hoxo-m`
   - Repository: `densratio_py`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. Configure the GitHub environment named `pypi` in repository settings.
   Required reviewers are recommended for this environment.
3. Do not add a long-lived PyPI API token to GitHub secrets. Trusted Publishing
   uses GitHub OIDC and short-lived PyPI credentials during the release job.

## Release checklist

1. Update package metadata and docs:
   - Set the next version in `setup.py`.
   - Move release notes from `CHANGELOG.md`'s `Unreleased` section into a
     dated version section.
   - Update README examples if the public API changed.
2. Run local release checks:

   ```shell
   python -m pip install -e .[dev]
   python -m pytest
   python -m build
   python -m twine check dist/*
   ```

3. Open and merge a release-prep PR into `main`.
4. Confirm the GitHub Actions CI workflow is green on `main`.
5. Create a GitHub Release:
   - Tag: `vX.Y.Z`, matching the package version `X.Y.Z`.
   - Target: `main`.
   - Title: `densratio X.Y.Z`.
   - Body: the matching section from `CHANGELOG.md`.
6. Publish the GitHub Release. The `Release` workflow will build the package
   and publish it to PyPI after the `pypi` environment is approved. The
   workflow rejects releases whose `vX.Y.Z` tag does not match the package
   version.
7. Verify the published package:

   ```shell
   python -m pip install --upgrade densratio==X.Y.Z
   python -c "from densratio import densratio, uLSIF, RuLSIF, KLIEP; print('ok')"
   ```

## Manual dry run

Use the `workflow_dispatch` trigger on the `Release` workflow to build and
check distributions without publishing to PyPI.
