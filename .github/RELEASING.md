# Releasing tts-with-rvc

Current release candidate: **0.1.10**, branch **releases**.

## Build and check

Use Python 3.12 for packaging. From the repository root:

```bash
python -m pip install 'build>=1.2,<2' 'twine>=6,<7' 'packaging>=24,<27'
python .github/scripts/package.py check
python -m build
python -m twine check --strict dist/*
python .github/scripts/package.py verify
python .github/scripts/package.py smoke
```

Use a clean `dist` directory containing exactly one wheel and one sdist for
this version. `python -m build` creates the wheel from the sdist. The smoke
check installs the wheel without runtime dependencies in a temporary venv.

CI also runs unit tests for Python 3.10, 3.11 and 3.12 on Windows and Linux.
Runtime inference with downloaded voice models and multiple physical GPUs
requires separate application testing.

## Version updates

```bash
python .github/scripts/package.py version --bump
```

This increases the last numeric component and updates both `pyproject.toml`
and `tts_with_rvc/__init__.py`. To choose a specific version:

```bash
python .github/scripts/package.py version --set 0.2.0
```

Update the README heading and release notes in the same commit. CI rejects
mismatched versions, an existing PyPI version, or a version below the latest
stable release. Version changes are explicit local commits.

## Configure publishing

GitHub repository: **Atm4x/tts-with-rvc**. Create the **pypi-rvc** Environment and
restrict deployment branches to **releases**.

Choose one authentication method:

- **API token:** set secret `PYPI_API_TOKEN` in the Environment. An account-wide
  token with permission for all packages may instead be set as a repository
  secret; a project-scoped token must be placed in its package's Environment.
- **Trusted Publishing:** leave `PYPI_API_TOKEN` unset and register this publisher
  in PyPI project settings: owner `Atm4x`, repository `tts-with-rvc`,
  workflow `release.yml`, Environment `pypi-rvc`.

For the shared TTS repository, use `pypi-rvc` for the PyTorch package and
`pypi-onnx` for ONNX if they have different project-scoped tokens.
TestPyPI tokens and 2FA recovery codes cannot authorize production PyPI uploads.

After authentication is configured, add the repository variable
`PYPI_PUBLISH_ENABLED` with value `true` in **Settings → Secrets and variables →
Actions → Variables**. Until then, CI builds the package and skips publishing
and GitHub Release creation. The shared TTS repository variable enables the
publish job on both `releases` and `releases-onnx`.

## Publish

Commit the README and synchronized version files, then push **releases**.
The workflow tests, builds, validates, publishes to PyPI and creates a GitHub
Release with tag `tts-with-rvc/v<version>` and the same wheel/sdist artifacts.

If authentication was configured after the candidate commit was already pushed,
rerun all jobs of that workflow run, or trigger another push after checking that
the version has not been uploaded yet.

If PyPI publishing succeeded but GitHub Release creation failed, rerun only the
failed job. A full rerun is rejected because that PyPI version now exists.
If only one archive was uploaded, inspect the PyPI release before retrying;
already-uploaded files cannot be overwritten.

See the [PyPI Trusted Publisher guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [PyPI API token help](https://pypi.org/help/#apitoken).
