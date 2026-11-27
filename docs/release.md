# Release checklist

1. Review the scientific matrix, known limitations, independent tests, synthetic figures and reported failures.
2. Verify separate statement and branch thresholds, Ruff lint and format, Mypy and external-package tests on the proposed commit.
3. Build wheel and source distributions, run Twine strict checks and install each from outside the checkout.
4. Obtain explicit project-owner approval for promotion. Register the PyPI trusted publisher for owner Phasetensor, repository SurfaceCheck, workflow publish.yml, environment pypi. Use an environment reviewer for publication.
5. After approval, build once from the tagged commit and publish the same files to GitHub and PyPI. Compare SHA-256 per filename with both remote services and install the exact public version in a clean environment.

This development branch has no release tag or public distribution. A 404 for the normalized PyPI name only establishes the absence of a currently visible project.
