"""Regression tests.

These tests differ from unit and integration tests in three ways:

1. **Golden-file style**: many of them compare current behaviour against a
   recorded, hand-verified baseline (top-k IDs, IDF values, file format).
   If a change is intentional, update the golden value in the same commit.

2. **Known-bug markers**: bugs documented in the technical report but not
   yet fixed are encoded with ``pytest.mark.xfail(strict=True)``. The day
   somebody fixes the bug, the test starts XPASSing and fails the build,
   forcing the xfail to be removed at the same time as the fix.

3. **Cross-version compatibility**: artefact-format tests guard against
   silent serialisation changes that would break users with an existing
   ``data/indexer/`` directory.
"""
