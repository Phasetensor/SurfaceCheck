# SurfaceCheck

SurfaceCheck helps options researchers audit European call quotes for one expiry. Given increasing strikes, bid and ask prices, and a discount factor, it checks whether the spreads admit nonnegative, decreasing, convex prices with vertical-spread slopes bounded below by the negative discount factor. It checks strike shape only. Forward-dependent price bounds and calendar spreads are outside this release.

```python
from surfacecheck import check_slice

result = check_slice([90, 100, 110], [11, 7, 3], [13, 11, 5])
assert result["status"] == "spreads_feasible"
```

The result includes midpoint violations by strike and a feasible price vector when one exists. Source installation: `python -m pip install .`.
