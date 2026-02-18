# SubScript Function Reference

Complete documentation of all functions in the SubScript library (v1.0.10), organized by module.

**Last Updated:** 2026-02-18
**Repository:** https://github.com/cgannonucm/SubScript

---

## Table of Contents

1. [Core Data Structures](#core-data-structures) - `tabulatehdf5.py`
2. [Node Extraction](#node-extraction) - `scripts/nodes.py`
3. [Node Filters](#node-filters) - `scripts/nfilters.py`
4. [Spatial Projections](#spatial-projections) - `scripts/spatial.py`
5. [Histograms & Distributions](#histograms--distributions) - `scripts/histograms.py`
6. [Decorators & Wrappers](#decorators--wrappers) - `wrappers.py`
7. [Time-Series Tracking](#time-series-tracking) - `tracking.py`
8. [Batch Processing](#batch-processing) - `macros.py`
9. [External Data Integration](#external-data-integration) - `external.py`
10. [Utilities](#utilities) - `util.py`

---

## Core Data Structures

### class NodeProperties

**Module:** `subscript.tabulatehdf5`

Dictionary-like wrapper for Galacticus node data with filtering and lazy-loading support.

**Attributes:**
- `data` (dict/UserDict): Underlying data dictionary
- `_nodefilter` (np.ndarray or None): Boolean mask for filtering
- `_startn` (int): Start index for slicing
- `_stopn` (int or None): Stop index for slicing

**Key Methods:**

#### `filter(nodefilter: np.ndarray[bool]) → NodeProperties`
Apply a boolean filter to select subset of nodes.

```python
filtered = node_props.filter(np.array([True, False, True, ...]))
```

#### `unfilter() → NodeProperties`
Recursively remove all filters to access raw data.

```python
unfiltered = filtered.unfilter()  # Returns to original data
```

#### `get_filter() → np.ndarray[bool]`
Retrieve the current active filter (or all-True if none applied).

```python
current_filter = node_props.get_filter()
```

**Notes:**
- Lazy loads from h5py.Dataset on first access
- Caching controlled by `subscript.defaults.Meta.cache`
- Supports slicing with `_startn` and `_stopn` for tree separation

---

### get_galacticus_outputs()

**Module:** `subscript.tabulatehdf5`

**Signature:**
```python
def get_galacticus_outputs(galout: h5py.File) → np.ndarray[int]
```

**Description:**
Extract sorted output snapshot indices from a Galacticus HDF5 file.

**Parameters:**
- `galout` (h5py.File): Open Galacticus HDF5 file

**Returns:**
- np.ndarray[int]: Sorted array of output indices (e.g., [0, 1, 2, 3, ...])

**Example:**
```python
import h5py
from subscript.tabulatehdf5 import get_galacticus_outputs

gout = h5py.File('galacticus.hdf5')
outputs = get_galacticus_outputs(gout)
print(outputs)  # [0 1 2 3 4]
gout.close()
```

**Notes:**
- Reads from `Outputs` group in HDF5 file
- Extracts integer from key names like `Output0`, `Output1`
- Returns sorted indices

---

### get_custom_dsets()

**Module:** `subscript.tabulatehdf5`

**Signature:**
```python
def get_custom_dsets(goutn: h5py.Group) → dict[str, Callable]
```

**Description:**
Generate example custom datasets from a Galacticus output group. Custom datasets are on-the-fly computed properties.

**Parameters:**
- `goutn` (h5py.Group): HDF5 group for a specific output snapshot

**Returns:**
- dict[str, Callable]: Dictionary mapping custom dataset names to lambda functions

**Included Custom Datasets:**

1. **`custom_node_tree`**: Which tree each node belongs to
   ```python
   tree_indices = np.concatenate([np.full(count, index)
                                  for count, index in zip(counts, treenums)])
   ```

2. **`custom_node_tree_outputorder`**: Position in output order per node
   ```python
   output_order = np.concatenate([np.full(count, i)
                                  for i, count in enumerate(counts)])
   ```

3. **`custom_id`**: Unique integer ID for each node (0, 1, 2, ...)
   ```python
   custom_id = np.arange(nodecount)
   ```

**Example:**
```python
import h5py
from subscript.tabulatehdf5 import get_custom_dsets, tabulate_trees

gout = h5py.File('galacticus.hdf5')
trees = tabulate_trees(gout, custom_dsets=get_custom_dsets)
custom_dsets = get_custom_dsets(gout['Outputs']['Output0']['nodeData'])

# Use with nodedata
node_ids = nodedata(trees[0], key='custom_id')
```

**Notes:**
- Useful for creating node identifiers that span multiple trees
- Returns callables (lambdas) for lazy evaluation
- Can be extended with custom key_map for `symphony_to_galacticus_like_dict()`

---

### tabulate_trees()

**Module:** `subscript.tabulatehdf5`

**Signature:**
```python
def tabulate_trees(gout: h5py.File, out_index: int = -1,
                   custom_dsets: Callable = None, **kwargs) → list[NodeProperties]
```

**Description:**
Load and tabulate node properties for all merger trees in a Galacticus output snapshot. Automatically separates nodes into per-tree NodeProperties objects.

**Parameters:**
- `gout` (h5py.File): Open Galacticus HDF5 file
- `out_index` (int, default=-1): Snapshot index to load
  - `-1`: Latest snapshot (default)
  - `0, 1, 2, ...`: Specific snapshot
- `custom_dsets` (Callable, optional): Function returning custom dataset dict
- `**kwargs`: Additional arguments (currently unused)

**Returns:**
- list[NodeProperties]: List of NodeProperties, one per merger tree

**Example:**
```python
import h5py
from subscript.tabulatehdf5 import tabulate_trees, get_galacticus_outputs

gout = h5py.File('galacticus.hdf5')

# Load latest snapshot (default)
trees = tabulate_trees(gout)
print(f"Number of trees: {len(trees)}")

# Load specific snapshot
outputs = get_galacticus_outputs(gout)
trees_early = tabulate_trees(gout, out_index=outputs[0])  # First snapshot

# Analyze each tree independently
for i, tree in enumerate(trees):
    print(f"Tree {i}: {len(tree['basicMass'])} nodes")

gout.close()
```

**Implementation Details:**
- Reads all datasets from `Outputs/Output{index}/nodeData`
- Filters to keep only datasets with shape[0] == nodecount
- Creates separate NodeProperties for each tree using `_startn` and `_stopn` slicing
- Applies custom datasets if provided

**Notes:**
- Each tree is analyzed independently
- Use with `summarize=True` in gscript-decorated functions to get statistics across trees
- Tree separation allows multi-tree simulations to be analyzed individually

---

## Node Extraction

### nodedata()

**Module:** `subscript.scripts.nodes`

**Signature:**
```python
@gscript
def nodedata(gout: NodeProperties, key: (str | Iterable[str]), **kwargs) → np.ndarray | list[np.ndarray]
```

**Description:**
Extract one or more fields from Galacticus node data. Wrapped with `@gscript` decorator for automatic filtering, tree handling, and statistics.

**Parameters:**
- `gout` (h5py.File | NodeProperties | dict): Galacticus data
- `key` (str or iterable[str]): Single key or list of keys to extract
- `nfilter` (Callable | np.ndarray[bool], optional): Node filter to apply
- `summarize` (bool, optional): Return statistics across trees
- `statfuncs` (list[Callable], optional): Statistics functions (default: [np.mean])
- `out_index` (int, optional): Output index if gout is h5py.File

**Returns:**
- np.ndarray: If single key and single tree
- list[np.ndarray]: If multiple trees
- list[list[np.ndarray]]: If multiple keys
- Depends on `summarize` parameter

**Examples:**

Single property:
```python
from subscript.scripts.nodes import nodedata
from subscript.defaults import ParamKeys

mass = nodedata(gout, key=ParamKeys.mass_basic)
```

Multiple properties:
```python
mass, radius, z = nodedata(
    gout,
    key=[ParamKeys.mass_basic, ParamKeys.rvir, ParamKeys.z_lastisolated]
)
```

With filter:
```python
from subscript.scripts.nfilters import subhalos
subhalo_mass = nodedata(gout, key=ParamKeys.mass_basic, nfilter=subhalos)
```

Statistics across trees:
```python
result = nodedata(
    gout,
    key=ParamKeys.mass_basic,
    nfilter=subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
# result[0] = mean_array
# result[1] = std_array
```

**Notes:**
- Automatically handles h5py.File, NodeProperties, or dict input
- Applies `nfilter` if provided
- Returns per-tree list for h5py.File with multiple trees
- Use `summarize=True` for cross-tree statistics

---

### nodecount()

**Module:** `subscript.scripts.nodes`

**Signature:**
```python
@gscript
def nodecount(gout: NodeProperties, **kwargs) → int
```

**Description:**
Count the number of nodes selected by the current filter.

**Parameters:**
- `gout` (h5py.File | NodeProperties | dict): Galacticus data
- `nfilter` (Callable | np.ndarray[bool], optional): Node filter
- `summarize` (bool, optional): Return statistics across trees
- `statfuncs` (list[Callable], optional): Statistics functions
- `out_index` (int, optional): Output index if gout is h5py.File

**Returns:**
- int: Number of nodes passing filter
- list[int]: One per tree if from h5py.File
- Depends on `summarize` parameter

**Examples:**

Total nodes:
```python
total = nodecount(gout)
```

Filtered nodes:
```python
from subscript.scripts.nfilters import subhalos
subhalo_count = nodecount(gout, nfilter=subhalos)
```

Statistics across trees:
```python
counts = nodecount(
    gout,
    nfilter=subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
```

**Implementation:**
- Sums the active filter mask: `np.sum(gout.get_filter())`
- Very fast operation on filtered data

---

## Node Filters

All filter functions are decorated with `@gscript` and support the freeze pattern.

### Classification Filters

#### allnodes()

**Signature:** `allnodes(gout, **kwargs) → np.ndarray[bool]`

**Description:** Return a boolean array selecting all nodes (no filtering).

```python
all_filter = nf.allnodes(gout)
# Returns array of all True
```

#### hosthalos()

**Signature:** `hosthalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs) → np.ndarray[bool]`

**Description:** Select only host halos (isolated nodes where `is_isolated == 1`).

```python
host_filter = nf.hosthalos(gout)
host_mass = nodedata(gout, ParamKeys.mass_basic, nfilter=host_filter)
```

#### subhalos()

**Signature:** `subhalos(gout, key_is_isolated=ParamKeys.is_isolated, **kwargs) → np.ndarray[bool]`

**Description:** Select only subhalos (non-isolated nodes where `is_isolated == 0`).

```python
sub_filter = nf.subhalos(gout)
subhalo_mass = nodedata(gout, ParamKeys.mass_basic, nfilter=sub_filter)
```

#### most_massive_progenitor()

**Signature:** `most_massive_progenitor(gout, key_mass_basic=ParamKeys.mass_basic, **kwargs) → np.ndarray[bool]`

**Description:** Select only the single most massive node in the dataset.

```python
mmp_filter = nf.most_massive_progenitor(gout)
# Returns array with True at index of max mass, False elsewhere
```

**Notes:**
- Useful for tracking main progenitor branch in merger trees
- Returns single True value at argmax(mass)

---

### Spatial Filters

#### r3d()

**Signature:** `r3d(gout, rmin, rmax, **kwargs) → np.ndarray[bool]`

**Description:** Select nodes within a 3D radial interval from the origin.

```python
nearby_filter = nf.r3d(None, 0, 0.1)  # Freeze pattern
mass_nearby = nodedata(gout, ParamKeys.mass_bound, nfilter=nearby_filter)

# Or directly
filter_direct = nf.r3d(gout, 0, 0.1)
```

**Parameters:**
- `rmin` (float): Minimum 3D radius
- `rmax` (float): Maximum 3D radius
- Returns nodes where `rmin ≤ r₃d ≤ rmax`

**Notes:**
- Uses Euclidean distance: `r = sqrt(x² + y² + z²)`
- Default coordinates: `ParamKeys.x/y/z`
- Can be frozen with `None` for reuse

#### r2d()

**Signature:** `r2d(gout, rmin, rmax, normvector, **kwargs) → np.ndarray[bool]`

**Description:** Select nodes within a 2D projected radial interval.

```python
# Project onto plane perpendicular to z-axis
lens_filter = nf.r2d(None, 20e-3, 50e-3, normvector=np.array([0, 0, 1]))
mass_in_range = nodedata(gout, ParamKeys.mass_bound, nfilter=lens_filter)
```

**Parameters:**
- `rmin`, `rmax` (float): 2D radius bounds
- `normvector` (array-like, shape (3,) or (n, 3)): Normal vector(s) defining projection plane

**Notes:**
- Projects 3D position onto plane perpendicular to `normvector`
- Uses formula: `r₂d = sqrt(|r|² - (r·n̂)²)`
- Can handle multiple projection axes as separate "trees"
- Common vectors: `[1,0,0]`, `[0,1,0]`, `[0,0,1]` for x, y, z projections

#### withinrv()

**Signature:** `withinrv(gout, key_rvir=ParamKeys.rvir, key_mass_basic=ParamKeys.mass_basic, inclusive=True, **kwargs) → np.ndarray[bool]`

**Description:** Select subhalos within the virial radius of the most massive progenitor.

```python
vir_filter = nf.withinrv(gout)
interior_subhalos = nodedata(gout, ParamKeys.mass_bound, nfilter=vir_filter)
```

**Parameters:**
- `key_rvir` (str): Key for virial radius
- `key_mass_basic` (str): Key for identifying most massive progenitor
- `inclusive` (bool): Include nodes exactly at virial radius

**Implementation:**
- Finds most massive progenitor using `most_massive_progenitor()`
- Gets its virial radius
- Selects nodes with `r₃d ≤ rvir` (or `<` if inclusive=False)

---

### Property-based Filters

#### interval()

**Signature:**
```python
def interval(gout, min, max, key=None, getval=None,
             inclmin=True, inclmax=False, **kwargs) → np.ndarray[bool]
```

**Description:** Select nodes where a property falls within a numerical interval.

```python
# By key
mass_filter = nf.interval(gout, 1e9, 1e10, key=ParamKeys.mass_bound)

# By computed value
r3d_filter = nf.interval(gout, 0, 0.05, getval=project3d)

# Reusable (frozen)
mass_frozen = nf.interval(None, 1e9, 1e10, key=ParamKeys.mass_bound)
```

**Parameters:**
- `min`, `max` (float): Interval bounds
- `key` (str, optional): Property key to filter on
- `getval` (Callable, optional): Function to compute values from gout
- `inclmin`, `inclmax` (bool): Whether to include bounds (default: min inclusive, max exclusive)

**Notes:**
- Either `key` or `getval` must be provided
- Default interval is `[min, max)` (left-inclusive, right-exclusive)
- Can be frozen with `None` for reuse

#### subhalos_valid()

**Signature:**
```python
def subhalos_valid(gout, mass_min=-np.inf, mass_max=np.inf,
                   key_mass=ParamKeys.mass, ...) → np.ndarray[bool]
```

**Description:** Convenience filter combining subhalos classification, virial radius cut, and mass range.

```python
valid_filter = nf.subhalos_valid(gout, mass_min=1e9, mass_max=1e12)
valid_mass = nodedata(gout, ParamKeys.mass_bound, nfilter=valid_filter)
```

**Applies:**
1. `subhalos()` - Select only subhalos
2. `withinrv()` - Within virial radius
3. `interval()` - Within mass range

**Notes:**
- Shorthand for common analysis workflow
- Default mass bounds are infinite (no constraint)

---

### Boolean Logic Operators

#### logical_and()

**Signature:**
```python
def logical_and(arg1: (np.ndarray[bool] | Callable),
                arg2: (np.ndarray[bool] | Callable)) → Callable
```

**Description:** Create logical AND of two filters or boolean arrays.

```python
from subscript.scripts.nfilters import logical_and, subhalos, r3d

# Combine filters
inner_subhalos = logical_and(subhalos, r3d(None, 0, 0.1))
mass = nodedata(gout, ParamKeys.mass_bound, nfilter=inner_subhalos)

# Combine arrays
array_and = logical_and(bool_array1, bool_array2)
```

**Notes:**
- Works with both callables (filters) and arrays
- Returns callable that can be used as `nfilter`
- Implements: `arg1(...) & arg2(...)`

#### logical_or()

**Signature:**
```python
def logical_or(arg1: (np.ndarray[bool] | Callable),
               arg2: (np.ndarray[bool] | Callable)) → Callable
```

**Description:** Create logical OR of two filters or boolean arrays.

```python
# Nodes either in inner region OR above mass threshold
inner_or_massive = logical_or(
    r3d(None, 0, 0.05),
    interval(None, 1e11, np.inf, key=ParamKeys.mass_bound)
)
```

#### logical_not()

**Signature:**
```python
def logical_not(arg: (np.ndarray[bool] | Callable)) → Callable
```

**Description:** Create logical NOT (inversion) of a filter or array.

```python
# Select host halos (inverse of subhalos)
host_filter = logical_not(subhalos)
```

---

### Deprecated Filter Names

The following old names still work but are deprecated (warnings can be disabled):

| Old Name | New Name | Status |
|----------|----------|--------|
| `nfilter_all()` | `allnodes()` | Deprecated |
| `nfilter_halos()` | `hosthalos()` | Deprecated |
| `nfilter_subhalos()` | `subhalos()` | Deprecated |
| `nfilter_most_massive_progenitor()` | `most_massive_progenitor()` | Deprecated |
| `nfilter_virialized()` | `withinrv()` | Deprecated |
| `nfilter_subhalos_valid()` | `subhalos_valid()` | Deprecated |
| `nfilter_range()` | `interval()` | Deprecated |
| `nfilter_project3d()` | `r3d()` | Deprecated |
| `nfilter_project2d()` | `r2d()` | Deprecated |
| `nfor()` | `logical_or()` | Deprecated |
| `nfand()` | `logical_and()` | Deprecated |
| `nfnot()` | `logical_not()` | Deprecated |
| `nfiler_project3d()` | `r3d()` | Typo (deprecated) |

**Disable deprecation warnings:**
```python
from subscript.defaults import Meta
Meta.disableDepreciatedWarning = True
```

---

## Spatial Projections

### project3d()

**Module:** `subscript.scripts.spatial`

**Signature:**
```python
@gscript
def project3d(gout, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs) → np.ndarray
```

**Description:** Compute the 3D Euclidean distance of each node from the origin.

**Parameters:**
- `gout` (NodeProperties | dict): Galacticus data
- `key_x/y/z` (str): Keys for coordinate arrays (default: ParamKeys.x/y/z)

**Returns:**
- np.ndarray (shape (N,)): 3D distance for each node

**Formula:**
```
r₃d = sqrt(x² + y² + z²)
```

**Example:**
```python
from subscript.scripts.spatial import project3d

r3d = project3d(gout)
print(r3d.shape)  # (N,)
print(r3d.min(), r3d.max())
```

**Notes:**
- Default coordinates: `ParamKeys.x`, `ParamKeys.y`, `ParamKeys.z`
- Can use relative coordinates: `key_x=ParamKeys.relx`, etc.
- Can combine with filters for spatial analysis

---

### project2d()

**Module:** `subscript.scripts.spatial`

**Signature:**
```python
@gscript_proj
def project2d(gout, normvector, key_x=ParamKeys.x, key_y=ParamKeys.y, key_z=ParamKeys.z, **kwargs) → np.ndarray
```

**Description:** Project 3D positions onto a 2D plane orthogonal to a normal vector.

**Parameters:**
- `gout` (NodeProperties | dict): Galacticus data
- `normvector` (array-like, shape (3,) or (n, 3)): Normal vector(s) defining projection plane
- `key_x/y/z` (str): Keys for coordinate arrays

**Returns:**
- np.ndarray (shape (N,)): 2D projected distance for each node
- If multiple normvectors: shape (m*N,) where m is number of vectors

**Formula:**
```
r₂d = sqrt(|r|² - (r·n̂)²)
where n̂ = normvector / |normvector|
```

**Examples:**

Single projection:
```python
import numpy as np
from subscript.scripts.spatial import project2d

# Project onto plane perpendicular to z-axis
r2d_z = project2d(gout, normvector=np.array([0, 0, 1]))
```

Multiple projections:
```python
# Project along x, y, z axes simultaneously
r2d_all = project2d(gout, normvector=np.identity(3))
# Returns list of 3 arrays (one per axis)
```

With filters:
```python
from subscript.scripts.nfilters import r2d
lens_region = r2d(None, 20e-3, 50e-3, normvector=np.array([0, 0, 1]))
```

**Notes:**
- Decorated with `@gscript_proj` for multiple projection support
- Projects perpendicular to `normvector`
- Each vector in multi-vector case treated as separate "tree"
- Useful for lensing studies (projecting along line-of-sight)

---

## Histograms & Distributions

### hist()

**Module:** `subscript.scripts.histograms`

**Signature:**
```python
@gscript
def hist(gout, key_hist=None, getval=None, bins=None, range=None,
         density=False, weights=None, kwargs_hist=None, **kwargs) → tuple
```

**Description:** Compute a histogram using data from Galacticus output.

**Parameters:**
- `gout` (NodeProperties | dict): Galacticus data
- `key_hist` (str, optional): Key to histogram
- `getval` (Callable, optional): Function to compute values
- `bins`, `range`, `density`, `weights`: Passed to `np.histogram()`
- `nfilter` (optional): Filter to apply

**Returns:**
- tuple: (counts, bin_edges) as from `np.histogram()`

**Examples:**

Histogram by key:
```python
from subscript.scripts.histograms import hist
from subscript.defaults import ParamKeys

counts, edges = hist(gout, key_hist=ParamKeys.mass_bound, bins=30)
```

Histogram computed quantity:
```python
from subscript.scripts.spatial import project3d

counts, edges = hist(gout, getval=project3d, bins=np.linspace(0, 1, 20))
```

With filter and statistics:
```python
from subscript.scripts.nfilters import subhalos

result = hist(
    gout,
    key_hist=ParamKeys.mass_bound,
    bins=np.logspace(9, 13, 30),
    nfilter=subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
counts_mean, counts_std = result[0]
```

---

### massfunction()

**Module:** `subscript.scripts.histograms`

**Signature:**
```python
@gscript
def massfunction(gout, key_mass=ParamKeys.mass, bins=None, range=None, **kwargs) → tuple
```

**Description:** Compute mass function as number density per unit mass bin width: dn/dM.

**Parameters:**
- `gout` (NodeProperties | dict): Galacticus data
- `key_mass` (str): Mass key (default: ParamKeys.mass)
- `bins`, `range`: Histogram parameters
- `nfilter` (optional): Filter to apply

**Returns:**
- tuple: (dn/dM, bin_edges)

**Formula:**
```
dn/dM = histogram / bin_width
```

**Example:**
```python
from subscript.scripts.histograms import massfunction
from subscript.scripts.nfilters import subhalos
from subscript.defaults import ParamKeys
import numpy as np
import matplotlib.pyplot as plt

# Mass function of subhalos
dn_dm, m_bins = massfunction(
    gout,
    bins=np.logspace(9, 13, 30),
    key_mass=ParamKeys.mass_bound,
    nfilter=subhalos
)

# Plot
plt.loglog(m_bins[:-1], dn_dm)
plt.xlabel('Mass [M☉]')
plt.ylabel('dn/dM')
```

**Notes:**
- Divides histogram counts by bin widths
- Good for comparing across different bin configurations
- Can be normalized (density=True) in histogram

---

### spatial3d_dn()

**Signature:**
```python
@gscript
def spatial3d_dn(gout, bins=None, range=None, kwargs_hist=None, **kwargs) → tuple
```

**Description:** Compute 3D radial number density distribution: counts per radial bin.

**Returns:**
- tuple: (counts, bin_edges)

---

### spatial3d_dndv()

**Module:** `subscript.scripts.histograms`

**Signature:**
```python
@gscript
def spatial3d_dndv(gout, bins=None, range=None, kwargs_hist=None, **kwargs) → tuple
```

**Description:** Compute 3D radial number density per unit volume: dn/dV.

**Parameters:**
- `gout` (NodeProperties | dict): Galacticus data
- `bins`: Radius bin edges
- `nfilter` (optional): Filter to apply

**Returns:**
- tuple: (dn/dV, bin_edges)

**Formula:**
```
dV = (4/3)π(r₁³ - r₀³)
dn/dV = counts / dV
```

**Example:**
```python
from subscript.scripts.histograms import spatial3d_dndv
from subscript.scripts.nfilters import subhalos

dn_dv, r_bins = spatial3d_dndv(
    gout,
    bins=np.linspace(0, 1, 20),
    nfilter=subhalos
)

plt.loglog(r_bins[:-1], dn_dv)
plt.xlabel('Radius [Mpc]')
plt.ylabel('dn/dV')
```

**Notes:**
- Uses 3D spherical volume elements
- Useful for radial density profiles

---

### spatial2d_dn()

**Signature:**
```python
@gscript_proj
def spatial2d_dn(gout, normvector, bins=None, range=None, kwargs_hist=None, **kwargs) → tuple
```

**Description:** Compute 2D projected number density distribution: counts per projected radial bin.

**Returns:**
- tuple: (counts, bin_edges)

---

### spatial2d_dnda()

**Module:** `subscript.scripts.histograms`

**Signature:**
```python
@gscript_proj
def spatial2d_dnda(gout, normvector, bins=None, range=None, kwargs_hist=None, **kwargs) → tuple
```

**Description:** Compute 2D projected number density per unit area: dn/dA.

**Parameters:**
- `gout` (NodeProperties | dict): Galacticus data
- `normvector` (array-like): Projection axis
- `bins`: Projected radius bin edges

**Returns:**
- tuple: (dn/dA, bin_edges)

**Formula:**
```
dA = π(r₁² - r₀²)
dn/dA = counts / dA
```

**Example:**
```python
from subscript.scripts.histograms import spatial2d_dnda
import numpy as np

# Surface density along z-axis
dn_da, r_bins = spatial2d_dnda(
    gout,
    normvector=np.array([0, 0, 1]),
    bins=np.linspace(0, 0.2, 20)
)

plt.semilogy(r_bins[:-1], dn_da)
plt.xlabel('Projected radius [Mpc]')
plt.ylabel('dn/dA')
```

**Notes:**
- Useful for lensing convergence/surface density
- Can handle multiple projection axes

---

### Histogram Helper Functions

#### bin_avg()

**Signature:** `bin_avg(bins) → np.ndarray`

**Description:** Compute bin center values from bin edges.

```python
from subscript.scripts.histograms import bin_avg

centers = bin_avg(bin_edges)
# For edges [0, 1, 2, 3], returns [0.5, 1.5, 2.5]
```

#### bin_size()

**Signature:** `bin_size(bins) → np.ndarray`

**Description:** Compute bin widths from bin edges.

```python
from subscript.scripts.histograms import bin_size

widths = bin_size(bin_edges)
# For edges [0, 1, 3, 4], returns [1, 2, 1]
```

---

## Decorators & Wrappers

### @gscript Decorator

**Module:** `subscript.wrappers`

**Description:** Decorator that wraps a function to automatically:
- Handle multiple input types (h5py.File, NodeProperties, dict, list)
- Apply node filters
- Iterate over multiple merger trees
- Compute statistics across trees

**Function Signature:**
```python
def gscript(func: Callable) → Callable
```

**Wrapped Function Accepts:**
- `gout` (h5py.File | NodeProperties | dict | list): Input data
- `nfilter` (Callable | np.ndarray[bool], optional): Node filter
- `summarize` (bool, optional): Compute statistics across trees
- `statfuncs` (list[Callable], optional): Statistics functions (default: [np.mean])
- `out_index` (int, optional): Output index if gout is h5py.File
- `**kwargs`: Passed to wrapped function

**Example:**
```python
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys
import numpy as np

@gscript
def custom_analysis(gout, **kwargs):
    """Compute mass ratio for filtered nodes."""
    mass_bound = gout[ParamKeys.mass_bound]
    mass_basic = gout[ParamKeys.mass_basic]
    return np.mean(mass_bound / mass_basic)

# Use with all SubScript features
from subscript.scripts.nfilters import subhalos

result = custom_analysis(
    gout,
    nfilter=subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
# result[0] = mean of (mean mass ratio per tree)
# result[1] = std of (mean mass ratio per tree)
```

**Implementation Notes:**
- Automatically unfilters data before applying custom filter
- Iterates over all trees if given h5py.File
- Applies summarization to combine per-tree results
- Can return lists of arrays (multiple outputs)

---

### @gscript_proj Decorator

**Module:** `subscript.wrappers`

**Description:** Specialized decorator for functions using projection with one or multiple normal vectors. Treats each vector as a separate "tree".

**Example:**
```python
from subscript.wrappers import gscript_proj
from subscript.scripts.spatial import project2d

@gscript_proj
def custom_projection(gout, normvector, **kwargs):
    r2d = project2d(gout, normvector=normvector)
    return np.percentile(r2d, [25, 50, 75])

# Single vector
result_single = custom_projection(gout, normvector=[0, 0, 1])

# Multiple vectors - each as separate realization
result_multi = custom_projection(gout, normvector=np.identity(3))
# Returns 3 separate results (one per axis)
```

---

### freeze()

**Module:** `subscript.wrappers`

**Signature:**
```python
def freeze(func: Callable, **kwargs) → Callable
```

**Description:** Create a new function with fixed keyword arguments (partial application).

**Example:**
```python
from subscript.wrappers import freeze
from subscript.scripts.nodes import nodedata
from subscript.scripts.nfilters import subhalos
from subscript.defaults import ParamKeys

# Define once
get_subhalo_mass = freeze(
    nodedata,
    key=ParamKeys.mass_bound,
    nfilter=subhalos
)

# Use multiple times
for gout in galacticus_files:
    mass = get_subhalo_mass(gout)  # nfilter, key already applied
```

**Use Cases:**
- Reusable analysis functions
- Complex filter definitions
- Consistent cross-file analysis

---

### multiproj()

**Module:** `subscript.wrappers`

**Signature:**
```python
def multiproj(func: Callable, nfilter: Callable | np.ndarray) → Callable
```

**Description:** Create a projection function with fixed filter and support for multiple normal vectors.

**Example:**
```python
from subscript.wrappers import multiproj
from subscript.scripts.nodes import nodecount
from subscript.scripts.nfilters import subhalos
import numpy as np

# Create function
count_subhalos_proj = multiproj(nodecount, nfilter=subhalos)

# Use with different axes
result = count_subhalos_proj(gout, normvector=np.identity(3))
# Returns count of subhalos in each projection
```

---

### format_nodedata()

**Module:** `subscript.wrappers`

**Signature:**
```python
def format_nodedata(gout: (h5py.File | NodeProperties | dict | Iterable),
                    out_index: int = -1) → list[NodeProperties]
```

**Description:** Convert various data types into a list of NodeProperties objects.

**Parameters:**
- `gout`: h5py.File, NodeProperties, dict, or iterable of these
- `out_index`: Output index if gout is h5py.File

**Returns:**
- list[NodeProperties]: List of NodeProperties objects

**Notes:**
- Used internally by `@gscript` decorator
- Handles arbitrary nesting of inputs
- Flattens to single list of NodeProperties

---

## Time-Series Tracking

### track_subhalos()

**Module:** `subscript.tracking`

**Signature:**
```python
def track_subhalos(galacticus_out: h5py.File, nodeIndices: array-like,
                   treeIndex: int, param_keys: list[str] = None) → tuple
```

**Description:** Extract time-series data for specified subhalo nodes across all Galacticus snapshots.

**Requirements:**
```xml
<!-- Galacticus must include nodeOperator: -->
<nodeOperator value="indexShift" />
```

**Parameters:**
- `galacticus_out` (h5py.File): Open Galacticus HDF5 file
- `nodeIndices` (array-like): Node indices to track (from `ParamKeys.node_index`)
- `treeIndex` (int): Which merger tree to extract from
- `param_keys` (list[str], optional): Parameters to extract (default: all available)

**Returns:**
- tuple: (subhalo_data, zsnaps)
  - `subhalo_data` (dict): {node_id: {param_key: time_series_array, 'zsnap': redshifts}}
  - `zsnaps` (np.ndarray): Redshifts at each snapshot (averaged over host halos)

**Example:**
```python
import h5py
from subscript.tracking import track_subhalos
from subscript.scripts.nodes import nodedata
from subscript.scripts.nfilters import interval, subhalos, logical_and
from subscript.tabulatehdf5 import tabulate_trees
from subscript.defaults import ParamKeys

gout = h5py.File('galacticus.hdf5')

# Select subhalos of interest
trees = tabulate_trees(gout)
mass_filter = logical_and(
    subhalos,
    interval(None, 1e9, 1e10, key=ParamKeys.mass_basic)
)
node_ids = nodedata(trees[0], 'nodeIndex', nfilter=mass_filter)

# Extract time-series
ts_data, zsnaps = track_subhalos(
    gout,
    nodeIndices=node_ids[:10],
    treeIndex=0,
    param_keys=[ParamKeys.mass_bound, ParamKeys.satellite_tidal_field]
)

# Access time-series for first subhalo
node_id = list(ts_data.keys())[0]
mass_evolution = ts_data[node_id][ParamKeys.mass_bound]
z_evolution = ts_data[node_id]['zsnap']

gout.close()
```

**Implementation Details:**
- Snapshots processed in reverse (newest to oldest)
- Returns per-snapshot redshift averaged over all nodes
- Each node tracks full parameter history
- Very memory-intensive for large samples

**Notes:**
- **Requires** `<nodeOperator value="indexShift" />` in Galacticus config
- Node indices must be consistent across snapshots
- Can be slow for many snapshots and subhalos
- Cache results to disk using pickle

---

### track_subhalo()

**Module:** `subscript.tracking`

**Signature:**
```python
def track_subhalo(subhalos_over_time: dict, zsnaps: np.ndarray,
                  nodeindex: int, param_keys: list[str]) → tuple
```

**Description:** Filter a single subhalo's time-series to retain only satellite phase snapshots.

**Parameters:**
- `subhalos_over_time` (dict): Output from `track_subhalos()`
- `zsnaps` (np.ndarray): Snapshot redshifts from `track_subhalos()`
- `nodeindex` (int): Node ID to extract
- `param_keys` (list[str]): Parameters to include in output

**Returns:**
- tuple: (filtered_data, filtered_zsnaps)
  - `filtered_data` (dict): {param_key: filtered_array}
  - `filtered_zsnaps` (np.ndarray): Filtered redshifts

**Filtering Criteria:**
- Removes snapshots where `is_isolated == 1` (host halo)
- Removes snapshots where `mass_bound ≤ 0` (unbound)

**Example:**
```python
from subscript.tracking import track_subhalos, track_subhalo
import matplotlib.pyplot as plt

# Get full time-series
ts_data, zsnaps = track_subhalos(gout, node_ids, treeIndex=0,
                                 param_keys=[ParamKeys.mass_bound, ParamKeys.is_isolated])

# Filter to satellite phase only
node_id = node_ids[0]
data_sat, z_sat = track_subhalo(
    ts_data, zsnaps, node_id,
    [ParamKeys.mass_bound]
)

# Plot mass evolution in satellite phase
plt.plot(z_sat, data_sat[ParamKeys.mass_bound])
plt.xlabel('Redshift')
plt.ylabel('Bound Mass [M☉]')
plt.show()
```

**Notes:**
- Automatically requires `is_isolated` and `mass_bound` in tracking data
- Removes early snapshots before infall and late unbound snapshots
- Useful for studying tidal stripping history

---

## Batch Processing

### macro_add()

**Module:** `subscript.macros`

**Signature:**
```python
def macro_add(macros: dict[str, Callable], macro: Callable,
              label: str = None, **kwargs) → dict[str, Callable]
```

**Description:** Add a new macro (analysis function) to a macro dictionary.

**Example:**
```python
from subscript.macros import macro_add
from subscript.wrappers import freeze
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import hosthalos
from subscript.defaults import ParamKeys

macros = {}

# Add host mass macro
macros = macro_add(
    macros,
    freeze(nodedata, key=ParamKeys.mass_basic, nfilter=hosthalos),
    label='host_mass'
)

# Add subhalo count
macros = macro_add(
    macros,
    freeze(nodecount),
    label='total_nodes'
)
```

---

### macro_run()

**Module:** `subscript.macros`

**Signature:**
```python
def macro_run(macros: dict[str, Callable], gouts: Iterable[h5py.File],
              statfuncs: list[Callable] = None, runner: Callable = None) → dict
```

**Description:** Execute a set of macros on multiple Galacticus outputs and compile results.

**Parameters:**
- `macros` (dict): Macro dictionary from `macro_add()`
- `gouts` (Iterable[h5py.File]): Multiple Galacticus output files
- `statfuncs` (list[Callable], optional): Statistics functions (default: [np.mean])
- `runner` (Callable, optional): Custom runner function

**Returns:**
- dict: Results structured as `{macro_label: {output_name: {statfunc_name: array}}}`

**Example:**
```python
import h5py
import numpy as np
from subscript.macros import macro_run

# Open multiple simulations
gout_files = [h5py.File(f) for f in ['sim1.hdf5', 'sim2.hdf5', 'sim3.hdf5']]

# Run all macros
results = macro_run(
    macros,
    gout_files,
    statfuncs=[np.mean, np.std]
)

# Access results
host_masses = results['host_mass']
# host_masses[filename]['mean'] = mean for that file
# host_masses[filename]['std'] = std for that file

for f in gout_files:
    f.close()
```

---

### macro_write_out_hdf5()

**Module:** `subscript.macros`

**Signature:**
```python
def macro_write_out_hdf5(f: h5py.File, macro_out: dict,
                         notes: str = None, stamp_date: bool = True) → None
```

**Description:** Write macro results to HDF5 file with optional metadata.

**Parameters:**
- `f` (h5py.File): Open HDF5 file to write to
- `macro_out` (dict): Results from `macro_run()`
- `notes` (str, optional): Metadata notes to store
- `stamp_date` (bool): Include current datetime

**Example:**
```python
import h5py
from subscript.macros import macro_write_out_hdf5

with h5py.File('analysis_results.hdf5', 'w') as f:
    macro_write_out_hdf5(
        f,
        results,
        notes="Comparison across 3 CDM + 3 SIDM simulations",
        stamp_date=True
    )
    # File now contains groups for each macro with datasets per output

# Later, read results
with h5py.File('analysis_results.hdf5', 'r') as f:
    host_mass_mean = f['host_mass']['mean'][:]
    print(f"Created: {f.attrs['date']}")
    print(f"Notes: {f.attrs['notes']}")
```

---

## External Data Integration

### symphony_to_galacticus_like_dict()

**Module:** `subscript.external`

**Signature:**
```python
def symphony_to_galacticus_like_dict(sim_data, z_snap,
                                     key_map: dict = KEY_MAP_SYMPHONY_DEFAULT,
                                     isnap: int = -1, tree_index: int = 1) → dict
```

**Description:** Convert Symphony simulation subhalo catalogs to Galacticus-compatible format.

**Parameters:**
- `sim_data`: Symphony halo data (structured array tuple)
- `z_snap` (np.ndarray): Redshift values at each snapshot
- `key_map` (dict, optional): Mapping from Galacticus keys to Symphony keys
- `isnap` (int): Snapshot index to extract (default: -1 for latest)
- `tree_index` (int): Tree index to assign to all halos

**Returns:**
- dict: Dictionary with Galacticus-like structure

**Default Mappings:**
- `mass_basic` ← `mpeak`
- `mass_bound` ← `mvir`
- `rvir` ← `rvir`
- `x/y/z` ← `x/y/z` (coordinates)
- `z_lastisolated` ← `merger_snap` (converted to redshift)
- `is_isolated` ← Computed (first halo = 1, rest = 0)

**Example:**
```python
from subscript.external import symphony_to_galacticus_like_dict
from subscript.scripts.nodes import nodedata
from subscript.scripts.nfilters import subhalos
from subscript.defaults import ParamKeys

# Load Symphony data (format-specific code)
# sim_data = load_symphony_data(...)
# z_snap = np.array([...])

# Convert to Galacticus format
gal_data = symphony_to_galacticus_like_dict(sim_data, z_snap)

# Now can use all SubScript functions
mass = nodedata(gal_data, key=ParamKeys.mass_basic, nfilter=subhalos)
```

**Notes:**
- Converts coordinate units (kpc → Mpc if needed)
- Creates custom `is_isolated` with first halo as host
- Assigns all subhalos to single tree
- Custom key_map allows for alternate source formats

---

## Utilities

### deprecated()

**Module:** `subscript.util`

**Signature:**
```python
def deprecated(reason: str) → Callable
```

**Description:** Decorator to mark functions as deprecated with warning.

**Example:**
```python
from subscript.util import deprecated

@deprecated("Use new_function() instead")
def old_function():
    pass

# Calling old_function() will emit DeprecationWarning
```

**Disable Warnings:**
```python
from subscript.defaults import Meta
Meta.disableDepreciatedWarning = True
```

---

### is_arraylike()

**Module:** `subscript.util`

**Signature:**
```python
def is_arraylike(obj) → bool
```

**Description:** Check if object can be converted to numpy array.

**Example:**
```python
from subscript.util import is_arraylike

is_arraylike([1, 2, 3])      # True
is_arraylike(np.array([1]))  # True
is_arraylike("string")        # False (ambiguous)
```

**Implementation:**
- Attempts `np.asarray(obj)`
- Returns True if successful, False if exception raised

---

## Configuration & Meta

### Meta Class

**Module:** `subscript.defaults`

**Attributes:**
- `cache` (bool): Cache HDF5 reads in memory (default: True)
  - True: Faster but uses more memory
  - False: Slower but lower memory footprint
- `disableDepreciatedWarning` (bool): Suppress deprecation warnings (default: False)

**Example:**
```python
from subscript.defaults import Meta

# Disable caching for large files
Meta.cache = False

# Suppress old function name warnings
Meta.disableDepreciatedWarning = True
```

---

## Summary Table

| Module | Function | Category | Key Use |
|--------|----------|----------|---------|
| `tabulatehdf5` | `tabulate_trees()` | Core | Load Galacticus snapshot |
| `tabulatehdf5` | `get_galacticus_outputs()` | Core | List available snapshots |
| `tabulatehdf5` | `get_custom_dsets()` | Core | Add custom node IDs |
| `scripts.nodes` | `nodedata()` | Extraction | Get node properties |
| `scripts.nodes` | `nodecount()` | Extraction | Count filtered nodes |
| `scripts.nfilters` | `subhalos()` | Filter | Select subhalos |
| `scripts.nfilters` | `hosthalos()` | Filter | Select host halos |
| `scripts.nfilters` | `r3d()` | Filter | 3D spatial selection |
| `scripts.nfilters` | `r2d()` | Filter | 2D projected selection |
| `scripts.nfilters` | `interval()` | Filter | Property range filter |
| `scripts.nfilters` | `logical_and/or/not()` | Filter | Combine filters |
| `scripts.spatial` | `project3d()` | Spatial | 3D distance |
| `scripts.spatial` | `project2d()` | Spatial | 2D projection |
| `scripts.histograms` | `massfunction()` | Analysis | dn/dM distribution |
| `scripts.histograms` | `spatial3d_dndv()` | Analysis | 3D density profile |
| `scripts.histograms` | `spatial2d_dnda()` | Analysis | 2D surface density |
| `wrappers` | `@gscript` | Decorator | Auto-filter, multi-tree |
| `wrappers` | `freeze()` | Wrapper | Partial application |
| `tracking` | `track_subhalos()` | Time-series | Extract time evolution |
| `tracking` | `track_subhalo()` | Time-series | Filter to satellite phase |
| `macros` | `macro_run()` | Batch | Multi-file analysis |
| `external` | `symphony_to_galacticus_like_dict()` | Integration | Convert Symphony data |

