# SubScript: Galacticus Analysis Library

**SubScript** is a Python library providing ergonomic utility functions for analyzing Galacticus semi-analytic model outputs. 

**Repository:** https://github.com/cgannonucm/SubScript
**Package:** `subhaloscript` (v1.1.0)
**Author:** Charles Gannon (cgannon@ucmerced.edu)

---

## Quick Start

### Installation

```bash
# Via pip
pip install subhaloscript

# Via conda
conda install cgannonucm::subhaloscript
```

### Basic Workflow

```python
import h5py
from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.nodes import nodedata
from subscript.defaults import ParamKeys
import subscript.scripts.nfilters as nf

# 1. Open Galacticus HDF5 output
gout = h5py.File('galacticus_output.hdf5')

# 2. Extract node data from all trees
mass = nodedata(gout, key=ParamKeys.mass_basic, nfilter=nf.subhalos)

# 3. Close when done
gout.close()
```

---

## Core Architecture

### NodeProperties

A dictionary-like wrapper for Galacticus node data that supports:
- **Filtering**: Boolean masks to select subsets of nodes
- **Lazy loading**: Reads from h5py datasets only when accessed
- **Tree separation**: Automatically slices nodes into separate trees

**Key methods:**
- `filter(nodefilter)` - Apply a boolean filter
- `unfilter()` - Recursively remove all filters to access raw data
- `get_filter()` - Retrieve the current filter mask

### Multiple Merger Trees

Galacticus outputs contain multiple independent merger trees. Functions automatically:
- Process each tree separately when needed
- Return results per-tree as lists
- Support statistical summarization across all trees via `summarize=True`

### tabulate_trees()

Loads all node data from a Galacticus snapshot:

```python
from subscript.tabulatehdf5 import tabulate_trees, get_galacticus_outputs

# Get available output indices
outputs = get_galacticus_outputs(gout)  # Returns sorted snapshot indices

# Load specific output (default: latest)
trees = tabulate_trees(gout, out_index=-1)  # Returns list of NodeProperties

# Each tree can be analyzed independently
mass_tree0 = nodedata(trees[0], key=ParamKeys.mass_basic)
```

---

## Parameter Keys (ParamKeys)

Access Galacticus properties via `subscript.defaults.ParamKeys`:

### Spatial Coordinates
```python
ParamKeys.x, ParamKeys.y, ParamKeys.z              # Absolute orbital positions (kpc)
ParamKeys.relx, ParamKeys.rely, ParamKeys.relz    # Positions relative to host
```

### Mass Properties
```python
ParamKeys.mass / ParamKeys.mass_basic      # Infall mass (fixed at infall)
ParamKeys.mass_bound                       # Current bound mass (changes due to stripping)
ParamKeys.mass_halo_enclosed_current       # Enclosed mass within radius
```

### Structural Properties
```python
ParamKeys.rvir                             # Virial radius
ParamKeys.scale_radius / ParamKeys.rscale  # Scale radius (r_s) for density profile
ParamKeys.density_profile_radius           # Array of radii for density profile
ParamKeys.density_profile                  # Array of densities at radii
ParamKeys.concentration                    # Concentration parameter
ParamKeys.tnfw_rt                          # Tidal truncation radius (truncated NFW)
ParamKeys.tnfw_p0                          # Density normalization (truncated NFW)
ParamKeys.dark_matter_profile_scale        # Dark matter profile scale
```

### Classification & Hierarchy
```python
ParamKeys.is_isolated                      # 1 = host halo, 0 = subhalo
ParamKeys.hierarchylevel                   # Depth in merger tree
ParamKeys.hierarchy_level_depth            # Max depth below this node
ParamKeys.hierarchy_level_maximum          # Overall tree depth
```

### History & Infall
```python
ParamKeys.z_lastisolated                   # Redshift when last isolated (infall z)
ParamKeys.basic_time_last_isolated         # Time when last isolated
```

### Satellite Properties (subhalos only)
```python
ParamKeys.satellite_tidal_field            # Tidal field strength
ParamKeys.satellite_tidal_heating_normalized  # Normalized tidal heating
ParamKeys.satellite_velocity_x/y/z         # Velocity components relative to host
```

### Node Tracking
```python
ParamKeys.node_index                       # Unique node identifier (with indexShift operator)
ParamKeys.parent_index                     # Index of parent node
ParamKeys.satellite_index                  # Index if this is a satellite
ParamKeys.sibling_index                    # Index of sibling node
ParamKeys.subsampling_weight               # Subsampling weight for node
```

### Tree Metadata
```python
ParamKeys.merger_tree_index                # Which tree (0, 1, 2, ...)
ParamKeys.merger_tree_count                # Number of nodes in tree
ParamKeys.merger_tree_seed                 # Random seed for tree
ParamKeys.merger_tree_weight               # Weight of tree for statistics
ParamKeys.merger_tree_start_index          # Starting index in output
```

### Dark Matter Properties
```python
ParamKeys.dark_matter_temperature_virial   # Virial temperature
ParamKeys.dark_matter_velocity_virial      # Virial velocity
ParamKeys.dark_matter_profile_dmo_radius_velocity_max  # Radius of max velocity
ParamKeys.dark_matter_profile_dmo_velocity_max        # Maximum circular velocity
```

### Spin & Angular Momentum
```python
ParamKeys.spin_angular_momentum            # Magnitude of angular momentum
ParamKeys.spin_angular_momentum_vector_x/y/z  # Components of angular momentum vector
ParamKeys.sphere_anglularmomentum          # Spheroid angular momentum
```

### Spheroid Properties
```python
ParamKeys.sphere_radius                    # Spheroid radius
ParamKeys.sphere_mass_stellar              # Stellar mass in spheroid
ParamKeys.sphere_mass_gas                  # Gas mass in spheroid
```

### Custom Fields
```python
ParamKeys.custom_id                        # Custom integer ID (0, 1, 2, ...)
ParamKeys.custom_tree_index                # Which tree each node belongs to
ParamKeys.custom_tree_outputorder          # Position in output order
```

---

## Node Filters (nfilters)

Filters select specific subsets of nodes. Located in `subscript.scripts.nfilters`.

### Classification Filters

```python
import subscript.scripts.nfilters as nf

nf.allnodes(gout)                # All nodes
nf.hosthalos(gout)               # Host halos only (is_isolated == 1)
nf.subhalos(gout)                # Subhalos only (is_isolated == 0)
nf.most_massive_progenitor(gout) # Single most massive node
```

### Spatial Filters

```python
# 3D radius from center (origin)
nf.r3d(gout, rmin, rmax)                   # Returns nodes with r_3d in [rmin, rmax]

# 2D projected radius (perpendicular to a plane)
nf.r2d(gout, rmin, rmax, normvector)       # Returns nodes with r_2d in [rmin, rmax]
# normvector can be [1,0,0], [0,1,0], [0,0,1] for x,y,z projections
# Can also pass multiple vectors as (n, 3) array

# Within virial radius of most massive progenitor
nf.withinrv(gout)                          # Subhalos inside virial radius
```

### Property-based Filters

```python
# General numerical interval
nf.interval(gout, min, max, key=ParamKeys.mass_bound)  # Values in [min, max]
nf.interval(gout, min, max, getval=project3d)          # Custom computed values

# Convenience filter combining subhalos, virial radius, and mass range
nf.subhalos_valid(gout, mass_min=1e9, mass_max=1e12)   # All filters combined
```

### Filter Combinations

Combine filters using boolean logic:

```python
from subscript.scripts.nfilters import logical_and, logical_or, logical_not

# AND: Both conditions true
combined = nf.logical_and(
    nf.subhalos,
    nf.r3d(None, 0, 0.05)  # Passing None "freezes" arguments
)

# OR: Either condition true
either = nf.logical_or(filter1, filter2)

# NOT: Invert condition
inverted = nf.logical_not(nf.subhalos)  # Select only host halos
```

**Note on freeze pattern:** Passing `None` as first argument "freezes" the filter:
```python
# Define reusable filters
spatial_filter = nf.r3d(None, 0, 0.1)    # Freezes r_min, r_max; gout passed later
mass_filter = nf.interval(None, 1e9, 1e10, key=ParamKeys.mass_bound)

# Use multiple times
mass1 = nodedata(gout1, key=ParamKeys.mass_bound, nfilter=spatial_filter)
mass2 = nodedata(gout2, key=ParamKeys.mass_bound, nfilter=spatial_filter)
```

### Deprecated Filter Names

Old filter names still work but are deprecated (use new names):
- `nfilter_all()` → `allnodes()`
- `nfilter_halos()` → `hosthalos()`
- `nfilter_subhalos()` → `subhalos()`
- `nfilter_most_massive_progenitor()` → `most_massive_progenitor()`
- `nfilter_virialized()` → `withinrv()`
- `nfilter_subhalos_valid()` → `subhalos_valid()`
- `nfilter_range()` → `interval()`
- `nfilter_project3d()` → `r3d()`
- `nfilter_project2d()` → `r2d()`
- `nfor()` → `logical_or()`
- `nfand()` → `logical_and()`
- `nfnot()` → `logical_not()`

---

## Core Analysis Functions

### Extract Node Data: nodedata()

```python
from subscript.scripts.nodes import nodedata

# Single property - returns per-tree list or single array
mass = nodedata(gout, key=ParamKeys.mass_basic)

# Multiple properties - returns list of arrays
mass, radius, z = nodedata(gout, key=[ParamKeys.mass_basic, ParamKeys.rvir, ParamKeys.z_lastisolated])

# With filter
subhalo_mass = nodedata(gout, key=ParamKeys.mass_basic, nfilter=nf.subhalos)

# Summarize across trees with statistics
result = nodedata(
    gout,
    key=ParamKeys.mass_basic,
    nfilter=nf.subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]  # Returns [mean_array, std_array]
)
```

### Count Nodes: nodecount()

```python
from subscript.scripts.nodes import nodecount

# Count total nodes
total = nodecount(gout)

# Count filtered nodes
subhalo_count = nodecount(gout, nfilter=nf.subhalos)

# Summarize counts across trees
counts = nodecount(gout, summarize=True, statfuncs=[np.mean, np.std])
```

### Spatial Projections

#### project3d() - 3D Euclidean Distance

```python
from subscript.scripts.spatial import project3d

# Default: distance from origin using x, y, z keys
r3d = project3d(gout)

# Custom coordinate keys
r3d = project3d(gout, key_x='customX', key_y='customY', key_z='customZ')

# Use in filters
nearby = nodedata(gout, key=ParamKeys.mass_bound, nfilter=nf.r3d(None, 0, 0.1))
```

#### project2d() - 2D Projected Distance

Projects 3D positions onto 2D plane orthogonal to a normal vector:

```python
from subscript.scripts.spatial import project2d
import numpy as np

# Project onto plane perpendicular to z-axis
r2d_z = project2d(gout, normvector=np.array([0, 0, 1]))

# Project onto plane perpendicular to x-axis
r2d_x = project2d(gout, normvector=np.array([1, 0, 0]))

# Multiple projections at once (treats each as separate "tree")
axes = np.identity(3)  # x, y, z axes
r2d_all = project2d(gout, normvector=axes)  # Returns 3 separate results

# Use in filters
near_einstein = nodedata(
    gout,
    key=ParamKeys.mass_bound,
    nfilter=nf.r2d(None, 20e-3, 50e-3, normvector=np.array([0, 0, 1]))
)
```

### Histograms and Distributions

#### hist() - Generic Histogram

```python
from subscript.scripts.histograms import hist

# Histogram a single property
counts, bin_edges = hist(gout, key_hist=ParamKeys.mass_bound, bins=30)

# Histogram a computed quantity
counts, bin_edges = hist(gout, getval=project3d, bins=np.linspace(0, 1, 20))

# With weighting
counts, bin_edges = hist(
    gout,
    key_hist=ParamKeys.mass_bound,
    bins=np.logspace(9, 13, 30),
    density=True,
    weights=np.ones(100)
)
```

#### massfunction() - dn/dM

```python
from subscript.scripts.histograms import massfunction

# Mass function (number density per unit log mass)
dn_dm, m_bins = massfunction(
    gout,
    bins=np.logspace(9, 13, 30),
    key_mass=ParamKeys.mass_bound,
    nfilter=nf.subhalos
)

# With statistics across trees
result = massfunction(
    gout,
    bins=np.logspace(9, 13, 30),
    nfilter=nf.subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
mean_mf, std_mf = result[0], result[1]
```

#### spatial3d_dndv() - 3D Radial Density

```python
from subscript.scripts.histograms import spatial3d_dndv

# Number density per unit volume vs radius
dn_dv, r_bins = spatial3d_dndv(
    gout,
    bins=np.linspace(0, 1, 20),
    nfilter=nf.subhalos
)
```

#### spatial2d_dnda() - 2D Projected Density

```python
from subscript.scripts.histograms import spatial2d_dnda

# Number density per unit area in 2D projection
dn_da, r_bins = spatial2d_dnda(
    gout,
    normvector=np.array([0, 0, 1]),  # Project along z
    bins=np.linspace(0, 0.2, 20)
)
```

### Helper Functions

```python
from subscript.scripts.histograms import bin_avg, bin_size

# Get bin centers
bin_centers = bin_avg(bin_edges)

# Get bin widths
widths = bin_size(bin_edges)
```

---

## Advanced Patterns

### Decorator: @gscript

The `@gscript` decorator wraps functions to automatically:
- Handle h5py.File, NodeProperties, dicts, or lists
- Apply node filters
- Iterate over multiple trees
- Compute statistics across trees

```python
from subscript.wrappers import gscript
import numpy as np

@gscript
def custom_analysis(gout, **kwargs):
    """Custom function operating on filtered node data."""
    return np.mean(gout[ParamKeys.mass_bound] / gout[ParamKeys.mass_basic])

# Use with all SubScript features
result = custom_analysis(
    gout,
    nfilter=nf.subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
```

### Decorator: @gscript_proj

For functions using projection with multiple normal vectors:

```python
from subscript.wrappers import gscript_proj

@gscript_proj
def custom_projection(gout, normvector, **kwargs):
    """Custom function with projection."""
    r2d = project2d(gout, normvector=normvector)
    return np.percentile(r2d, [25, 50, 75])

# Pass single vector
result_single = custom_projection(gout, normvector=[0, 0, 1])

# Or multiple vectors - each treated as separate realization
result_multi = custom_projection(gout, normvector=np.identity(3))
```

### freeze() - Partial Application

Create reusable functions with fixed arguments:

```python
from subscript.wrappers import freeze

# Define once
get_subhalo_mass = freeze(nodedata, key=ParamKeys.mass_bound, nfilter=nf.subhalos)

# Use on multiple files
for gout in galacticus_files:
    mass = get_subhalo_mass(gout)
```

### multiproj() - Fixed Filter with Projection

```python
from subscript.wrappers import multiproj

# Create projection function with fixed filter
projected_subhalo_count = multiproj(nodecount, nfilter=nf.subhalos)

# Use with different projection axes
result = projected_subhalo_count(gout, normvector=np.identity(3))
```

---

## Time-Series Analysis: Tracking

Extract properties of individual subhalos across all Galacticus snapshots.

> **Prefer `subhalo_timeseries()`** when extracting time-series for subhalos in a full tree. It wraps `track_subhalos()` + `track_subhalo()` with automatic disk caching, making repeated runs significantly faster. Use the lower-level `track_subhalos()` / `track_subhalo()` directly only when you need fine-grained control over which nodes to track or which parameters to extract.

### track_subhalos() - Extract Full Time-Series

```python
from subscript.tracking import track_subhalos
from subscript.tabulatehdf5 import get_galacticus_outputs

# **REQUIREMENT**: Galacticus must be run with:
# <nodeOperator value="indexShift" />

# Get node indices to track (e.g., from z=0 snapshot)
node_ids = nodedata(trees[0], key='nodeIndex', nfilter=my_filter)

# Extract time-series for these nodes across all snapshots
subhalo_data, zsnaps = track_subhalos(
    gout,
    nodeIndices=node_ids,
    treeIndex=0,
    param_keys=[
        ParamKeys.mass_bound,
        ParamKeys.satellite_tidal_field,
        ParamKeys.is_isolated
    ]
)

# Returns:
# subhalo_data[node_id]['mass_bound'] -> array over time
# subhalo_data[node_id]['zsnap'] -> redshift at each snapshot
# zsnaps -> redshifts for each snapshot (averaged over host halos)
```

### track_subhalo() - Filter to Satellite-Only Snapshots

```python
from subscript.tracking import track_subhalo

# Get time-series for a single subhalo, filtering to bound satellite snapshots
node_id = node_ids[0]
filtered_data, filtered_zsnaps = track_subhalo(
    subhalo_data,
    zsnaps,
    node_id,
    param_keys=[ParamKeys.mass_bound, ParamKeys.satellite_tidal_field]
)

# Removes snapshots where:
# - is_isolated == 1 (host halo)
# - mass_bound <= 0 (unbound)

# Now can plot evolution
import matplotlib.pyplot as plt
plt.plot(filtered_zsnaps, filtered_data[ParamKeys.mass_bound])
plt.xlabel('Redshift')
plt.ylabel('Bound Mass [M_☉]')
```

### subhalo_timeseries() - Cached Full-Tree Time-Series

Convenience wrapper that extracts and **caches** time-series for every subhalo in a tree:

```python
from subscript.subhalo_timeseries import subhalo_timeseries

# Extract (or load from cache) all subhalo time-series for tree 0
result = subhalo_timeseries(gout, tree_index=0)

# Force recompute, ignoring cache
result = subhalo_timeseries(gout, tree_index=0, refresh=True)

# result structure:
# result[node_id]['data']   -> dict of {param_key: time_series_array}
# result[node_id]['zsnaps'] -> corresponding redshift array

# Plot bound-mass evolution for a single subhalo
import matplotlib.pyplot as plt
from subscript.defaults import ParamKeys

for node_id, ts in result.items():
    plt.plot(ts['zsnaps'], ts['data'][ParamKeys.mass_bound])

plt.xlabel('Redshift')
plt.ylabel('Bound Mass [M_☉]')
plt.show()
```

**Cache behaviour:**
- Cache file is written to the same directory as the HDF5 file
- Filename: `{stem}-{sha256[:16]}-tree{tree_index}.pkl`
- Cache is invalidated automatically when the file content changes (hash-based)
- Pass `refresh=True` to force recomputation

**Requirements:** same as `track_subhalos()` — Galacticus must be run with `<nodeOperator value="indexShift" />`.

---

## Batch Processing: Macros

Run multiple analysis functions across multiple Galacticus outputs:

### macro_add() - Build Macro Dictionary

```python
from subscript.macros import macro_add
from subscript.wrappers import freeze

macros = {}

# Add individual macros
macros = macro_add(
    macros,
    freeze(nodedata, key=ParamKeys.mass_basic, nfilter=nf.hosthalos),
    label='host_mass'
)

macros = macro_add(
    macros,
    freeze(nodecount, nfilter=nf.subhalos),
    label='subhalo_count'
)
```

### macro_run() - Execute on Multiple Files

```python
from subscript.macros import macro_run
import h5py

# Open multiple Galacticus output files
gout_files = [h5py.File(f) for f in ['sim1.hdf5', 'sim2.hdf5', 'sim3.hdf5']]

# Run macros
results = macro_run(
    macros,
    gout_files,
    statfuncs=[np.mean, np.std]
)

# Results structure:
# results['host_mass']['out0'] -> array of host mass means
# results['host_mass']['out1'] -> array of host mass stds
# results['subhalo_count']['out0'] -> array of subhalo counts
```

### macro_write_out_hdf5() - Save Results

```python
from subscript.macros import macro_write_out_hdf5

with h5py.File('analysis_results.hdf5', 'w') as f:
    macro_write_out_hdf5(
        f,
        results,
        notes="Comparison across 3 simulations",
        stamp_date=True
    )
```

---

## Configuration

### Meta Settings

```python
from subscript.defaults import Meta

# Disable deprecation warnings for old filter names
Meta.disableDepreciatedWarning = False

# Units system (see Astropy Units Integration below)
Meta.units_enable = False          # Enable/disable astropy units (default: False)
Meta.units_in_si = None            # Dict mapping keys → astropy SI units
Meta.units_in_si_conversion = None # Dict mapping keys → numerical SI conversion factors
Meta.unit_bases = [apu.Msun, apu.kpc, apu.Myr]  # Base units for decomposition
```

---

## Astropy Units Integration

SubScript supports optional astropy units on all node properties via `subscript.units`.

### Quick Setup

```python
import h5py
from subscript.units import enableUnitsFromGalacticus

gout = h5py.File('galacticus.hdf5')

# Read unitsInSI from the HDF5 file and enable units globally
enableUnitsFromGalacticus(gout)

# Now all nodedata() calls return astropy Quantities in the base unit system
# Default bases: [Msun, kpc, Myr]
```

### Functions

#### `galacticus_units_si(gal_out)` — Extract SI conversion factors

Reads the `unitsInSI` attribute from every dataset in the final output's `nodeData` group and returns a dict of `{property_name: conversion_factor}`.

```python
from subscript.units import galacticus_units_si

units_si = galacticus_units_si(gout)
# units_si['basicMass'] → 1.989e30  (kg per unit stored in file)
```

#### `enableUnits(units_in_si_conversion, units_in_si=UNITS_IN_SI, base_units=Meta.unit_bases)` — Enable units manually

```python
from subscript.units import enableUnits, UNITS_IN_SI
from subscript.units import galacticus_units_si

conversion = galacticus_units_si(gout)
enableUnits(conversion)
# Meta.units_enable is now True; node properties return astropy Quantities
```

#### `enableUnitsFromGalacticus(galacticus_output, ...)` — Enable from file

Convenience wrapper that calls `galacticus_units_si()` then `enableUnits()` in one step.

### `UNITS_IN_SI` — Default unit map

Pre-built dict mapping all `ParamKeys` to their astropy SI units:

```python
from subscript.units import UNITS_IN_SI
from astropy import units as apu

# Examples:
# UNITS_IN_SI[ParamKeys.mass_basic]  → apu.kg
# UNITS_IN_SI[ParamKeys.x]           → apu.m
# UNITS_IN_SI[ParamKeys.rvir]        → apu.m
# UNITS_IN_SI[ParamKeys.satellite_tidal_field] → 1/apu.s**2
```

When units are enabled, `NodeProperties.__getitem__` automatically converts values:
```python
out_quantity = (raw_value * conversion_factor * si_unit).decompose(base_units)
```

---

## External Data: Symphony Integration

Convert Symphony simulation data to Galacticus-like format:

```python
from subscript.external import symphony_to_galacticus_like_dict

# Convert Symphony halo catalog to Galacticus format
gal_dict = symphony_to_galacticus_like_dict(
    sim_data,              # Symphony data arrays
    z_snap,                # Redshift values
    key_map=None,          # Custom mapping (uses defaults)
    isnap=-1,              # Snapshot index
    tree_index=1           # Tree index to assign
)

# Now can use with all SubScript functions
mass = nodedata(gal_dict, key=ParamKeys.mass_basic, nfilter=nf.subhalos)
```

---

## Common Workflows

### 1. Analyze Single Snapshot

```python
import h5py
import numpy as np
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import subhalos, r2d
from subscript.defaults import ParamKeys

gout = h5py.File('galacticus.hdf5')

# Basic properties
mass = nodedata(gout, ParamKeys.mass_bound, nfilter=subhalos)
print(f"Subhalo count: {nodecount(gout, nfilter=subhalos)}")
print(f"Mass range: {np.min(mass):.2e} - {np.max(mass):.2e}")

# Spatial selection (within Einstein radius for lensing)
lens_filter = r2d(None, 20e-3, 50e-3, normvector=np.array([0, 0, 1]))
lens_mass = nodedata(gout, ParamKeys.mass_bound, nfilter=lens_filter)
print(f"Subhalos in Einstein radius: {len(lens_mass)}")

gout.close()
```

### 2. Compare Distributions Across Simulations

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt
from subscript.scripts.histograms import massfunction
from subscript.scripts.nfilters import subhalos
from subscript.defaults import ParamKeys

# Open multiple simulations
sims = [h5py.File(f) for f in ['sim1.hdf5', 'sim2.hdf5', 'sim3.hdf5']]

bins = np.logspace(8, 12, 25)
fig, ax = plt.subplots()

for sim, label in zip(sims, ['CDM', 'SIDM 1', 'SIDM 2']):
    dn_dm, m_bins = massfunction(
        sim,
        bins=bins,
        nfilter=subhalos,
        summarize=True
    )
    ax.loglog(m_bins[:-1], dn_dm, label=label)

ax.set_xlabel('Mass [M_☉]')
ax.set_ylabel('dn/dM [dex$^{-1}$ Mpc$^{-3}$]')
ax.legend()
plt.show()

for sim in sims:
    sim.close()
```

### 3. Track Individual Subhalo Evolution

```python
import h5py
import numpy as np
from subscript.scripts.nodes import nodedata
from subscript.scripts.nfilters import interval, subhalos, logical_and
from subscript.tracking import track_subhalos, track_subhalo
from subscript.tabulatehdf5 import tabulate_trees
from subscript.defaults import ParamKeys

gout = h5py.File('galacticus.hdf5')

# Select subhalos of interest at z=0
trees = tabulate_trees(gout)
mass_filter = logical_and(
    subhalos,
    interval(None, 1e9, 1e10, key=ParamKeys.mass_basic)
)
node_ids = nodedata(trees[0], 'nodeIndex', nfilter=mass_filter)

# Extract full time-series
ts_data, zsnaps = track_subhalos(
    gout,
    nodeIndices=node_ids[:5],  # First 5 subhalos
    treeIndex=0,
    param_keys=[ParamKeys.mass_bound, ParamKeys.satellite_tidal_field]
)

# Plot mass evolution for first subhalo
node_id = list(ts_data.keys())[0]
data, z = track_subhalo(
    ts_data, zsnaps, node_id,
    [ParamKeys.mass_bound]
)

import matplotlib.pyplot as plt
plt.plot(z, data[ParamKeys.mass_bound])
plt.xlabel('Redshift')
plt.ylabel('Bound Mass [M_☉]')
plt.show()

gout.close()
```

### 4. Custom Analysis Function

```python
import numpy as np
from subscript.wrappers import gscript
from subscript.scripts.nfilters import subhalos, interval, logical_and
from subscript.defaults import ParamKeys
from subscript.scripts.spatial import project3d

@gscript
def subhalo_statistics(gout, rmax=0.1, mass_min=1e9, mass_max=1e12, **kwargs):
    """Compute statistics for subhalos in spatial and mass range."""

    # Get masses
    mass = gout[ParamKeys.mass_bound]

    # Get 3D radii
    r3d = project3d(gout)

    # Apply spatial cut
    inside = r3d < rmax

    # Filter data
    m_filt = mass[inside]

    # Count in mass range
    in_range = (m_filt > mass_min) & (m_filt < mass_max)
    count = np.sum(in_range)

    return {
        'count': count,
        'mean_mass': np.mean(m_filt[in_range]) if count > 0 else 0,
        'mean_radius': np.mean(r3d[inside])
    }

# Use the function
result = subhalo_statistics(
    gout,
    rmax=0.1,
    nfilter=subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std]
)
```

---

## Units & Conventions

- **Default coordinates**: physical (not comoving)
- **Distance**: Megaparsecs (Mpc) unless otherwise noted
- **Mass**: Solar masses (M_☉)
- **Time**: Gigayears (Gyr) or Billions of years

For cosmological conversions, use `colossus`:
```python
from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology("planck18")

# Convert kpc to arcsec at redshift z
rad_to_arcsec = 2.06E5
kpc_to_arcsec = (1E-3 / cosmo.angularDiameterDistance(z)) * rad_to_arcsec
r_arcsec = r_kpc * kpc_to_arcsec
```

---

## Troubleshooting

### Common Issues

**Q: "IndexError" when accessing filter results**
A: Filters return lists when operating on h5py.File. Access per-tree:
```python
mass = nodedata(gout, key=ParamKeys.mass_basic)[0]  # First tree
```

**Q: "nodeOperator indexShift" requirement for tracking**
A: tracking.py requires Galacticus to output node indices consistently across snapshots. Configure:
```xml
<nodeOperator value="indexShift" />
```

**Q: Deprecation warnings for old filter names**
A: Use new filter names (e.g., `subhalos()` instead of `nfilter_subhalos()`)

```

---

## Dependencies

```
numpy >= 1.0
pandas
scipy
h5py
scikit-learn
astropy
```

---

## References

### Comprehensive Function Reference

📄 **`references/subscript_functions.md`**

Complete documentation of all SubScript functions organized by module:
- **Function signatures** with full parameter descriptions
- **Return types** and data structures
- **Usage examples** for each function
- **Implementation notes** and performance considerations
- **Complete parameter key reference** (50+ available properties)
- **Summary table** of all functions by category

Use this document for detailed function lookup and integration into custom analysis workflows.

### Other Resources

- **Public Repository**: https://github.com/cgannonucm/SubScript
- **Galacticus**: https://github.com/galacticusorg/galacticus
- **Example Notebooks**: See `example-notebooks/` in the repository
- **Test Suite**: See `tests/` for usage examples
