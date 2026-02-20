# SubScript

The `subscript` python package provides a library of ergonomic utility functions for analyzing [Galacticus](https://github.com/galacticusorg/galacticus) subhalo data. Example notebooks can be found in the `example-notebooks/` directory.

## Installation

### Install via pip
```
pip install subhaloscript
```

### Install via conda
```
conda install cgannonucm::subhaloscript
```

## Features

### Statistics Across Multiple Trees

Galacticus outputs contain multiple independent merger trees. SubScript automatically separates nodes into their respective trees and provides built-in statistical summarization across all of them. Pass `summarize=True` and `statfuncs=[np.mean, np.std]` to any analysis function to get the mean and standard deviation computed over all trees in a single call.

```python
from subscript.scripts.histograms import massfunction
import subscript.scripts.nfilters as nf

out = massfunction(
    gout,
    bins=np.logspace(9, 13, 30),
    nfilter=nf.subhalos,
    summarize=True,
    statfuncs=[np.mean, np.std],
)
dndm_mean, mbins = out[0]
dndm_std, _      = out[1]
```

### Node Filtering

Select subsets of nodes using composable filter functions. Filters can be combined with boolean logic and "frozen" for reuse.

```python
import subscript.scripts.nfilters as nf

# Built-in filters
mass = nodedata(gout, key='basicMass', nfilter=nf.subhalos)
mass = nodedata(gout, key='basicMass', nfilter=nf.hosthalos)

# Spatial filters
inner = nf.r3d(None, 0, 0.05)  # Within 50 kpc (in Mpc)

# Combine with boolean logic
inner_subhalos = nf.logical_and(nf.subhalos, inner)
mass = nodedata(gout, key='basicMass', nfilter=inner_subhalos)
```

### Write Once, Reuse Everywhere

The `@gscript` decorator wraps any analysis function to automatically handle input formatting, node filtering, multi-tree iteration, and statistical summarization. Write your analysis logic once and reuse it across different files, filters, and statistical configurations.

```python
from subscript.wrappers import gscript
from subscript.defaults import ParamKeys

@gscript
def mass_ratio(gout, **kwargs):
    return np.mean(gout[ParamKeys.mass_bound] / gout[ParamKeys.mass_basic])

# Reuse with different filters, files, and statistics
result = mass_ratio(gout, nfilter=nf.subhalos, summarize=True, statfuncs=[np.mean, np.std])
```

### Subhalo Tracking Across Snapshots

Track individual subhalos across all Galacticus output snapshots to study their evolution over time. The `subhalo_timeseries()` function extracts and caches per-subhalo time-series data for an entire tree.

```python
from subscript.subhalo_timeseries import subhalo_timeseries

result = subhalo_timeseries(gout, tree_index=0)

for node_id, ts in result.items():
    plt.plot(ts['zsnaps'], ts['data'][ParamKeys.mass_bound])
```

### Astropy Unit Integration

Optional [astropy](https://www.astropy.org/) unit integration. When enabled, all node properties are returned as astropy `Quantity` objects with physical units, decomposed into a configurable base unit system (default: `[Msun, Mpc, Myr]`).

```python
from subscript.units import enableUnitsFromGalacticus

enableUnitsFromGalacticus(gout)

mass = nodedata(gout, key=ParamKeys.mass_basic, nfilter=nf.hosthalos)
# mass is now an astropy Quantity in solar masses

rvir = nodedata(gout, key=ParamKeys.rvir, nfilter=nf.hosthalos)
print(rvir[0].to(apu.kpc))  # Convert Mpc to kpc
```

See `example-notebooks/units.ipynb` for a full walkthrough.

### Claude Code Skill

This repository includes a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill that gives Claude detailed knowledge of the SubScript library. To use it in your project, copy the skill directory into your project's `.claude/skills/` folder:

```bash
# From your project root
mkdir -p .claude/skills
cp -r /path/to/SubScript/.claude/skills/galacticus-analysis .claude/skills/
```

When using Claude Code in your project, invoke the skill with:

```
/galacticus-analysis
```

Claude will then have access to the full SubScript API reference, including all function signatures, parameter keys, filter patterns, and usage examples, allowing it to write correct SubScript analysis code for your Galacticus data.

## Publication

If you use SubScript in your research, please cite:

> Gannon et al. (2025). *"Dark Matter Substructure: A Lensing Perspective"*
> [arXiv:2501.17362](https://arxiv.org/abs/2501.17362)

