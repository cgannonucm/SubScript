import h5py
import numpy as np
from numpy import testing
from astropy import units as apu

from subscript.defaults import ParamKeys, Meta
from subscript.tabulatehdf5 import tabulate_trees
from subscript.scripts.nodes import nodedata, nodecount
from subscript.scripts.nfilters import hosthalos, subhalos
from subscript.scripts.histograms import massfunction
from subscript.wrappers import freeze
from subscript.macros import macro_run
from subscript.units import (
    galacticus_units_si,
    enableUnitsFromGalacticus,
    UNITS_IN_SI,
)

PATH_DMO = "tests/data/test.hdf5"


def _disable_units():
    """Reset the global Meta state to disable units."""
    Meta.units_enable = False
    Meta.units_in_si = None
    Meta.units_in_si_conversion = None
    Meta.unit_bases = [apu.Msun, apu.Mpc, apu.Myr]


def _enable_units(gout):
    enableUnitsFromGalacticus(gout)


# ---------------------------------------------------------------------------
# galacticus_units_si and enableUnitsFromGalacticus
# ---------------------------------------------------------------------------

def test_galacticus_units_si():
    """galacticus_units_si returns dict with correct conversion factors."""
    gout = h5py.File(PATH_DMO)
    units_si = galacticus_units_si(gout)
    gout.close()

    assert isinstance(units_si, dict)
    assert ParamKeys.mass_basic in units_si
    # Mass conversion: kg per solar mass
    testing.assert_allclose(float(units_si[ParamKeys.mass_basic]), 1.98892e+30, rtol=1e-2)
    # Distance conversion: m per Mpc
    testing.assert_allclose(float(units_si[ParamKeys.x]), 3.08567758135e+22, rtol=1e-2)

    _disable_units()


def test_enable_units_sets_meta():
    """enableUnitsFromGalacticus sets all Meta attributes correctly."""
    _disable_units()
    gout = h5py.File(PATH_DMO)
    _enable_units(gout)
    gout.close()

    assert Meta.units_enable is True
    assert Meta.units_in_si is not None
    assert Meta.units_in_si_conversion is not None
    assert Meta.unit_bases == [apu.Msun, apu.Mpc, apu.Myr]
    _disable_units()


# ---------------------------------------------------------------------------
# UNITS_IN_SI coverage and correctness
# ---------------------------------------------------------------------------

def test_units_in_si_types():
    """UNITS_IN_SI maps mass->kg, distance->m, velocity->m/s, time->s, dimensionless."""
    for k in [ParamKeys.mass_basic, ParamKeys.mass_bound]:
        assert UNITS_IN_SI[k].is_equivalent(apu.kg)
    for k in [ParamKeys.x, ParamKeys.y, ParamKeys.z, ParamKeys.rvir]:
        assert UNITS_IN_SI[k].is_equivalent(apu.m)
    for k in [ParamKeys.satellite_velocity_x]:
        assert UNITS_IN_SI[k].is_equivalent(apu.m / apu.s)
    assert UNITS_IN_SI[ParamKeys.basic_time_last_isolated].is_equivalent(apu.s)
    for k in [ParamKeys.is_isolated, ParamKeys.z_lastisolated, ParamKeys.concentration]:
        assert UNITS_IN_SI[k].is_equivalent(apu.dimensionless_unscaled)


# ---------------------------------------------------------------------------
# nodedata returns Quantity with correct unit types
# ---------------------------------------------------------------------------

def test_nodedata_units_mass_distance_time():
    """nodedata returns Quantities with correct physical types (mass, distance, time)."""
    _disable_units()
    gout = h5py.File(PATH_DMO)
    _enable_units(gout)

    mass = nodedata(gout, key=ParamKeys.mass_basic, nfilter=hosthalos)

    print(mass)
    rvir = nodedata(gout, key=ParamKeys.rvir, nfilter=hosthalos)
    time = nodedata(gout, key=ParamKeys.basic_time_last_isolated, nfilter=hosthalos)

    gout.close()
    _disable_units()

    assert isinstance(mass, apu.Quantity)
    assert mass.unit.is_equivalent(apu.Msun)
    assert isinstance(rvir, apu.Quantity)
    assert rvir.unit.is_equivalent(apu.Mpc)
    assert isinstance(time, apu.Quantity)
    assert time.unit.is_equivalent(apu.Myr)


# ---------------------------------------------------------------------------
# Numerical values match with and without units
# ---------------------------------------------------------------------------

def test_nodedata_values_match():
    """Numerical values of nodedata should be the same with and without units."""
    gout = h5py.File(PATH_DMO)

    _disable_units()
    mass_no = nodedata(gout, key=ParamKeys.mass_basic, nfilter=hosthalos)
    rvir_no = nodedata(gout, key=ParamKeys.rvir, nfilter=hosthalos)

    _enable_units(gout)
    mass_yes = nodedata(gout, key=ParamKeys.mass_basic, nfilter=hosthalos)
    rvir_yes = nodedata(gout, key=ParamKeys.rvir, nfilter=hosthalos)

    gout.close()
    _disable_units()

    testing.assert_allclose(mass_yes.value, mass_no, rtol=1e-2)
    testing.assert_allclose(rvir_yes.value, rvir_no, rtol=1e-2)


def test_nodedata_subhalos_values_match():
    """Numerical values match for subhalos across all trees."""
    gout = h5py.File(PATH_DMO)

    _disable_units()
    mass_no = nodedata(gout, key=ParamKeys.mass_bound, nfilter=subhalos)

    _enable_units(gout)
    mass_yes = nodedata(gout, key=ParamKeys.mass_bound, nfilter=subhalos)

    gout.close()
    _disable_units()

    for m_no, m_yes in zip(mass_no, mass_yes):
        testing.assert_allclose(m_yes.value, m_no, rtol=1e-2)


def test_nodedata_summarize_with_units():
    """summarize + statfuncs should work with units enabled and match values."""
    gout = h5py.File(PATH_DMO)

    _disable_units()
    result_no = nodedata(gout, key=ParamKeys.mass_basic, nfilter=hosthalos,
                         summarize=True, statfuncs=[np.mean, np.std])

    _enable_units(gout)
    result_yes = nodedata(gout, key=ParamKeys.mass_basic, nfilter=hosthalos,
                          summarize=True, statfuncs=[np.mean, np.std])

    gout.close()
    _disable_units()

    print(result_yes[0])
    assert isinstance(result_yes[0], apu.Quantity)
    testing.assert_allclose(result_yes[0].value, np.asarray(result_no[0]).flatten(), rtol=1e-2)
    testing.assert_allclose(result_yes[1].value, np.asarray(result_no[1]).flatten(), rtol=1e-2)


# ---------------------------------------------------------------------------
# Mass function with units
# ---------------------------------------------------------------------------

def test_massfunction_values_match():
    """Mass function numerical results should match with and without units."""
    gout = h5py.File(PATH_DMO)
    bins = np.logspace(10, 11, 5)

    _disable_units()
    trees = tabulate_trees(gout)
    mf_no, bins_no = massfunction(trees[0], key_mass=ParamKeys.mass_basic, bins=bins)

    _enable_units(gout)
    trees_u = tabulate_trees(gout)
    mf_yes, bins_yes = massfunction(trees_u[0], key_mass=ParamKeys.mass_basic, bins=bins * apu.Msun)

    gout.close()
    _disable_units()

    mf_yes_val = mf_yes.value if isinstance(mf_yes, apu.Quantity) else mf_yes
    testing.assert_allclose(mf_yes_val, mf_no, rtol=1e-10)


def test_massfunction_summarize_with_units():
    """Mass function mean and std across trees should match with and without units."""
    gout = h5py.File(PATH_DMO)
    bins = np.logspace(10, 11, 5)

    _disable_units()
    result_no = massfunction(
        gout, key_mass=ParamKeys.mass_basic, bins=bins,
        summarize=True, statfuncs=[np.mean, np.std],
    )

    _enable_units(gout)
    result_yes = massfunction(
        gout, key_mass=ParamKeys.mass_basic, bins=bins * apu.Msun,
        summarize=True, statfuncs=[np.mean, np.std],
    )

    gout.close()
    _disable_units()

    # result is [[mean_mf, mean_bins], [std_mf, std_bins]]
    for i in range(2):  # mean, std
        for j in range(2):  # mf values, bin edges
            val_no = np.asarray(result_no[i][j])
            val_yes = result_yes[i][j]
            val_yes_num = val_yes.value if isinstance(val_yes, apu.Quantity) else np.asarray(val_yes)
            testing.assert_allclose(val_yes_num, val_no, rtol=1e-2)
    _disable_units()


# ---------------------------------------------------------------------------
# nodecount (should be unaffected by units)
# ---------------------------------------------------------------------------

def test_nodecount_with_units():
    """nodecount should return same values with units enabled."""
    gout = h5py.File(PATH_DMO)

    _disable_units()
    count_no = nodecount(gout, nfilter=subhalos, summarize=True)

    _enable_units(gout)
    count_yes = nodecount(gout, nfilter=subhalos, summarize=True)

    gout.close()
    _disable_units()

    val_yes = count_yes.value if isinstance(count_yes, apu.Quantity) else count_yes
    testing.assert_allclose(np.asarray(val_yes).flatten(),
                            np.asarray(count_no).flatten(), rtol=1e-10)

    _disable_units()


# ---------------------------------------------------------------------------
# Macros with units
# ---------------------------------------------------------------------------

def test_macro_run_with_units():
    """macro_run should work with units enabled and produce matching values."""
    gout = h5py.File(PATH_DMO)
    gout2 = h5py.File("tests/data/test-copy.hdf5")

    macros = {
        "haloMass": freeze(nodedata, key=ParamKeys.mass_basic, nfilter=hosthalos),
        "z": freeze(nodedata, key=ParamKeys.z_lastisolated, nfilter=hosthalos),
    }

    _disable_units()
    out_no = macro_run(macros, [gout, gout2], statfuncs=[np.mean, np.std])

    _enable_units(gout)
    out_yes = macro_run(macros, [gout, gout2], statfuncs=[np.mean, np.std])

    gout.close()
    gout2.close()

    for key in out_no:
        if key not in ['haloMass (mean)', 'haloMass (std)', 'z (mean)', 'z (std)']:
            continue
            
        for subkey in out_no[key]:
            val_no = out_no[key][subkey]
            val_yes = out_yes[key][subkey]          
            testing.assert_allclose(np.asarray(val_yes), np.asarray(val_no), rtol=1e-2)
    _disable_units()

def test_finalize():
    _disable_units()

if __name__ == "__main__":
    test_macro_run_with_units()

