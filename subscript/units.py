"""
Astropy unit integration for Galacticus node properties.

This module provides functions to configure optional astropy unit support for
the SubScript library.  When units are enabled, every node property returned
by :class:`~subscript.tabulatehdf5.NodeProperties` is an
:class:`astropy.units.Quantity` decomposed in the configured base-unit system
(default: ``[M☉, Mpc, Myr]``).

Typical usage
-------------
Enable units from a Galacticus HDF5 file (reads ``unitsInSI`` attributes)::

    from subscript.units import enableUnitsFromGalacticus
    enableUnitsFromGalacticus(gal_out)

Or supply conversion factors manually::

    from subscript.units import enableUnits, UNITS_IN_SI
    from subscript.tabulatehdf5 import galacticus_units_si
    enableUnits(galacticus_units_si(gal_out), UNITS_IN_SI)

Public API
----------
:data:`UNITS_IN_SI`
    Dictionary mapping :class:`~subscript.defaults.ParamKeys` string keys to
    their SI :mod:`astropy.units` unit objects.

:func:`galacticus_units_si`
    Extract per-dataset SI conversion factors from a Galacticus HDF5 file.

:func:`enableUnits`
    Configure the global unit state directly.

:func:`enableUnitsFromGalacticus`
    Convenience wrapper that reads conversion factors from an HDF5 file then
    calls :func:`enableUnits`.
"""
import csv

import h5py
from subscript.tabulatehdf5 import get_galacticus_outputs
from subscript.defaults import Meta, ParamKeys
from astropy import units as apu    

def galacticus_units_si(gal_out):
    """Extract SI units from a Galacticus HDF5 output file.

    Reads the unitsInSI attribute for all datasets in the nodeData group
    of the final output and optionally writes them to a CSV file.

    Parameters
    ----------
    gal_out : h5py.File
        Open Galacticus HDF5 file.
    output_path : str, optional
        Path to the output CSV file. If None, no file is written.

    Returns
    -------
    dict
        Dictionary mapping node property names to their SI unit values.
    """
    outputs = get_galacticus_outputs(gal_out)
    output_final = outputs[-1]

    node_data = gal_out[f"Outputs/Output{output_final}/nodeData"]

    units_si = {}
            
    for node_property_name in node_data:
        dataset = node_data[node_property_name]
        units_in_si = dataset.attrs.get("unitsInSI", 1.0)
        units_si[node_property_name] = float(units_in_si)
    
    return units_si


UNITS_IN_SI = {
    ParamKeys.x: apu.m,
    ParamKeys.y: apu.m,
    ParamKeys.z: apu.m,
    ParamKeys.relx: apu.m,
    ParamKeys.rely: apu.m,
    ParamKeys.relz: apu.m,
    ParamKeys.mass: apu.kg,
    ParamKeys.mass_bound: apu.kg,
    ParamKeys.rvir: apu.m,
    ParamKeys.mass_basic: apu.kg,
    ParamKeys.is_isolated: apu.dimensionless_unscaled,
    ParamKeys.hierarchylevel: apu.dimensionless_unscaled,
    ParamKeys.sphere_radius: apu.m,
    ParamKeys.sphere_anglularmomentum: apu.kg * apu.m**2 / apu.s,
    ParamKeys.sphere_mass_stellar: apu.kg,
    ParamKeys.sphere_mass_gas: apu.kg,
    ParamKeys.scale_radius: apu.m,
    ParamKeys.density_profile_radius: apu.m,
    ParamKeys.density_profile: apu.kg / apu.m**3,
    ParamKeys.z_lastisolated: apu.dimensionless_unscaled,
    ParamKeys.tnfw_rt: apu.m,
    ParamKeys.tnfw_p0: apu.kg / apu.m**3,
    ParamKeys.custom_id: apu.dimensionless_unscaled,
    ParamKeys.custom_tree_index: apu.dimensionless_unscaled,
    ParamKeys.custom_tree_outputorder: apu.dimensionless_unscaled,
    ParamKeys.concentration: apu.dimensionless_unscaled,
    ParamKeys.merger_tree_count: apu.dimensionless_unscaled,
    ParamKeys.merger_tree_index: apu.dimensionless_unscaled,
    ParamKeys.merger_tree_seed: apu.dimensionless_unscaled,
    ParamKeys.merger_tree_start_index: apu.dimensionless_unscaled,
    ParamKeys.merger_tree_weight: apu.dimensionless_unscaled,
    ParamKeys.basic_time_last_isolated: apu.s,
    ParamKeys.dark_matter_temperature_virial: apu.K,
    ParamKeys.dark_matter_velocity_virial: apu.m / apu.s,
    ParamKeys.dark_matter_profile_dmo_radius_velocity_max: apu.m,
    ParamKeys.dark_matter_profile_dmo_velocity_max: apu.m / apu.s,
    ParamKeys.dark_matter_profile_scale: apu.m,
    ParamKeys.mass_halo_enclosed_current: apu.kg,
    ParamKeys.hierarchy_level_depth: apu.dimensionless_unscaled,
    ParamKeys.hierarchy_level_maximum: apu.dimensionless_unscaled,
    ParamKeys.node_index: apu.dimensionless_unscaled,
    ParamKeys.subsampling_weight: apu.dimensionless_unscaled,
    ParamKeys.parent_index: apu.dimensionless_unscaled,
    ParamKeys.satellite_index: apu.dimensionless_unscaled,
    ParamKeys.satellite_velocity_x: apu.m / apu.s,
    ParamKeys.satellite_velocity_y: apu.m / apu.s,
    ParamKeys.satellite_velocity_z: apu.m / apu.s,
    ParamKeys.satellite_tidal_field: 1 / apu.s**2,
    ParamKeys.satellite_tidal_heating_normalized: apu.kg / apu.s**2,
    ParamKeys.sibling_index: apu.dimensionless_unscaled,
    ParamKeys.spin_angular_momentum: apu.dimensionless_unscaled,
    ParamKeys.spin_angular_momentum_vector_x: apu.dimensionless_unscaled,
    ParamKeys.spin_angular_momentum_vector_y: apu.dimensionless_unscaled,
    ParamKeys.spin_angular_momentum_vector_z: apu.dimensionless_unscaled,
}
"""Dictionary mapping Galacticus node property names to their corresponding SI units using astropy.units."""

def enableUnitsFromGalacticus(galacticus_output:h5py.File, units_in_si:dict=UNITS_IN_SI, base_units:list=Meta.unit_bases):
    """
    Enable astropy units globally for Galacticus node properties.

    This function extracts unit information from a Galacticus HDF5 output file and
    enables the corresponding astropy units for use throughout the application.

    Parameters
    ----------
    galacticus_output : h5py.File
        An open HDF5 file object containing Galacticus simulation output data.
    galacticus_output : h5py.File
        An open HDF5 file object containing Galacticus simulation output data.
    units_in_si : dict, optional
        A dictionary mapping unit names to their SI equivalents. 
        Defaults to UNITS_IN_SI.
    base_units : list, optional
        A list of base unit names to use for the unit system.
        Defaults to Meta.unit_bases.

    Returns
    -------
    None

    Notes
    -----
    This function modifies the global unit state by calling enableUnits()
    with units extracted from the Galacticus file combined with the provided
    unit definitions.

    See Also
    --------
    galacticus_units_si : Extract units in SI from Galacticus output
    enableUnits : Enable units globally with given definitions
    """
    """Globally enable astropy units for Galacticus node properties"""    
    units_in_si_from_galacticus = galacticus_units_si(galacticus_output)
    enableUnits(units_in_si_from_galacticus, units_in_si, base_units)


def enableUnits(units_in_si_conversion:dict, units_in_si:dict=UNITS_IN_SI, base_units:list=Meta.unit_bases):
    """
    This function configures the unit system for the entire application by setting
    global metadata that controls how node properties are converted to and from SI units.

    Parameters
    ----------
    units_in_si_conversion : dict
        A mapping of nodeData keys to numerical conversion factors that transform
        the corresponding nodeProperty values to SI units.
    units_in_si : dict, optional
        A dictionary defining the SI unit representations for node properties.
        Defaults to UNITS_IN_SI.
    base_units : list, optional
        A list of base unit identifiers for the unit system.
        Defaults to Meta.unit_bases.

    Returns
    -------
    None
        This function modifies global unit configuration and does not return a value.

    Notes
    -----
    This function sets the following global Meta attributes:
    - units_enable: Boolean flag indicating units are active
    - units_in_si: SI unit definitions
    - unit_bases: Base unit list
    - units_in_si_conversion: Unit conversion factors

    See Also
    --------
    Meta : Global metadata configuration object
    """

    Meta.units_enable = True
    Meta.units_in_si = units_in_si
    Meta.unit_bases = base_units
    Meta.units_in_si_conversion = units_in_si_conversion

