from astropy import units as apu    

class ParamKeys():
    """Library of default galacticus parameters."""
    x = 'positionOrbitalX'
    y = 'positionOrbitalY'
    z = 'positionOrbitalZ'
    relx = 'satellitePositionX'
    rely = 'satellitePositionY'
    relz = 'satellitePositionZ'
    mass = 'basicMass'
    mass_bound = 'satelliteBoundMass'
    rvir = 'darkMatterOnlyRadiusVirial'
    rscale = 'darkMatterProfileScaleRadius'
    mass_basic = 'basicMass'
    is_isolated = 'nodeIsIsolated'
    hierarchylevel = 'nodeHierarchyLevel'
    sphere_radius = 'spheroidRadius'
    sphere_anglularmomentum = 'spheroidAngularMomentum'
    sphere_mass_stellar = 'spheroidMassStellar'
    sphere_mass_gas = 'spheroidMassGas'
    scale_radius = 'darkMatterProfileScaleRadius'
    density_profile_radius = 'densityProfileRadius'
    density_profile = 'densityProfile'
    z_lastisolated = 'redshiftLastIsolated'
    tnfw_rt = 'radiusTidalTruncationNFW'
    tnfw_p0 = 'densityNormalizationTidalTruncationNFW'
    custom_id = 'custom_id'
    custom_tree_index = 'custom_node_tree'
    custom_tree_outputorder = 'custom_node_outputorder'
    concentration = 'concentration'
    merger_tree_count = 'mergerTreeCount'
    merger_tree_index = 'mergerTreeIndex'
    merger_tree_seed = 'mergerTreeSeed'
    merger_tree_start_index = 'mergerTreeStartIndex'
    merger_tree_weight = 'mergerTreeWeight'
    basic_time_last_isolated = 'basicTimeLastIsolated'
    dark_matter_temperature_virial = 'darkMatterOnlyTemperatureVirial'
    dark_matter_velocity_virial = 'darkMatterOnlyVelocityVirial'
    dark_matter_profile_dmo_radius_velocity_max = 'darkMatterProfileDMORadiusVelocityMaximum'
    dark_matter_profile_dmo_velocity_max = 'darkMatterProfileDMOVelocityMaximum'
    dark_matter_profile_scale = 'darkMatterProfileScale'
    mass_halo_enclosed_current = 'massHaloEnclosedCurrent'
    hierarchy_level_depth = 'nodeHierarchyLevelDepth'
    hierarchy_level_maximum = 'nodeHierarchyLevelMaximum'
    node_index = 'nodeIndex'
    subsampling_weight = 'nodeSubsamplingWeight'
    parent_index = 'parentIndex'
    satellite_index = 'satelliteIndex'
    satellite_velocity_x = 'satelliteVelocityX'
    satellite_velocity_y = 'satelliteVelocityY'
    satellite_velocity_z = 'satelliteVelocityZ'
    satellite_tidal_field = 'satelliteTidalField'
    satellite_tidal_heating_normalized = 'satelliteTidalHeatingNormalized'
    sibling_index = 'siblingIndex'
    spin_angular_momentum = 'spinAngularMomentum'
    spin_angular_momentum_vector_x = 'spinAngularMomentumVectorX'
    spin_angular_momentum_vector_y = 'spinAngularMomentumVectorY'
    spin_angular_momentum_vector_z = 'spinAngularMomentumVectorZ'
    
class Meta(): 
    disableDepreciatedWarning = False
    """If true, disable the warning about depreciated code."""

    units_enable = False
    """If true, use astropy units when reading data."""

    units_in_si = None
    """Dictionary mapping parameter keys to their corresponding astropy units in SI."""   

    units_in_si_conversion = None
    """Dictionary mapping parameter keys to a numerical value that converts the nodeProperty to an SI unit."""

    unit_bases = [apu.Msun, apu.Mpc, apu.Myr]
    """List of astropy units to use as bases for unit conversions."""

    disable_auto_format_scalar = False
    """
    If a function only outputs a single single value in an array per tree, 
    Ie [[1.0], [1.0], [1.0]], then automatically convert to a 1d array, Ie [1.0, 1.0, 1.0].
    If true, disable this automatic formatting. 
    WARNING: This may be set to true in the future, but is currently not recommended to set to true as it may cause unexpected output formats.
    For new code use disable_auto_format_scalar = True and explicitly format the output as needed.
    """

