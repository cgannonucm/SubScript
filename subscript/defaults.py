class Meta():
    cache = True
    """If true, use custom code when reading hdf5 to cache read data."""
    disableDepreciatedWarning = False

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
