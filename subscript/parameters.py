class NodePropertyLabel():
    #Names of galacticus parameters
    x = "positionOrbitalX"
    """The GALACTICUS output parameter for the X coordinate of the subhalo relative to the main halo"""
    y = "positionOrbitalY" 
    """The GALACTICUS output parameter for the Y coordinate of the subhalo relative to the main halo"""
    z = "positionOrbitalZ"
    """The GALACTICUS output parameter for the Z coordinate of the subhalo relative to the main halo"""
    
    #Position of the subhalo relative to it's host halo or subhalo (not top level)
    key_relx = "satellitePositionX"
    """The GALACTICUS output parameter for the X coordinate of the subhalo relative to the halo / subhalo that hosts it. 
    FOR SUB-SUBHALOS THIS IS NOT THE MAIN HALO"""
    key_rely = "satellitePositionY"
    """The GALACTICUS output parameter for the Y coordinate of the subhalo relative to the halo / subhalo that hosts it. 
    FOR SUB-SUBHALOS THIS IS NOT THE MAIN HALO"""
    key_relz = "satellitePositionZ"
    """The GALACTICUS output parameter for the Z coordinate of the subhalo relative to the halo / subhalo that hosts it. 
    FOR SUB-SUBHALOS THIS IS NOT THE MAIN HALO""" 

    mass_bound = "satelliteBoundMass"
    """The GALACTICUS output parameter for the gravitationally bound mass contained within a subhalo"""
    mass_basic = "basicMass"
    """The GALACTICUS output parameter for the mass at acrettion. Includes mass from substructure."""
    is_isolated = "nodeIsIsolated"
    """The GALACTICUS output parameter describing if the node is a halo / subhalo. 0 if subhalo 1 if halo"""
    hierarchylevel = "nodeHierarchyLevel"
    """The GALACTICUS output parameter describing the level of substructure the current halo exists at. For the "main" halo
    this would be 0, for subhstrucure 1, for subs-substructure 2,..."""
    rvir = 'darkMatterOnlyRadiusVirial'
    mass = "basicMass"
    
    sphere_radius = "spheroidRadius"
    """The GALACTICUS output parameter describing the sphereoid radius of a spheroid galaxy"""
    sphere_anglularmomentum = "spheroidAngularMomentum"
    """The GALACTICUS output parameter describing the angular momentum of a spheroid galaxy"""
    sphere_mass_stellar = "spheroidMassStellar"
    """The GALACTICUS output parameter describing the stellar mass of a spheroid galaxy"""
    sphere_mass_gas = "spheroidMassGas"
    """The GALACTICUS output parameter describing the gas mass of a spheroid galaxy"""
    
    scale_radius = "darkMatterProfileScaleRadius"
    """The GALACTICUS output parameter describing the scale radius of the halo / subhalo"""
    density_profile_radius = "densityProfileRadius"
    """The GALACTICUS output parameter describing the density profile radii of the halo / subhalo"""
    density_profile = "densityProfile"
    
    z_lastisolated = "redshiftLastIsolated"
    
    custom_id = "custom_id"
    custom_tree_index = "custom_node_tree"
    custom_tree_outputorder = "custom_node_outputorder"