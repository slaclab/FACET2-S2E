"""Set the lattice in FACET-II control-system language and units.

``setLattice`` is the all-in-one function for lattice changes; the get/set
helpers translate between EPICS units (kG, GeV/c) and Bmad.

Moved from UTILITY_setLattice.py; the chicane-sextupole offset setters come
from simulationFunctions.py.
"""

import yaml

from .linac import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto


#This is the all-in-one function that should make any and all changes to the lattice
#This way there's a single place where default values need to be kept
def setLattice(
    tao,
    defaultsFile=None,
    verbose = False,
    **overrides
):
    """
    Configure the accelerator lattice by setting all relevant element strengths and parameters.

    This is the all-in-one function for making changes to the lattice, ensuring a single location for default values and overrides.
    Loads default settings from a YAML file, applies any overrides, and sets all linac, quad, bend, sextupole, kicker, and XTCAV parameters accordingly.

    Parameters
    ----------
    tao : Tao
        Active Tao simulation object.
    defaultsFile : str, optional
        Path to the YAML file containing default lattice settings. 
    verbose : bool, default False
        If True, prints information about the defaults file used.
    **overrides : dict
        Keyword arguments to override any default lattice settings.

    Returns
    -------
    None

    """

    if not defaultsFile:
        defaultsFile = f'{tao.filePathGlobal}/setLattice_configs/defaults.yml'
        if verbose:
            print(f"No defaults file provided to setLattice(). Using {defaultsFile}")
        
    with open(defaultsFile, 'r') as file:
        defaults = yaml.safe_load(file)
        
    # Combine defaults and overrides, with overrides taking precedence
    latticeSettings = {**defaults, **overrides}


    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = F")
    
    setLinacsHelper(
        tao,
        latticeSettings["L0BPhaseSet"], 
        latticeSettings["L0BEnergyOffset"], 
        latticeSettings["L1PhaseSet"], 
        latticeSettings["L1EnergyOffset"], 
        latticeSettings["L2PhaseSet"], 
        latticeSettings["L2EnergyOffset"], 
        latticeSettings["L3PhaseSet"], 
        latticeSettings["L3EnergyOffset"]
    )


    #2025-06-18 update: Reenable lattice calculations to let Bmad auto-update magnets for new energy. Then immediately disable again
    #This prevents an ambigious condition where Bmad may overwrite the setLattice() magnet values
    tao.cmd("set global lattice_calc_on = T")
    tao.cmd("set global lattice_calc_on = F")

    setAllInjectorQuads(tao, 
                        latticeSettings["QA10361kG"], 
                        latticeSettings["QA10371kG"], 
                        latticeSettings["QE10425kG"], 
                        latticeSettings["QE10441kG"], 
                        latticeSettings["QE10511kG"], 
                        latticeSettings["QE10525kG"]
                       )
    
    setAllFinalFocusQuads(tao, 
                          latticeSettings["Q5FFkG"], 
                          latticeSettings["Q4FFkG"], 
                          latticeSettings["Q3FFkG"], 
                          latticeSettings["Q2FFkG"], 
                          latticeSettings["Q1FFkG"], 
                          latticeSettings["Q0FFkG"], 
                          latticeSettings["Q0DkG"], 
                          latticeSettings["Q1DkG"], 
                          latticeSettings["Q2DkG"]
                         )

    setWChicaneLaunchQuads(tao, 
                           latticeSettings["Q19851kG"], 
                           latticeSettings["Q19871kG"]
                          )
    
    setAllWChicaneBends(tao, 
                        latticeSettings["B1EkG"], 
                        latticeSettings["B2EkG"], 
                        latticeSettings["B3EkG"]
                       )
    
    setAllWChicaneQuads(tao, 
                        latticeSettings["Q1EkG"], 
                        latticeSettings["Q2EkG"], 
                        latticeSettings["Q3EkG"], 
                        latticeSettings["Q4EkG"], 
                        latticeSettings["Q5EkG"], 
                        latticeSettings["Q6EkG"]
                       )

    if latticeSettings["symmetricSextupoleStrengths"]:
        #Everything set symmetrically to "L" values
        setAllWChicaneSextupoles(tao, 
                                 latticeSettings["S1ELkG"], 
                                 latticeSettings["S2ELkG"], 
                                 latticeSettings["S3ELkG"], 
                                 latticeSettings["S3ELkG"], 
                                 latticeSettings["S2ELkG"], 
                                 latticeSettings["S1ELkG"]
                                )
    else: 
        setAllWChicaneSextupoles(tao, 
                         latticeSettings["S1ELkG"], 
                         latticeSettings["S2ELkG"], 
                         latticeSettings["S3ELkG"], 
                         latticeSettings["S3ERkG"], 
                         latticeSettings["S2ERkG"], 
                         latticeSettings["S1ERkG"]
                        )

    setAllWChicaneMovers(tao,
                         latticeSettings["S1EL_xOffset"], 
                         latticeSettings["S1EL_yOffset"], 
                         latticeSettings["S2EL_xOffset"], 
                         latticeSettings["S2EL_yOffset"], 
                         latticeSettings["S2ER_xOffset"], 
                         latticeSettings["S2ER_yOffset"], 
                         latticeSettings["S1ER_xOffset"], 
                         latticeSettings["S1ER_yOffset"]
                        )

    setXTCAV(tao, 
             latticeSettings["XTCAVvoltage"], 
             latticeSettings["XTCAVphase"]
            )

    
    setAllFinalFocusKickers(tao,
                        latticeSettings["XC1FFkG"],
                        latticeSettings["XC3FFkG"],
                        latticeSettings["YC1FFkG"],
                        latticeSettings["YC2FFkG"]
                        )

    setBendGeVc(tao, "B5D36", latticeSettings["B5D36GeVc"])
    
    
    #Reenable lattice calculations
    tao.cmd("set global lattice_calc_on = T")

    return


def setLinacsHelper(tao, L0BPhaseSet, L0BEnergyOffset, L1PhaseSet, L1EnergyOffset, L2PhaseSet, L2EnergyOffset, L3PhaseSet, L3EnergyOffset):
    """
    Helper function to set linac phases and gradients for all major linac sections.

    Parameters
    ----------
    tao : Tao
        Active Tao simulation object.
    L0BPhaseSet : float
        Phase setting for L0B (degrees).
    L0BEnergyOffset : float
        Energy offset for L0B (eV).
    L1PhaseSet : float
        Phase setting for L1 (degrees).
    L1EnergyOffset : float
        Energy offset for L1 (eV).
    L2PhaseSet : float
        Phase setting for L2 (degrees).
    L2EnergyOffset : float
        Energy offset for L2 (eV).
    L3PhaseSet : float
        Phase setting for L3 (degrees).
    L3EnergyOffset : float
        Energy offset for L3 (eV).

    Returns
    -------
    None


    """
    [L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers] = getLinacMatchStrings(tao)
    
    # === Make changes to base lattice ===
    tao.cmd(f'set ele L0BF PHI0 = {L0BPhaseSet/360.0}') #DNT. "loadNominalValues_2Bunch.m" had this set to zero
    tao.cmd(f'set ele L0BF VOLTAGE = {5.95e7 + L0BEnergyOffset}') #DNT. Base value was 7.1067641E+07, new value (5.95e7) set to change output energy to 125 MeV (down from 136.5 MeV)
    
    #L1AssertPHI0 = -19 #DNT. "loadNominalValues_2Bunch.m" had this set to -19 degrees
    setLinacPhase(        tao, L1MatchStrings, L1PhaseSet ) 
    setLinacGradientAuto( tao, L1MatchStrings, L1EnergyOffset + 0.335e9 - 0.125e9 ) 
    
    #L2AssertPHI0 = -37 #DNT. "loadNominalValues_2Bunch.m" had this set to -37 degrees
    setLinacPhase(        tao, L2MatchStrings, L2PhaseSet ) 
    setLinacGradientAuto( tao, L2MatchStrings, L2EnergyOffset + 4.5e9 - 0.335e9 )
    
    setLinacPhase(        tao, L3MatchStrings, L3PhaseSet )
    setLinacGradientAuto( tao, L3MatchStrings, L3EnergyOffset + 10.0e9 - 4.5e9 )


def setBendkG(tao, bendName, integratedFieldkG):
    """Set bend based on EPICS-style integrated field. This involves a sign flip!"""
    bendLength = tao.ele_gen_attribs(bendName)["L"]

    desiredGradientkG = integratedFieldkG / bendLength

    #Bmad uses Tesla and opposite sign!
    tao.cmd(f"set ele {bendName} B_FIELD = {-1 * desiredGradientkG/10}")

    return


def getBendkG(tao, bendName):
    """Get bend's present EPICS-style integrated field. This involves a sign flip!"""
    bendLength = tao.ele_gen_attribs(bendName)["L"]
    bendGradientTm = tao.ele_gen_attribs(bendName)["B_FIELD"]
    bendGradientkGm = bendGradientTm*10

    bendIntegratedFieldkG = bendGradientkGm * bendLength

    #Bmad uses opposite sign!
    return -1 * bendIntegratedFieldkG


def setQuadkG(tao, quadName, integratedFieldkG):
    """Set quad based on EPICS-style integrated field. This involves a sign flip!"""
    quadLength = tao.ele_gen_attribs(quadName)["L"]

    desiredGradientkG = integratedFieldkG / quadLength

    #Bmad uses Tesla and opposite sign!
    tao.cmd(f"set ele {quadName} B1_GRADIENT = {-1 * desiredGradientkG/10}")

    return


MAGNET_FIELD_KINDS = ("quadrupole", "sextupole", "octupole", "hkicker", "vkicker", "kicker")


def fixMagnetFields(tao, kinds=MAGNET_FIELD_KINDS):
    """Make the magnetic fields (B1_GRADIENT, B2_GRADIENT, BL_KICK, ...) of all elements of `kinds` the independent
    attributes (Bmad field_master = T), so that they stay fixed when the reference energy P0C changes, e.g. in the
    cavity energy feedback, and the normalized strengths (K1, K2, KICK) follow P0C instead.
    By default (field_master = F) Bmad keeps K1 and K2 and scales the fields with P0C: a quad set to its EPICS field
    with setQuadkG is then wrong by P0C_new/P0C_old after any later energy change (1.2% at BC20 after the feedback).
    Setting K1/K2 afterwards still works (the field is recomputed). Bends are not changed (see treat_dipoles)."""
    for kind in kinds:
        if len(tao.lat_list(f"{kind}::*", "ele.name", flags="-no_slaves")):
            tao.cmd(f"set ele {kind}::* field_master = T")
    return tao


def getQuadkG(tao, quadName):
    """Get quad's present EPICS-style integrated field. This involves a sign flip!"""
    quadLength = tao.ele_gen_attribs(quadName)["L"]
    quadGradientTm = tao.ele_gen_attribs(quadName)["B1_GRADIENT"]
    quadGradientkGm = quadGradientTm*10

    quadIntegratedFieldkG = quadGradientkGm * quadLength

    #Bmad uses opposite sign!
    return -1 * quadIntegratedFieldkG


def setSextkG(tao, sextName, integratedFieldkG):
    """Set sextupole based on EPICS-style integrated field. This involves a sign flip!"""
    sextLength = tao.ele_gen_attribs(sextName)["L"]

    desiredGradientkG = integratedFieldkG / sextLength

    #Bmad uses Tesla and opposite sign!
    tao.cmd(f"set ele {sextName} B2_GRADIENT = {-1 * desiredGradientkG/10}")

    return


def getSextkG(tao, sextName):
    """Get sextupoles's present EPICS-style integrated field. This involves a sign flip!"""
    sextLength = tao.ele_gen_attribs(sextName)["L"]
    sextGradientTm = tao.ele_gen_attribs(sextName)["B2_GRADIENT"]
    sextGradientkGm = sextGradientTm*10

    sextIntegratedFieldkG = sextGradientkGm * sextLength

    #Bmad uses opposite sign!
    return -1 * sextIntegratedFieldkG


def setXOffset(tao, eleName, offset):
    tao.cmd(f"set ele {eleName} X_OFFSET = {offset}")

    return


def setYOffset(tao, eleName, offset):
    tao.cmd(f"set ele {eleName} Y_OFFSET = {offset}")

    return


def setKickerkG(tao, kickerName, integratedFieldkG):
    """Set HKICKER or VKICKER based on EPICS-style integrated field"""

    tao.cmd(f"set ele {kickerName} BL_KICK = {integratedFieldkG/10}")

    return


def getKickerkG(tao, kickerName):
    """Return HKICKER or VKICKER EPICS-style integrated field"""

    return tao.ele_gen_attribs(kickerName)["BL_KICK"] * 10 


def setBendGeVc(tao, bendName, desiredBendGeVc):
    """
    Specify a bend's strength using "GeV/c" the way the control system does
    Presently, this only works on special cases where this conversion is specified
    """

    #Final spectrometer dipole, B5D36, is designed to give R46 = 6 mrad at 10 GeV (consistent with golden lattice) 
    #It achieves this at G = 6.1355967E-03 1/m
    if bendName == "B5D36":
        designEnergyGeVc = 10
        designG = 6.1355967E-03
        tao.cmd(f"set ele {bendName} G = { designG * desiredBendGeVc / designEnergyGeVc}")

    else:
        "Not a special case bend"
        return


def getBendGeVc(tao, bendName):
    """
    Based on the bend's setting and the design lattice, report the setting in "GeV/c" the way the control system does
    Presently, this only works on special cases where this conversion is specified
    """

    #Final spectrometer dipole, B5D36, is designed to give R46 = 6 mrad at 10 GeV (consistent with golden lattice) 
    #It achieves this at G = 6.1355967E-03 1/m
    if bendName == "B5D36":
        designEnergyGeVc = 10
        designG = 6.1355967E-03
        return tao.ele_gen_attribs(bendName)["G"] * designEnergyGeVc  / designG

    else:
        "Not a special case bend"
        return


    


def setAllInjectorQuads(tao, QA10361kG, QA10371kG, QE10425kG, QE10441kG, QE10511kG, QE10525kG):
    setQuadkG(tao, "QA10361", QA10361kG)
    setQuadkG(tao, "QA10371", QA10371kG)
    setQuadkG(tao, "QE10425", QE10425kG)
    setQuadkG(tao, "QE10441", QE10441kG)
    setQuadkG(tao, "QE10511", QE10511kG)
    setQuadkG(tao, "QE10525", QE10525kG)

    return


def setAllFinalFocusQuads(tao, Q5FFkG, Q4FFkG, Q3FFkG, Q2FFkG, Q1FFkG, Q0FFkG, Q0DkG, Q1DkG, Q2DkG):

    setQuadkG(tao, "Q5FF", Q5FFkG)
    setQuadkG(tao, "Q4FF", Q4FFkG)
    setQuadkG(tao, "Q3FF", Q3FFkG)
    setQuadkG(tao, "Q2FF", Q2FFkG)
    setQuadkG(tao, "Q1FF", Q1FFkG)
    setQuadkG(tao, "Q0FF", Q0FFkG)
    setQuadkG(tao, "Q0D", Q0DkG)
    setQuadkG(tao, "Q1D", Q1DkG)
    setQuadkG(tao, "Q2D", Q2DkG)

    return


def setAllWChicaneBends(tao, B1EkG, B2EkG, B3EkG):
    
    setBendkG(tao, "B1LE", B1EkG)
    setBendkG(tao, "B2LE", B2EkG)
    setBendkG(tao, "B3LE", B3EkG)
    setBendkG(tao, "B3RE", B3EkG)
    setBendkG(tao, "B2RE", B2EkG)
    setBendkG(tao, "B1RE", B1EkG)

    return


def setAllWChicaneQuads(tao, Q1EkG, Q2EkG, Q3EkG, Q4EkG, Q5EkG, Q6EkG):
    
    setQuadkG(tao, "Q1EL", Q1EkG)
    setQuadkG(tao, "Q2EL", Q2EkG)
    setQuadkG(tao, "Q3EL_1", Q3EkG)
    setQuadkG(tao, "Q3EL_2", Q3EkG)
    setQuadkG(tao, "Q4EL_1", Q4EkG)
    setQuadkG(tao, "Q4EL_2", Q4EkG)
    setQuadkG(tao, "Q4EL_3", Q4EkG)
    setQuadkG(tao, "Q5EL", Q5EkG)
    setQuadkG(tao, "Q6E", Q6EkG)
    setQuadkG(tao, "Q5ER", Q5EkG)
    setQuadkG(tao, "Q4ER_1", Q4EkG)
    setQuadkG(tao, "Q4ER_2", Q4EkG)
    setQuadkG(tao, "Q4ER_3", Q4EkG)
    setQuadkG(tao, "Q3ER_1", Q3EkG)
    setQuadkG(tao, "Q3ER_2", Q3EkG)
    setQuadkG(tao, "Q2ER", Q2EkG)
    setQuadkG(tao, "Q1ER", Q1EkG)
    

    return


### Sextupole settings

def setAllWChicaneSextupoles(tao, S1ELkG, S2ELkG, S3ELkG, S3ERkG, S2ERkG, S1ERkG):
    """Set all chicane sextupole strengths in the Tao lattice."""
    setSextkG(tao, "S1EL",   S1ELkG)
    setSextkG(tao, "S2EL",   S2ELkG)
    setSextkG(tao, "S3EL_1", S3ELkG)
    setSextkG(tao, "S3EL_2", S3ELkG)
    setSextkG(tao, "S3ER_1", S3ERkG)
    setSextkG(tao, "S3ER_2", S3ERkG)
    setSextkG(tao, "S2ER",   S2ERkG)
    setSextkG(tao, "S1ER",   S1ERkG)
    return tao


def setAllWChicaneSextupolesXOffsets(tao, S1EL_dx, S2EL_dx, S3EL_dx, S3ER_dx, S2ER_dx, S1ER_dx):
    """Set all chicane sextupole horizontal offsets in the Tao lattice."""
    tao.cmd(f'set ele {"S1EL"} X_OFFSET = {S1EL_dx}')
    tao.cmd(f'set ele {"S2EL"} X_OFFSET = {S2EL_dx}')
    tao.cmd(f'set ele {"S3EL_1"} X_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3EL_2"} X_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3ER_1"} X_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S3ER_2"} X_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S2ER"} X_OFFSET = {S2ER_dx}')
    tao.cmd(f'set ele {"S1ER"} X_OFFSET = {S1ER_dx}')
    return tao


def setAllWChicaneSextupolesYOffsets(tao, S1EL_dx, S2EL_dx, S3EL_dx, S3ER_dx, S2ER_dx, S1ER_dx):
    """Set all chicane sextupole vertical offsets in the Tao lattice."""
    tao.cmd(f'set ele {"S1EL"} Y_OFFSET = {S1EL_dx}')
    tao.cmd(f'set ele {"S2EL"} Y_OFFSET = {S2EL_dx}')
    tao.cmd(f'set ele {"S3EL_1"} Y_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3EL_2"} Y_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3ER_1"} Y_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S3ER_2"} Y_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S2ER"} Y_OFFSET = {S2ER_dx}')
    tao.cmd(f'set ele {"S1ER"} Y_OFFSET = {S1ER_dx}')
    return tao


def setAllWChicaneMovers(tao, S1EL_xOffset, S1EL_yOffset, S2EL_xOffset, S2EL_yOffset, S2ER_xOffset, S2ER_yOffset, S1ER_xOffset, S1ER_yOffset):
    
    setXOffset(tao, "S1EL", S1EL_xOffset)
    setYOffset(tao, "S1EL", S1EL_yOffset)
    setXOffset(tao, "S2EL", S2EL_xOffset)
    setYOffset(tao, "S2EL", S2EL_yOffset)
    setXOffset(tao, "S2ER", S2ER_xOffset)
    setYOffset(tao, "S2ER", S2ER_yOffset)
    setXOffset(tao, "S1ER", S1ER_xOffset)
    setYOffset(tao, "S1ER", S1ER_yOffset)
    
    return


def setAllFinalFocusKickers(tao, XC1FFkG, XC3FFkG, YC1FFkG, YC2FFkG):
    
    setKickerkG(tao, "XC1FF", XC1FFkG)
    setKickerkG(tao, "XC3FF", XC3FFkG)
    setKickerkG(tao, "YC1FF", YC1FFkG)
    setKickerkG(tao, "YC2FF", YC2FFkG)

    return


def setXTCAV(tao, XTCAVvoltage, XTCAVphase):
    tao.cmd(f"set ele XTCAVF VOLTAGE = {XTCAVvoltage}")
    tao.cmd(f"set ele XTCAVF PHI0 = {XTCAVphase / 360.0}")


def setWChicaneLaunchQuads(tao, Q19851kG, Q19871kG):
    
    setQuadkG(tao, "Q19851", Q19851kG)
    setQuadkG(tao, "Q19871", Q19871kG)

    return
