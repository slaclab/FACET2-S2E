import numpy as np
import pandas as pd

# L1MatchStrings = []
# L2MatchStrings = []
# L3MatchStrings = []
# selectMarkers = []

def getLinacMatchStrings(tao):
    """
    Determine the strings required to match Bmad cavity elements
    
    Parameters
    ----------
    tao : Tao object
        Active Tao simulation object.

    Returns
    -------
    list
        [L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers] as numpy arrays of element names.
    """
    
    global L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers
    
    #These more general match strings might cause problems by including both lords and slaves!
    L1MatchStrings = ["K11_1*", "K11_2*"]
    L2MatchStrings = ["K11_4*", "K11_5*", "K11_6*", "K11_7*", "K11_8*", "K12_*", "K13_*", "K14_*"]
    L3MatchStrings = ["K15_*", "K16_*", "K17_*", "K18_*", "K19_*"]
    
    #Therefore, use lat_list and these general search strings to get specific elements
    #Recall that, by default, lat_list has the flag "-track_only" so lords are not included
    #But some of the tracking elements are super_slaves which cannot be `set`
    
    #"-array_out" flag is on by default but if flags are edited, need to re-add manually to get numeric values back
    
    L1MatchStrings = np.concatenate( [ tao.lat_list(i,"ele.name", flags="-no_slaves") for i in L1MatchStrings ] )
    L2MatchStrings = np.concatenate( [ tao.lat_list(i,"ele.name", flags="-no_slaves") for i in L2MatchStrings ] )
    L3MatchStrings = np.concatenate( [ tao.lat_list(i,"ele.name", flags="-no_slaves") for i in L3MatchStrings ] )
    
    
    selectMarkers  = np.array(["ENDDL10", "BEGBC11_1", "BC11CEND", "BEGBC14_1", "ENDBC14_2", "BEGBC20", "ENDBC20", "PENT"])
    
    return [L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers]

def getEnergyChangeFromElements(tao, activeMatchStrings):
    """
    Calculate the energy change imparted by selected elements
    
    activeMatchStrings may be a string of a common name ("L1", "L2", or "L3") or an actual list of strings to match

    
    Parameters
    ----------
    tao : Tao object
        Active Tao simulation object.
    activeMatchStrings : str or list
        String (e.g. 'L1', 'L2', 'L3') or list of element names to match.

    Returns
    -------
    float
        Net energy change (eV).

    """
    
    activeMatchStrings = matchStringWrapper(tao, activeMatchStrings)
    
    #VOLTAGE is just gradient times length; need to manually include phase info
    voltagesArr = [tao.lat_list(i, "ele.VOLTAGE", flags="-no_slaves -array_out") for i in activeMatchStrings]
    voltagesArr = np.concatenate(voltagesArr)
    
    angleMultArr = [tao.lat_list(i, "ele.PHI0", flags="-no_slaves -array_out") for i in activeMatchStrings]
    angleMultArr = np.concatenate(angleMultArr)
    angleMultArr = [np.cos(i * (2*3.1415) ) for i in angleMultArr] #Recall Bmad uses units of 1/2pi

    return( np.dot(voltagesArr, angleMultArr) )


def setLinacGradientAuto(tao, activeMatchStrings, targetVoltage): 
    """
    Set all elements to a constant gradient required to achieve target voltage change

    activeMatchStrings may be a string of a common name ("L1", "L2", or "L3") or an actual list of strings to match    
    

    Parameters
    ----------
    tao : Tao object
        Active Tao simulation object.
    activeMatchStrings : str or list
        String (e.g. 'L1', 'L2', 'L3') or list of element names to match.
    targetVoltage : float
        Desired total voltage gain.
    """
    
    activeMatchStrings = matchStringWrapper(tao, activeMatchStrings)
    
    #Set to a fixed gradient so everything is the same. Not exactly physical but probably close
    baseGradient = 1.0e7
    for i in activeMatchStrings: tao.cmd(f'set ele {i} GRADIENT = {baseGradient}')
    
    #See the resulting voltage gain
    voltageSum  = getEnergyChangeFromElements(tao, activeMatchStrings)
    
    #print(voltageSum/1e6)
    
    #Uniformly scale gradient to hit target voltage
    for i in activeMatchStrings: tao.cmd(f'set ele {i} GRADIENT = {baseGradient*targetVoltage/voltageSum}')
    
    voltageSum  = getEnergyChangeFromElements(tao, activeMatchStrings)
    
    #print(voltageSum/1e6)

def scaleLinacGradientAuto(tao, activeMatchStrings, targetVoltage):
    """
    Scale the present gradients of the elements together to achieve the target voltage change, keeping their
    ratios (e.g. per-klystron amplitudes from the DAQ). Falls back to setLinacGradientAuto (equal gradients)
    if the elements give no energy change.
    """
    activeMatchStrings = matchStringWrapper(tao, activeMatchStrings)
    voltageSum = getEnergyChangeFromElements(tao, activeMatchStrings)
    if voltageSum == 0:
        return setLinacGradientAuto(tao, activeMatchStrings, targetVoltage)
    gradients = {i: tao.ele_gen_attribs(i)["GRADIENT"] for i in activeMatchStrings}
    for i, g in gradients.items(): tao.cmd(f'set ele {i} GRADIENT = {g*targetVoltage/voltageSum}')


def setLinacPhase(tao, activeMatchStrings, phi0Deg):
    """Set all elements to a phase

    activeMatchStrings may be a string of a common name ("L1", "L2", or "L3") or an actual list of strings to match

    Parameters
    ----------
    tao : Tao object
        Active Tao simulation object.
    activeMatchStrings : str or list
        String (e.g. 'L1', 'L2', 'L3') or list of element names to match.
    phi0Deg : float
        Phase to set, in degrees.
    """
    
    activeMatchStrings = matchStringWrapper(tao, activeMatchStrings)
    
    for i in activeMatchStrings: tao.cmd(f'set ele {i} PHI0 = {phi0Deg / 360.}')


def matchStringWrapper(tao, activeMatchStrings):
    """
    Translate common names ("L1", "L2", or "L3") into relevant match strings
    This is just to make life a little simpler for people.
    Usually activeMatchStrings should be an array
    However, if it's a string that's a common name ("L1", "L2", or "L3"), return the relevant match strings
    Otherwise just spit out whatever was provided

    Parameters
    ----------
    tao : Tao object
        Active Tao simulation object.
    activeMatchStrings : str or array-like
        String ('L1', 'L2', 'L3') or array of element names.

    Returns
    -------
    array-like
        Array of element names to match.
    """
    if isinstance(activeMatchStrings, str):
        [L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers] = getLinacMatchStrings(tao)
        if activeMatchStrings == "L1":
            return L1MatchStrings
        if activeMatchStrings == "L2":
            return L2MatchStrings
        if activeMatchStrings == "L3":
            return L3MatchStrings


    return activeMatchStrings


#Broken! If you comma separate too many values it just starts ignoring them...
#def printETOTValues(activeElements):
#    printThing = tao.cmd(f"show lat {','.join(activeElements)} -att E_TOT");
#    [print(row) for row in printThing];

#Broken! If you comma separate too many values it just starts ignoring them...
#def printArbValues(activeElements, attString):
#    printThing = tao.cmd(f"show lat {','.join(activeElements)} -att {attString}");
#    [print(row) for row in printThing];

def printArbValues(tao, activeElements, attString):
    """
    Print the values of a specified attribute for a list of elements.

    Parameters
    ----------
    tao : Tao object
        Active Tao simulation object.
    activeElements : array-like
        List or array of element names. If a numpy array, will be converted to list for compatibility.
    attString : str
        Attribute name to print (e.g. 'GRADIENT', 'PHI0').

    Notes
    -----
    This function retrieves the names and the specified attribute values for each element in `activeElements` using Tao's `lat_list` method.
    The attribute values are divided by 1e9 for unit normalization (e.g., to convert from eV to GeV if the attribute is voltage-related).
    The results are combined into a 2D array (`printThing`) with element names and their corresponding attribute values.
    """
    #The .tolist() is because of issues when activeElements is a numpy array
    namesList = [ tao.lat_list(i, "ele.name", flags="-no_slaves -array_out") for i in activeElements.tolist() ]
    namesList = np.concatenate(namesList)
    valuesList = [ tao.lat_list(i, f"ele.{attString}",flags="-no_slaves -array_out") for i in activeElements.tolist() ]
    valuesList = np.concatenate(valuesList) / 1e9
    
    printThing = np.transpose([namesList, valuesList])
    print(pd.DataFrame(printThing))