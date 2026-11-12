"""Tune the lattice energy profile (cavities and dipoles) to match desired beam energies.

Moved from simulationFunctions.py.
"""

import numpy as np

from .lattice.linac import setLinacGradientAuto, scaleLinacGradientAuto, matchStringWrapper
from .lattice.optics import get_element_array
from .core import getBeamAtElement


### Correct the lattice to match the desired Pz

def tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=False):
    '''
    This function scales the cavities to match the desired_P0Cs_MeV.
    change_only_L0B: see the "edit_energy_based_on_beam_inj" function.
    '''
    desired_P0Cs_MeV = [desired_P0Cs_MeV[0] if desired_P0Cs_MeV[0] is not None else 125,
                        desired_P0Cs_MeV[1] if desired_P0Cs_MeV[1] is not None else 335,
                        desired_P0Cs_MeV[2] if desired_P0Cs_MeV[2] is not None else 4500,
                        desired_P0Cs_MeV[3] if desired_P0Cs_MeV[3] is not None else 10000]
    if change_only_L0B:
        current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
        current_pz_start = tao.ele_gen_attribs('L0AFEND')["P0C"]*1e-6
        coef_e = 1 + (desired_P0Cs_MeV[0] - current_pz_location)/(current_pz_location-current_pz_start)
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    else:
        current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
        current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]*1e-6
        coef_e = 1 + (desired_P0Cs_MeV[0] - current_pz_location)/(current_pz_location-current_pz_start)
        l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')

    # scale the present gradients (keeps per-klystron amplitudes, e.g. from the DAQ; equal gradients stay equal)
    scaleLinacGradientAuto(tao, "L1", (desired_P0Cs_MeV[1]-desired_P0Cs_MeV[0])*1e6)
    scaleLinacGradientAuto(tao, "L2", (desired_P0Cs_MeV[2]-desired_P0Cs_MeV[1])*1e6)
    scaleLinacGradientAuto(tao, "L3", (desired_P0Cs_MeV[3]-desired_P0Cs_MeV[2])*1e6)

    return tao


#### Treat dipoles

def save_dipoles(tao, dipoleEnergiesMeV=[125, 335, 4500, 10000]):
    '''
    Changes the lattice energy to the dipoleEnergiesMeV, saves the dipole fields, and changes the lattice energy back to the original values.
    '''
    initial_energy_bx = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
    initial_energy_bc11 = tao.ele_gen_attribs('BC11CBEG')["P0C"]*1e-6
    initial_energy_bc14 = tao.ele_gen_attribs('ENDL2F')["P0C"]*1e-6
    initial_energy_bc20 = tao.ele_gen_attribs('ENDL3F_2')["P0C"]*1e-6

    # Change the energies to the desired ones
    tao = tune_to_P0Cs(tao, desired_P0Cs_MeV=dipoleEnergiesMeV)

    fields = {}

    elems = get_element_array(tao, "BEGINNING", "END", values_to_show=["SBend"])[:,1]
    for ele in elems:
        fields[ele] = tao.ele_gen_attribs(ele)["B_FIELD"]

    # Change the energies back
    tao = tune_to_P0Cs(tao, desired_P0Cs_MeV=[initial_energy_bx,initial_energy_bc11,initial_energy_bc14,initial_energy_bc20])

    return fields


default_bend_fields = {
    'BCX10451': 0.4399498913496881,
    'BCX10461': -0.4399498913496881,
    'BCX10475': -0.4399498913496881,
    'BCX10481': 0.4399498913496881,
    'BX10661': 0.6242944753331842,
    'BX10751': 0.6242944753331842,
    'BCX11314': 0.5167328829297726,
    'BCX11331': -0.5167328829297726,
    'BCX11338': -0.5167328829297726,
    'BCX11355': 0.5167328829297726,
    'BCX14720': 1.145800533192734,
    'BCX14796': -1.145800533192734,
    'BCX14808': -1.145800533192734,
    'BCX14883': 1.145800533192734,
    'B1LE': -0.7088897708589302,
    'B2LE': 0.5998061022132172,
    'B3LE': -0.6450166589406902,
    'B3RE': -0.6450166589406902,
    'B2RE': 0.5998061022132172,
    'WIGE1': 0.3417645361988085,
    'WIGE2': -0.3417645361988085,
    'WIGE3': 0.3417645361988085,
    'B1RE': -0.7088897708589302,
    'B5D36': -0.2046605187706695
}


def treat_dipoles(tao, fields=default_bend_fields):
    '''
    A function to deal with the dipoles. Uses the fields from the input.
    If fields is not provided, the default_bend_fields are used (corresponding to the nominal energies).
    Use this function after all other adjustments (before running the simulation) to set the fields to the nominal values.
    WARNING! Will not work properly if the dipoles were set to field_master=True at any point.
    '''
    for k, v in fields.items():
        current_field = tao.ele_gen_attribs(k)["B_FIELD"]
        tao.cmd(f'set ele {k} DB_FIELD = {v-current_field}')
    return tao


#### Adjust based on the beam


def edit_energy_based_on_beam_inj(tao, location="BX0FBEG", desiredPzMeV=125, change_only_L0B=False):
    '''
    Scales the injector cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    change_only_L0B: scale only L0B. I think that at FACET we do exactly that.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    '''
    if change_only_L0B:
        current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
        current_pz_start = np.mean(getBeamAtElement(tao, "L0AFEND", tToZ=False).pz)
        coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    else:
        current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
        current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]
        #current_pz_location = tao.ele_gen_attribs(location)["P0C"]
        coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
        l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    return tao


def edit_energy_based_on_beam_L1(tao, location="BC11CBEG", desiredPzMeV=335):
    '''
    Scales the L1 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'BX0FBEG', tToZ=False).pz)
    activeMatchStrings = matchStringWrapper(tao, "L1")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao


def edit_energy_based_on_beam_L2(tao, location="ENDL2F", desiredPzMeV=4500):
    '''
    Scales the L2 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'BC11CBEG', tToZ=False).pz)
    activeMatchStrings = matchStringWrapper(tao, "L2")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao


def edit_energy_based_on_beam_L3(tao, location="ENDL3F_2", desiredPzMeV=10000):
    '''
    Scales the L3 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'ENDL2F', tToZ=False).pz)
    activeMatchStrings = matchStringWrapper(tao, "L3")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao
