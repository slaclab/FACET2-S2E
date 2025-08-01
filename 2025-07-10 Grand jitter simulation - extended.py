import multiprocessing
import os
import json

from UTILITY_quickstart import *


################################################################################################################

#PyTao monkey patch, from ChatGPT https://chatgpt.com/c/6842031d-dd38-800b-a4f1-5c98f857ffc0
import types
def safe_get_output(self, reset=True):
    n_lines = self.so_lib.tao_c_out_io_buffer_num_lines()
    lines = []
    for i in range(1, n_lines + 1):
        raw = self.so_lib.tao_c_out_io_buffer_get_line(i)
        try:
            lines.append(raw.decode('utf-8'))
        except UnicodeDecodeError:
            lines.append("[Error decoding output line]")
    if reset:
        self.so_lib.tao_c_out_io_buffer_reset()
    return lines


################################################################################################################






def jitterLinac(
    tao,
    
    L0BMatchStrings = None,
    L1MatchStrings  = None,
    L2MatchStrings  = None,
    L3MatchStrings  = None,
    
    L0BPhaseErrorDeg = 0.1,
    L1PhaseErrorDeg  = 0.7,
    L2PhaseErrorDeg  = 0.4,
    L3PhaseErrorDeg  = 0.4,

    L0BGradientErrorPercent = 0.5, 
    L1GradientErrorPercent  = 0.25,
    L2GradientErrorPercent  = 0.3,
    L3GradientErrorPercent  = 0.3,
):
    #Hardcoded numbers from https://docs.google.com/spreadsheets/d/1xeCUImz5uFSq6QA3wV91dG38s-8cyVXQMGw9hjPKa6M/edit?gid=0#gid=0


    ######################################################
    #Apply same errors to all elements in the linac region
    ######################################################

    clipLimit = 1e9
    
    #Convert to "turns"
    L0BPhaseError = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L0BPhaseErrorDeg / 360.0 
    L1PhaseError  = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L1PhaseErrorDeg  / 360.0 
    L2PhaseError  = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L2PhaseErrorDeg  / 360.0 
    L3PhaseError  = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L3PhaseErrorDeg  / 360.0

    #Give as multiplier to base gradient
    L0BGradientErrorRelative = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L0BGradientErrorPercent / 100.0
    L1GradientErrorRelative  = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L1GradientErrorPercent  / 100.0
    L2GradientErrorRelative  = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L2GradientErrorPercent  / 100.0
    L3GradientErrorRelative  = np.clip( np.random.normal(), a_min = -1*clipLimit, a_max = clipLimit) * L3GradientErrorPercent  / 100.0
    
    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = F")    
    

    [ tao.cmd(f"change ele {ele} PHI0 {L0BPhaseError}") for ele in L0BMatchStrings ]
    [ tao.cmd(f"change ele {ele} PHI0 {L1PhaseError}")  for ele in L1MatchStrings  ]
    [ tao.cmd(f"change ele {ele} PHI0 {L2PhaseError}")  for ele in L2MatchStrings  ]
    [ tao.cmd(f"change ele {ele} PHI0 {L3PhaseError}")  for ele in L3MatchStrings  ]
    
    for ele in L0BMatchStrings:
        baseGradient = tao.ele_gen_attribs(ele)["GRADIENT"]
        specificGradientError = L0BGradientErrorRelative * baseGradient
        tao.cmd(f"change ele {ele} GRADIENT {specificGradientError}")

    for ele in L1MatchStrings:
        baseGradient = tao.ele_gen_attribs(ele)["GRADIENT"]
        specificGradientError = L1GradientErrorRelative * baseGradient
        tao.cmd(f"change ele {ele} GRADIENT {specificGradientError}")

    for ele in L2MatchStrings:
        baseGradient = tao.ele_gen_attribs(ele)["GRADIENT"]
        specificGradientError = L2GradientErrorRelative * baseGradient
        tao.cmd(f"change ele {ele} GRADIENT {specificGradientError}")

    for ele in L3MatchStrings:
        baseGradient = tao.ele_gen_attribs(ele)["GRADIENT"]
        specificGradientError = L3GradientErrorRelative * baseGradient
        tao.cmd(f"change ele {ele} GRADIENT {specificGradientError}")
    
    #Reenable lattice calculations
    tao.cmd("set global lattice_calc_on = T")

    return {
    "L0BPhaseError" : L0BPhaseError,
    "L1PhaseError"  : L1PhaseError,
    "L2PhaseError"  : L2PhaseError,
    "L3PhaseError"  : L3PhaseError,

    "L0BGradientErrorRelative" : L0BGradientErrorRelative,
    "L1GradientErrorRelative"  : L1GradientErrorRelative,
    "L2GradientErrorRelative"  : L2GradientErrorRelative,
    "L3GradientErrorRelative"  : L3GradientErrorRelative
    }


    
def hashDict(d):
    return str(abs(hash(json.dumps(d, sort_keys=True))))














    
def worker(config):


    csrTF = True
    transverseWakes = False

    
    if config["startingPoint"] == "TwoBunch":
        importedDefaultSettings = loadConfig("setLattice_configs/2024-10-14_twoBunch_baseline.yml")
    elif config["startingPoint"] == "OneBunchA":
        importedDefaultSettings = loadConfig("setLattice_configs/2024-10-22_oneBunch_baseline3.yml")
    elif config["startingPoint"] == "OneBunchB":
        importedDefaultSettings = loadConfig("setLattice_configs/2024-10-22_oneBunch_baseline2.yml")
    elif config["startingPoint"] == "OneBunchC":
        importedDefaultSettings = loadConfig("setLattice_configs/2024-10-22_oneBunch_baseline2.1.yml")
    else:
        print("Illegal starting point")
        return


    #Disable all magic
    importedDefaultSettings = importedDefaultSettings | { "centerBC14" : False, "centerBC20" : False, "centerMFFF" : False, "assertBC14Energy" : False, "assertBC20Energy" : False }
    
    tao = initializeTao(
        inputBeamFilePathSuffix = importedDefaultSettings["inputBeamFilePathSuffix"],
        csrTF = csrTF,
        numMacroParticles=1e4,
        scratchPath = "/tmp",
        randomizeFileNames = True,
        transverseWakes = transverseWakes
    )

    #Monkey patch defined above
    tao.get_output = types.MethodType(safe_get_output, tao)



    
    # Set up lattice
    setLattice(tao, **importedDefaultSettings)

    L1MatchStrings, L2MatchStrings, L3MatchStrings, selectMarkers = getLinacMatchStrings(tao)
    L0BMatchStrings = ["L0BF"]

    disableAutoMagnetEnergyCompensation(tao)


    
    #Jitter
    if config["jitterParameters"] == "KlystronJitter":
    
        jitterDict = jitterLinac(
            tao,
                
            L0BMatchStrings = L0BMatchStrings,
            L1MatchStrings = L1MatchStrings,
            L2MatchStrings = L2MatchStrings,
            L3MatchStrings = L3MatchStrings,
            
            L0BPhaseErrorDeg = 0.1,                #Only one klystron on L0B
            L1PhaseErrorDeg  = 0.7 / np.sqrt(2),   #Two klystrons in L1
            L2PhaseErrorDeg  = 0.4 / np.sqrt(25),  #Estimate 25 klystrons in L2. DNT!
            L3PhaseErrorDeg  = 0.4 / np.sqrt(25),  #Estimate 25 klystrons in L3. DNT!
        
            L0BGradientErrorPercent = 0.5,                  #Only one klystron on L0B
            L1GradientErrorPercent  = 0.25 / np.sqrt(2),    #Two klystrons in L1
            L2GradientErrorPercent  = 0.3  / np.sqrt(25),   #Estimate 25 klystrons in L2. DNT!
            L3GradientErrorPercent  = 0.3  / np.sqrt(25),   #Estimate 25 klystrons in L3. DNT!
        )

    elif config["jitterParameters"] == "ChunkJitter":

        jitterDict = jitterLinac(
            tao,
                
            L0BMatchStrings = L0BMatchStrings,
            L1MatchStrings = L1MatchStrings,
            L2MatchStrings = L2MatchStrings,
            L3MatchStrings = L3MatchStrings,
            
            L0BPhaseErrorDeg = 0.1,  
            L1PhaseErrorDeg  = 0.7,  
            L2PhaseErrorDeg  = 0.4,  
            L3PhaseErrorDeg  = 0.4,  
        
            L0BGradientErrorPercent = 0.5,   
            L1GradientErrorPercent  = 0.25,  
            L2GradientErrorPercent  = 0.3,   
            L3GradientErrorPercent  = 0.3,   
        )

    else:
        print("Illegal jitterParameters")
        return
    
    print(jitterDict)
    hashStr = hashDict(jitterDict)


    #OPTIONAL: disable all apertures. Very nonphysical, but I want to see if this solves some headaches
    # totalNumElements = len(tao.lat_list("*", "ele.name"))
    # for eleII in range(totalNumElements):
    #     try:
    #         tao.cmd(f"set ele {eleII} APERTURE_AT = NO_APERTURE")
    #     except:
    #         pass



    #Apply misalignments
    if config["alignment"] == "PinkCurveSteered":
        with open('./other_configs/pinkCurveMisalignments.json', 'r') as f:
            misalignmentConfig = json.load(f)       
        applyOtherConfig(tao, misalignmentConfig) 

        if config["startingPoint"] == "TwoBunch":
            with open('./other_configs/2025-06-13_pinkCurveSteering_2024-10-14_twoBunch_baseline_extended.json', 'r') as f:
                steeringConfig = json.load(f)
            applyOtherConfig(tao, steeringConfig) 
        
        elif config["startingPoint"] == "OneBunchA" or config["startingPoint"] == "OneBunchB": 
            with open('./other_configs/2025-06-12_pinkCurveSteering.json', 'r') as f:
                steeringConfig = json.load(f)
            applyOtherConfig(tao, steeringConfig) 

        else:
            print("Problem when steering?")
            return

    elif config["alignment"] == "BlueCurveSteered":
        with open('./other_configs/blueCurveMisalignments.json', 'r') as f:
            misalignmentConfig = json.load(f)       
        applyOtherConfig(tao, misalignmentConfig) 

        if config["startingPoint"] == "TwoBunch":
            with open('./other_configs/2025-06-13_blueCurveSteering_2024-10-14_twoBunch_baseline_extended.json', 'r') as f:
                steeringConfig = json.load(f)
            applyOtherConfig(tao, steeringConfig) 
        
        elif config["startingPoint"] == "OneBunchA" or config["startingPoint"] == "OneBunchB": 
            with open('./other_configs/2025-06-12_blueCurveSteering.json', 'r') as f:
                steeringConfig = json.load(f)
            applyOtherConfig(tao, steeringConfig) 
            
        else:
            print("Problem when steering?")
            return
            
    elif config["alignment"] == "Winter2025BBASteered":
        with open('./other_configs/winter2025BBAMisalignments.json', 'r') as f:
            misalignmentConfig = json.load(f)       
        applyOtherConfig(tao, misalignmentConfig) 

        if config["startingPoint"] == "TwoBunch":
            with open('./other_configs/2025-06-13_winter2025BBASteering_2024-10-14_twoBunch_baseline_extended.json', 'r') as f:
                steeringConfig = json.load(f)
            applyOtherConfig(tao, steeringConfig) 
        
        elif config["startingPoint"] == "OneBunchA" or config["startingPoint"] == "OneBunchB": 
            with open('./other_configs/2025-06-12_winter2025BBASteering.json', 'r') as f:
                steeringConfig = json.load(f)
            applyOtherConfig(tao, steeringConfig) 
        
        else:
            print("Problem when steering?")
            return


    
    elif config["alignment"] == "Aligned":
        pass

    else:
        print("Illegal alignment term")
        return

    
    try:
        #Track
        trackBeam(tao, **importedDefaultSettings)

        exportPath = "/Users/nmajik/tmpSLACData/"+config["startingPoint"]+"_"+config["alignment"]+"_"+config["jitterParameters"]+"/"
        
        try:
            os.mkdir(exportPath)
        except:
            pass
        
        runSpecsDict = config | {"csrTF":csrTF, "transverseWakes":transverseWakes} | jitterDict
        tao.cmd("write bmad "+exportPath+hashStr+"_lat.bmad")
        
        # elementName : friendlyName
        dumpPoints = {
            "BC11CBEG"  : "BEGBC11",
            "ENDBC11_2" : "ENDBC11",
            "BEGBC14_1" : "BEGBC14",
            "ENDBC14_2" : "ENDBC14",
            "BEGBC20"   : "BEGBC20",
            "PENT"      : "PENT"
        }

        for dumpEle in dumpPoints.keys():
            dumpEleFriendlyName = dumpPoints[dumpEle]
        
            P = getBeamAtElement(tao, dumpEle)
            #fig = plotMod(P, 'z', 'pz', bins=300, xlim = (-200e-6, 100e-6), ylim = (9.6e9, 10.2e9))
    
    
            exportDict = runSpecsDict | getBeamSpecs(P) 
        
    

            P.write(exportPath+hashStr+"_"+dumpEleFriendlyName+".h5")      
            
            with open(exportPath+hashStr+"_"+dumpEleFriendlyName+"_jitterDict.json", "w") as f:
                json.dump(exportDict, f)
    
            #fig.savefig(exportPath+hashStr+"_"+dumpEleFriendlyName+".png")


    except Exception as e:
        print("An error occurred:", e)

    return

    

if __name__ == "__main__":
    num_workers = 6 #8


    tasks = [
        {"startingPoint": startingPoint, "alignment": alignment, "jitterParameters": jitterParameters}
        for ii in range(100)
        for startingPoint in ["OneBunchC"] #["TwoBunch", "OneBunchA", "OneBunchB"]
        for alignment in {"Aligned"} #{"Aligned", "PinkCurveSteered", "BlueCurveSteered", "Winter2025BBASteered"}
        for jitterParameters in {"ChunkJitter", "KlystronJitter"}
    ]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker, tasks)

    for res in results:
        print(res)