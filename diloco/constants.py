from name import Datatype, Supercomputer


### DATA
# 40m (leandro's source)
MAX_CRITICAL_BATCH_SIZE = 40*10**6


### COMPUTE
NUM_SEC_PER_STEP = 0.5


# NOTE: https://www.electronicspecifier.com/news/analysis/nvidia-s-h100-microchips-projected-to-surpass-energy-consumption-of-entire-nations#:~:text=Each%20H100%20GPU%2C%20running%20at,to%20the%20average%20American%20household.
# at 61% utilization
# H100_ELECTRICITY_CONSUMPTION_WATTS_PER_HOUR = 3740*1000

# source: https://semianalysis.com/2024/06/17/100000-h100-clusters-power-network/#power-challenges
H100_WATT = 700
H100_EQUIPMENT_WATT = 575
TOTAL_H100_WATT = H100_WATT + H100_EQUIPMENT_WATT

H100_THEORICAL_PEAK_FLOPS_FP16_TC = 989e12
H100_THEORICAL_PEAK_FLOPS_FP8_TC = 1979e12
H100_COST_PER_HOUR = 2
H100_COST_PER_GPU = 30000

BFLOAT16_MFU = 0.55
FP8_MFU = 0.40

UTILIZED_FP8_FLOPS = H100_THEORICAL_PEAK_FLOPS_FP8_TC * FP8_MFU
UTILIZED_BFLOAT16_FLOPS = H100_THEORICAL_PEAK_FLOPS_FP16_TC * BFLOAT16_MFU


DATATYPE_TO_SIZE = {
    Datatype.FP8: 1,
    Datatype.BFLOAT16: 2
}


### STORAGE
FP8_BYTES = 1
BFLOAT16_BYTES = 2


### COMMUNICATION
EARTH_EQUATORIAL_DIAMETER = 12742 # km, 12742 km


### UNIT
NUM_SECONDS_IN_A_DAY = 60 * 60 * 24
NUM_SECONDS_IN_A_YEAR = NUM_SECONDS_IN_A_DAY * 365
SPEED_OF_LIGHT = 299792458 # m/s, km/s = 299792.458

FRANCE_SUPERCOMPUTERS = {
    # google map: https://www.google.com/maps/place/IDRIS+-+CNRS/@48.7071906,2.1727666,16z/data=!3m1!4b1!4m6!3m5!1s0x47e67f52d2e55399:0xc648872cfba08d78!8m2!3d48.7071906!4d2.1753469!16s%2Fg%2F1tf7f_1v?entry=ttu&g_ep=EgoyMDI0MTEyNC4xIKXMDSoASAFQAw%3D%3D
    "JEAN_ZAY": Supercomputer("JEAN_ZAY", (48.7071906, 2.1753469)),

    # NOTE: located at "Very Large Computing Centre tgcc"
    # tgcc stands for "Très Grand Centre de Calcul"
    # google map: https://www.google.com/maps/place/IN2P3+Computing+Center/@45.7826699,4.8626987,17z/data=!3m1!4b1!4m6!3m5!1s0x47f4ea97be7678f3:0x26c07dda0ba3db55!8m2!3d45.7826699!4d4.865279!16s%2Fg%2F1216x10y?entry=ttu&g_ep=EgoyMDI0MTEyNC4xIKXMDSoASAFQAw%3D%3D
    "JOLIOT_CURIE": Supercomputer("JOLIOT_CURIE", (45.7826699, 4.8626)),
    # NOTE: Météo-France
    # "METEO_FRANCE": Supercomputer("METEO_FRANCE", (45.7826699, 4.8626987)),
    
    # NOTE: Lawrence Livermore National Laboratory
    "EL_CAPITAN": Supercomputer("EL_CAPITAN", (37.6869634,-121.7084555)),
}

