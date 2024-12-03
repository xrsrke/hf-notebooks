from name import Datatype


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

BFLOAT16_MFU = 0.55
FP8_MFU = 0.40

UTILIZED_FP8_FLOPS = H100_THEORICAL_PEAK_FLOPS_FP8_TC * FP8_MFU
UTILIZED_BFLOAT16_FLOPS = H100_THEORICAL_PEAK_FLOPS_FP16_TC * BFLOAT16_MFU


DATATYPE_TO_SIZE = {
    Datatype.FP8: 1,
    Datatype.BFLOAT16: 2
}


### UNIT
NUM_SECONDS_IN_A_DAY = 60 * 60 * 24
NUM_SECONDS_IN_A_YEAR = NUM_SECONDS_IN_A_DAY * 365
SPEED_OF_LIGHT = 299792458 # m/s
