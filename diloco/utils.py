from name import Datatype, Transformer
from constants import TOTAL_H100_WATT

# def chinchilla_flops(seq_len, vocab_size, d_model, num_heads, num_layers, ffw_size):
#     """ 
#     Calculate total number of FLOPs, see Chinchilla 
#     paper Appendix F as reference: https://arxiv.org/pdf/2203.15556.pdf
#     """ 
#     key_size = d_model // num_heads

#     # embeddings
#     embeddings = 2 * seq_len * vocab_size * d_model

#     # attention
#     # key, query, value projections
#     attention = 2 * 3 * seq_len * d_model * (key_size * num_heads)
#     # key @ query logits
#     attlogits = 2 * seq_len * seq_len * (key_size * num_heads)
#     # softmax
#     attsoftmax = 3 * num_heads * seq_len * seq_len # 3* is for subtract (max), exp, divide (?)
#     # softmax @ value reductions
#     attvalue = 2 * seq_len * seq_len * (key_size * num_heads)
#     # final linear
#     attlinear = 2 * seq_len * (key_size * num_heads) * d_model
#     att = attention + attlogits + attsoftmax + attvalue + attlinear
#     # feed forward
#     dense = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)

#     # logits
#     logits = 2 * seq_len * d_model * vocab_size
    
#     # this is what you'd expect:
#     # forward_flops = embeddings + num_layers * (att + dense) + logits
#     # but:
#     # per author correspondence apparently there is typo in the paper,
#     # they do not count embeddings and logits to repro table 4. So instead:
#     forward_flops = num_layers * (att + dense)
#     backward_flops = 2 * forward_flops # as in Kaplan et al. 2020
#     total_flops = forward_flops + backward_flops

#     return total_flops


### SCALING LAWS

def compute_training_flops(dataset_size, model_size):
    # reference: https://blog.eleuther.ai/transformer-math/#compute-requirements
    return 6 * dataset_size * model_size

def get_dataset_size_from_model_size(model_size):
    # According to Chinchilla scaling laws, optimal number of training tokens D = 20 * N
    return 20 * model_size

def get_maximum_tokens_per_step(model_size):
    # source: leandro's slide: https://docs.google.com/presentation/d/1uFd95VFSefD_Pom12kZ6q7ZppBJuT-T1vSGMUojDaBQ/edit#slide=id.p
    # given the best scenario, we can do 40m tokens per step
    return 40*10**6


def get_num_training_steps(dataset_size, model_size):
    return dataset_size / get_maximum_tokens_per_step(model_size)


def calculate_total_steps(model_size, dataset_size, global_batch_size):
    """ compute the total number of steps for a given model size and global batch size """
    # dataset_size = get_dataset_size_from_model_size(model_size)
    num_total_steps = dataset_size // global_batch_size
    return num_total_steps


def calculate_total_flops(model_size, dataset_size):
    # dataset_size = get_dataset_size_from_model_size(model_size)
    total_flops = compute_training_flops(dataset_size=dataset_size, model_size=model_size)
    return total_flops


def calculate_flops_per_step(model_size, global_batch_size):
    flops_per_step = compute_training_flops(dataset_size=global_batch_size, model_size=model_size) 
    return flops_per_step


def calculate_num_h100s_per_step(model_size, global_batch_size, hardware_flops):
    """
    hardware_flops: utilized harfware flops
    """
    flops_per_step = calculate_flops_per_step(model_size, global_batch_size)
    return flops_per_step // hardware_flops


def get_time_per_step(model_size, global_batch_size, hardware_flops):
    flop_per_step = calculate_flops_per_step(model_size, global_batch_size)
    num_h100s = calculate_num_h100s_per_step(model_size, global_batch_size, hardware_flops)
    aggregate_throughput = num_h100s * hardware_flops
    
    try:
        if aggregate_throughput == 0:
            return 0
        return flop_per_step / aggregate_throughput
    except:
        assert 1 == 1


def calculate_total_time_to_train_a_model(model_size, dataset_size, global_batch_size, time_per_step):
    total_steps = calculate_total_steps(model_size, dataset_size=dataset_size, global_batch_size=global_batch_size)
    return time_per_step * total_steps


def get_training_cost(model_size, dataset_size, global_batch_size, utilized_flops, cost_per_hour):
    num_h100s = calculate_num_h100s_per_step(model_size, global_batch_size, utilized_flops)
    time_per_step = get_time_per_step(model_size, global_batch_size, utilized_flops)
    time_per_step = time_per_step / 3600 # convert to hours
    num_steps = calculate_total_steps(model_size, dataset_size, global_batch_size)
    return num_h100s * cost_per_hour * time_per_step * num_steps


### MODEL SIZE

def get_deepseek_model_size(n_layers, d_model, seq_len):
    # equation 2: M=72 n_{\text {layer }} d_{\text {model }}^2+12 n_{\text {layer }} d_{\text {model }} l_{\text {seq }}
    return  72 * n_layers * d_model**2 + 12 * n_layers * d_model * seq_len

def get_model_gradient_size(model_size: int, datatype: Datatype):
    from constants import DATATYPE_TO_SIZE
    return model_size * DATATYPE_TO_SIZE[datatype]


def get_critical_batch_size(model_size):
    # page 9, deepseek's critical batch size scaling law
    # https://arxiv.org/abs/2401.02954
    return 0.2920 * (model_size ** 0.3271)


### COMMUNICATION
def calculate_surface_distance(coord1, coord2):
    from geopy.distance import great_circle
    """
    Calculate the shortest distance between two points along the Earth's surface.
    Uses the great circle distance (orthodromic distance).
    
    Args:
        coord1: Tuple of (latitude, longitude) for first point
        coord2: Tuple of (latitude, longitude) for second point
        
    Returns:
        Distance in kilometers
    """
    return great_circle(coord1, coord2).kilometers


def compute_minimum_latency_between_clusters(cluster_1_name, cluster_2_name):
    from constants import FRANCE_SUPERCOMPUTERS, SPEED_OF_LIGHT
    cluster_1_coordinate = FRANCE_SUPERCOMPUTERS[cluster_1_name].coordinate
    cluster_2_coordinate = FRANCE_SUPERCOMPUTERS[cluster_2_name].coordinate
    distance = calculate_surface_distance(cluster_1_coordinate, cluster_2_coordinate)
    minimum_latency = distance / SPEED_OF_LIGHT
    return minimum_latency

def calculate_total_minimum_comm_latency_to_train_a_model(model_size, global_batch_size, minimum_latency):
    total_steps = calculate_total_steps(model_size, global_batch_size)
    return total_steps * minimum_latency


def calculate_comm_time_given_comm_volume(comm_volume, bandwidth, world_size):
    """
    comm_volume: in bytes
    bandwidth: in bytes per second
    """
    return comm_volume / (bandwidth/world_size)


### MEMORY
def calculate_weight_memory(num_parameters, bytes_per_param):
    return num_parameters * bytes_per_param


def calculate_kv_cache(transformer: Transformer, dtype_bytes: int):
    # https://github.com/stas00/ml-engineering/tree/master/inference#kv-caching
    # return dtype_bytes * 2 * num_hidden_layers * hidden_size * num_key_value_heads / num_attention_heads
    kv_cache_per_token = dtype_bytes * 2 * transformer.n_layers * transformer.hidden_size * transformer.n_key_value_heads / transformer.n_heads
    kv_cache_per_sequence = kv_cache_per_token * transformer.ctx_length

    # kv_cache_per_sequence = dtype_bytes * 2 * transformer.n_layers * transformer.n_heads * (transformer.hidden_size / transformer.n_heads) * transformer.ctx_length
    return kv_cache_per_sequence

### ELECTRICITY CONSUMPTION
def calculate_electricity_consumption_of_an_h100(power = TOTAL_H100_WATT, time = None):
    """
    time is in seconds
    Energy $(\mathrm{E})$ is the product of Power $(\mathrm{P})$ and Time $(\mathrm{t})$ :

    $$
    E=P \times t
    $$


    If Time is in Seconds:
    - Power (P): 1275 Watts
    - Time (t): $x$ seconds
    - Energy (E) in Joules (since 1 Watt = 1 Joule/second)
    """
    assert time is not None
    return power * time


##### UNIT CONVERSIONS #####


def convert_bytes_to_terabytes(bytes_value):
    """
    Converts a value in bytes to terabytes (TB).
    
    Args:
        bytes_value (int or float): The number of bytes to convert.

    Returns:
        float: The value in terabytes.
    """
    # if not isinstance(bytes_value, (int, float)):
    #     raise TypeError("Input must be a number (int or float).")
    
    # 1 terabyte = 1e12 bytes
    terabytes = bytes_value / 1e12
    return terabytes


def convert_to_million_format(number):
    # if not isinstance(number, (int, float)):
    #     raise TypeError("Input must be a number (int or float).")
    
    millions = number / 1e6

    if abs(millions) < 0.1:  # Consider very small numbers close to 0
        return number
    else:
        # return f"{millions:.1f}m"
        return f"{millions:,.1f}m"


def convert_to_billion_format(number):
    """
    Converts a number to the format 'xB', where 'B' represents billions.
    
    Args:
        number (int or float): The numeric value to convert.

    Returns:
        str: The formatted string in 'xB' format.
    """
    # if not isinstance(number, (int, float)):
    #     raise TypeError("Input must be a number (int or float).")
    
    # Convert to billions
    number = float(number)
    billions = number / 1e9
    # return f"{billions:.2f}B"
    return f"{billions:,.2f}B"


def convert_to_xt_format(number):
    """
    Converts a number to the format 'xT', where 'T' represents trillions.
    
    Args:
        number (int or float): The numeric value to convert.

    Returns:
        str: The formatted string in 'xT' format.
    """
    # if not isinstance(number, (int, float)):
    #     raise TypeError("Input must be a number (int or float).")
    
    # Convert to trillions
    number = float(number)
    trillions = number / 1e12
    if abs(trillions) < 0.1:  # Consider very small numbers close to 0
        return convert_to_billion_format(number)
    else:
        return f"{trillions:.1f}T"


def convert_to_petaflops(flops):
    """
    Convert a number representing FLOPs into a string representing petaflops.

    Args:
        flops (float): The number of FLOPs.

    Returns:
        str: A string representation of FLOPs in petaflops.
    """
    petaflops = flops / (10**15)
    return "{:,}".format(petaflops) + " PFLOPs"

def convert_to_exaflops(flops):
    """
    Convert a number representing FLOPs into a string representing exaflops.

    Args:
        flops (float): The number of FLOPs.

    Returns:
        str: A string representation of FLOPs in exaflops.
    """
    exaflops = flops / (10**18)
    return "{:,}".format(exaflops) + " EFLOPs"


def convert_bytes_to_megabytes(bytes_count):
    """
    Convert a number representing bytes into megabytes.

    Args:
        bytes_count (int or float): The number of bytes.

    Returns:
        str: A string representation of bytes in megabytes.
    """
    megabytes = bytes_count / (10**6)
    return f"{megabytes:.3f} MB"


def convert_bytes_to_gigabytes(bytes_count):
    """
    Convert a number representing bytes into gigabytes.

    Args:
        bytes_count (int or float): The number of bytes.

    Returns:
        str: A string representation of bytes in gigabytes.
    """
    gigabytes = bytes_count / (10**9)
    return f"{gigabytes:.3f} GB"



def convert_bytes_to_terabytes(bytes_count):
    """
    Convert a number representing bytes into terabytes.

    Args:
        bytes_count (int or float): The number of bytes.

    Returns:
        str: A string representation of bytes in terabytes.
    """
    terabytes = bytes_count / (10**12)
    return f"{terabytes:.3f} TB"


def convert_seconds_to_days(seconds):
    """
    Convert a number of seconds into days.

    Args:
        seconds (int): The number of seconds.

    Returns:
        str: A string representation of the equivalent time in days.
    """
    days = seconds / (24 * 60 * 60)  # 1 day = 24 hours * 60 minutes * 60 seconds
    return "{:.1f}".format(days) + " days"

def convert_seconds_to_years(seconds):
    """
    Convert a number of seconds into years.

    Args:
        seconds (int): The number of seconds.

    Returns:
        str: A string representation of the equivalent time in years.
    """
    # 1 year = 365.25 days (to account for leap years) * 24 hours * 60 minutes * 60 seconds
    seconds_in_a_year = 365.25 * 24 * 60 * 60
    years = seconds / seconds_in_a_year
    # return "{:,}".format(years) + " years"
    return f"{years:.2f} years"

def convert_watts_to_megawatts(watts):
    """
    Convert a number of watts into megawatts.

    Args:
        watts (float or int): The number of watts.

    Returns:
        str: A string representation of the equivalent power in megawatts.
    """
    megawatts = watts / 1_000_000  # 1 megawatt = 1,000,000 watts
    return f"{megawatts:.3f} MW"


def convert_watts_to_terawatts(watts):
    """
    Convert a number of watts into terawatts.

    Args:
        watts (float or int): The number of watts.

    Returns:
        str: A string representation of the equivalent power in terawatts.
    """
    terawatts = watts / 1_000_000_000_000  # 1 terawatt = 1,000,000,000,000 watts
    return f"{terawatts:.3f} TW"
