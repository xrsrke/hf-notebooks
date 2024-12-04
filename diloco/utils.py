from name import Datatype

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


### MODEL SIZE

def get_model_gradient_size(model_size: int, datatype: Datatype):
    from constants import DATATYPE_TO_SIZE
    return model_size * DATATYPE_TO_SIZE[datatype]


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


##### UNIT CONVERSIONS #####

def convert_to_xt_format(number):
    """
    Converts a number to the format 'xT', where 'T' represents trillions.
    
    Args:
        number (int or float): The numeric value to convert.

    Returns:
        str: The formatted string in 'xT' format.
    """
    if not isinstance(number, (int, float)):
        raise TypeError("Input must be a number (int or float).")
    
    # Convert to trillions
    trillions = number / 1e12
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
    return f"{petaflops:.3f} PFLOPs"


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
    return f"{days:.3f} days"


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
    return f"{years:.3f} years"


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

