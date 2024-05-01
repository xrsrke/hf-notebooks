from typing import Optional, List
import torch

import sys
sys.path.append("/Users/huggingface/DATA/notebooks/hf-notebooks/doremi/experiments/")

from constants import DOMAIN_NAMES

def print_name_weights(names: List[str] = DOMAIN_NAMES, weights: Optional[torch.Tensor] = None):
    assert weights is not None

    for name, weight in zip(names, weights):
        print(name, weight)

    print(f"----- sorthing bellow ---")
    
    for idx in torch.argsort(weights, descending=True):
        print(names[idx], weights[idx], idx)


def plot_domain_weights(names: List[str] = DOMAIN_NAMES, weights: Optional[torch.Tensor] = None):
    assert weights is not None

    import matplotlib.pyplot as plt

    # Create bar chart
    plt.figure(figsize=(10, 8))
    plt.barh(names, weights.tolist(), color='skyblue')
    plt.xlabel('Weights')
    plt.ylabel('Names')
    plt.title('Bar Chart of Names vs Weights')
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    
    plt.show()

def plot_domain_weights_comparison(names: List[str] = DOMAIN_NAMES, doremi_weights: Optional[torch.Tensor] = None, reference_weights: Optional[torch.Tensor] = None):
    import matplotlib.pyplot as plt
    import numpy as np

    # Setting the positions and width for the bars
    positions = np.arange(len(names))
    width = 0.35

    # Plotting both the doremi_weights and reference_weights
    plt.figure(figsize=(10, 8))
    plt.bar(positions - width/2, doremi_weights, width, label='Doremi Weights')
    plt.bar(positions + width/2, reference_weights, width, label='Token ratio Weights')

    plt.ylabel('Weights')
    plt.title('Comparison of Doremi Weights and Reference Weights')
    plt.xticks(positions, names, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.show()


def compute_eval_stats(doremi, reference):        
    def compute_num_tasks(doremi, reference):
        num_outperform_tasks = 0
        num_same_performance_tasks = 0
        total_outperform_percents = 0
        total_underperform_percents = 0
        
        doremi_total_acc = 0
        reference_total_acc = 0
        
        num_tasks = len(doremi)
        
        for name in doremi:
            tuned_acc = doremi[name]['acc']
            ref_acc = reference[name]['acc']
            
            doremi_total_acc += tuned_acc
            reference_total_acc += ref_acc        
            if tuned_acc > ref_acc:
                num_outperform_tasks += 1
                total_outperform_percents += tuned_acc - ref_acc
            elif tuned_acc == ref_acc:
                num_same_performance_tasks += 1
            else:
                total_underperform_percents += ref_acc - tuned_acc

        failed = num_tasks - num_outperform_tasks - num_same_performance_tasks

        print(f"The number of tasks that DoReMi outperforms the reference model: {num_outperform_tasks} = {(num_outperform_tasks/num_tasks):.2%}")
        print(f"The number of tasks that DoReMi underperforms the reference model: {failed} = {(failed/num_tasks):.2%}")
        print(f"The number of tasks that DoReMi performs the same as the reference model: {num_same_performance_tasks} = {(num_same_performance_tasks/num_tasks):.2%}")

        print(f"On average, DoReMi outperforms the reference model per task by: {(total_outperform_percents/num_outperform_tasks):.2%}")
        print(f"On average, DoReMi underperforms the reference model per task by: {(total_underperform_percents/failed if failed != 0 else 0):.2%}")

        # print(f"-------- Average of all tasks ({total} tasks) --------")
        print(f"The average accuracy of DoReMi per task: {doremi_total_acc/num_tasks:.2%}")
        print(f"The average accuracy of the reference model per task: {reference_total_acc/num_tasks:.2%}")
        
        # print(f"-------- Eval Per Task (DoReMi vs Reference) --------")
        # for name in doremi:
        #     print(f"{name}: {doremi[name]['acc']:.2%} vs {reference[name]['acc']:.2%}")
        
    compute_num_tasks(doremi, reference)
    

def compute_avg_eval(data):
    sum_acc, sum_acc_norm, sum_acc_stderr, sum_acc_norm_stderr, count = 0, 0, 0, 0, 0

    # Iterate through the dictionary to calculate sums and count
    for key, values in data.items():
        if key.startswith("mmlu:"):
            sum_acc += values["acc"]
            sum_acc_norm += values["acc_norm"]
            sum_acc_stderr += values["acc_stderr"]
            sum_acc_norm_stderr += values["acc_norm_stderr"]
            count += 1
            
    average_acc = sum_acc / count if count else 0
    average_acc_norm = sum_acc_norm / count if count else 0
    average_acc_stderr = sum_acc_stderr / count if count else 0
    average_acc_norm_stderr = sum_acc_norm_stderr / count if count else 0

    new_data = {
        "mmlu:average": {
            "acc": average_acc,
            "acc_norm": average_acc_norm,
            "acc_stderr": average_acc_stderr,
            "acc_norm_stderr": average_acc_norm_stderr
        }
    }
    
    sum_acc, sum_acc_norm, sum_acc_stderr, sum_acc_norm_stderr, count = 0, 0, 0, 0, 0
    
    for key, values in data.items():
        if key.startswith("arc:"):
            sum_acc += values["acc"]
            sum_acc_norm += values["acc_norm"]
            sum_acc_stderr += values["acc_stderr"]
            sum_acc_norm_stderr += values["acc_norm_stderr"]
            count += 1
            
    average_acc = sum_acc / count if count else 0
    average_acc_norm = sum_acc_norm / count if count else 0
    average_acc_stderr = sum_acc_stderr / count if count else 0
    average_acc_norm_stderr = sum_acc_norm_stderr / count if count else 0

    new_data["arc:average"] = {
        "acc": average_acc,
        "acc_norm": average_acc_norm,
        "acc_stderr": average_acc_stderr,
        "acc_norm_stderr": average_acc_norm_stderr
    }
    
    for key, values in data.items():
        if not key.startswith("mmlu:") and not key.startswith("arc:"):
            new_data[key] = values
    
    return new_data



def plot_eval(doremi, reference):
    import pandas as pd
    
    pd.set_option('display.max_rows', None)
    
    tasks = list(doremi.keys())
    columns = ['Task', 'DoReMi ACC', 'Reference ACC', 'DoReMi AccNorm', 'Reference AccNorm']
    data = []
    
    for task in tasks:
        row = [
            task,
            doremi[task]['acc'],
            reference[task]['acc'],
            doremi[task].get('acc_norm', 'N/A'),  # Using 'N/A' for missing 'acc_norm' values
            reference[task].get('acc_norm', 'N/A')
        ]
        data.append(row)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(data, columns=columns)

    pd.set_option('display.max_rows', None)

    def highlight_greater_doremi_acc(row):
        if row['DoReMi ACC'] > row['Reference ACC']:
            return ['background-color: #59cc0c'] * len(row)  # Apply yellow background to entire row
        elif row['DoReMi ACC'] == row['Reference ACC']:
            return ['background-color: #cc9c0c'] * len(row)  # Apply yellow background to entire row
        else:
            return [''] * len(row)  # No styling for rows that don't meet the condition
    
    # Apply the styling function to the DataFrame
    styled_df = comparison_df.style.apply(highlight_greater_doremi_acc, axis=1)
    return styled_df


def get_mamba_dataset_sizes():
    fineweb_datasets = [
        ("fineweb/CC-MAIN-2023-50", 299918912908),
        ("fineweb/CC-MAIN-2023-40", 234577593868),
        ("fineweb/CC-MAIN-2023-23", 159795454845),
        ("fineweb/CC-MAIN-2023-14", 124908970667),
        ("fineweb/CC-MAIN-2023-06", 91084249541),
        ("fineweb/CC-MAIN-2022-49", 84783043327),
        ("fineweb/CC-MAIN-2022-27", 74920986479),
        ("fineweb/CC-MAIN-2022-33", 73033112530),
        ("fineweb/CC-MAIN-2022-05", 61205145838),
    ]

    fineweb_total_tokens = sum([x for _, x in fineweb_datasets])
    
    # source: https://huggingface.slack.com/archives/C06F8VDSF37/p1708173629417409?thread_ts=1708035008.026629&cid=C06F8VDSF37
    datasets = [
        ("stack_full", 300006286251),
        ("fineweb", fineweb_total_tokens),
        ("c4", 174635677947),
        ("arxiv", 30328913436),
        ("synthetic-data", 28596897142),
        ("stack-pull-requests", 20341817979),
        ("stack-jupyter-scripts", 16944691904),
        ("stack-jupyter-structured", 15298112220),
        ("open-web-math", 14011668716),
        ("stack-issues", 11380942372),
        ("stackoverflow", 10370972771),
        ("wikipedia", 5341954858),
        ("project-gutenberg", 4953685397),
        ("deepmind-math", 4837141843),
        ("stack-kaggle-scripts", 1726142721),
        ("stack-documentation", 1656392322),
    ]

    # Convert to a dictionary
    dataset_dict = {name: tokens for name, tokens in datasets}

    return [dataset_dict[name] for name in DOMAIN_NAMES]

def get_mamba_dataset_weights():
    dataset_sizes = get_mamba_dataset_sizes()
    total_tokens = sum(dataset_sizes)
    return [size/total_tokens for size in dataset_sizes]
