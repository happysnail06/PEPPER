import os
import json
from tqdm import tqdm
import argparse
import re
import random
import math

import sys
sys.path.append("..")

###########  For TEST ############
datasets = ['opendialkg','redial']
models = ['unicrs','barcor','chatgpt','kbrd','macrs','chatcrs']

turn_num = 20
core_num = 10
setting = 'main'
###########  For TEST ############

class RecMetric:
    """
        Computational Metric Evaluation Class
    """
    def __init__(self, turn_num,  k_list=[5, 10, 25]):
        # K = 1, 10, 25
        self.k_list = k_list
        self.turn_num = turn_num
        self.reset_metric()

    def reset_metric(self):
        # Initialize metrics for each turn and each k
        self.count = 0  # For tracking total number of dialogues
        self.count_for_recall = 1
        self.turn_metrics = {}
        self.ndcg_all = 0.0
        for turn in range(self.turn_num+1):
            self.turn_metrics[turn] = {}  # Initialize an empty dict for each turn
            self.turn_metrics[turn]['Preference Coverage'] = 0.0
            for k in self.k_list:
                self.turn_metrics[turn][f'recall@{k}'] = 0.0      # Recall
                self.turn_metrics[turn][f'precision@{k}'] = 0.0   # Precision

    def compute_Preference_Coverage(self, recommended, gt_labels, turn):
        # Match
        correct = set(recommended) & set(gt_labels)
        
        # Recall
        recall = len(correct) / len(gt_labels)
        
        self.turn_metrics[turn]['Preference Coverage']+= recall
        
        return recall
    
    def compute_recall(self, recommended, gt_labels, k):
        # Calculate recall@k
        recommended_at_k = recommended[:k]  # Top-k recommendations
        correct = set(recommended_at_k) & set(gt_labels)
        
        # if set(recommended_at_k) & set(gt_labels):
        #     recall = 1
        # else:
        #     recall = 0
        # return recall
        recall = len(correct) / len(gt_labels)

        return recall
    
    def compute_best_rank(self, recommended, gt_labels,best_ranks):
        for gt_label in gt_labels:
            if gt_label in recommended:
                current_rank = recommended.index(gt_label) + 1

                if best_ranks[gt_label] == -1 or current_rank < best_ranks[gt_label]:
                    best_ranks[gt_label] = current_rank
        return best_ranks
    
    
    def compute_ndcg(self, best_ranks):
        total_ndcg = 0.0
        num_labels = len(best_ranks)
        
        for gt_label, rank in best_ranks.items():
            if rank != -1:
                dcg = 1 / math.log2(rank + 1)
            else:
                dcg = 0.0
            
            idcg = 1 / math.log2(1 + 1)  # IDCG when the label is ideally ranked at 1
            
            # Calculate NDCG current label
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            # Accumulate NDCG scores
            total_ndcg += ndcg
        
        # Average the NDCG over all labels
        average_ndcg = total_ndcg / num_labels if num_labels > 0 else 0.0
        self.ndcg_all+=average_ndcg
        return average_ndcg

    def compute_precision(self, recommended, gt_labels, k):
        # Calculate precision@k
        recommended_at_k = recommended[:k]  # Top-k recommendations
        relevant_recommended = set(recommended_at_k) & set(gt_labels)
        precision = len(relevant_recommended) / k if k > 0 else 0
        return precision

    def evaluate_turn(self, recommended, gt_labels):
        # Evaluate recall and precision for each turn in a dialogue
        metrics = {}
        for k in self.k_list:
            metrics[f'recall@{k}'] = self.compute_recall(recommended, gt_labels, k)
            metrics[f'precision@{k}'] = self.compute_precision(recommended, gt_labels, k)
        return metrics
    
    def accumulate_turn_metrics(self, all_turn_metrics):
        # Accumulate recall and precision for each turn and each k across all dialogues
        for turn, turn_metrics in enumerate(all_turn_metrics):
            for k in self.k_list:
                self.turn_metrics[turn][f'recall@{k}'] += turn_metrics[f'recall@{k}']
                self.turn_metrics[turn][f'precision@{k}'] += turn_metrics[f'precision@{k}']
        
        # for turn in range(self.turn_num):
        #     for k in self.k_list:
        #         print(self.turn_metrics[turn][f'recall@{k}'])
        self.count_for_recall += 1  # Increment the count of dialogues processed

    def average_metrics(self):
        # Average recall and precision per turn across all dialogues
        report = []  # Create a list to store turn-level metrics
        report.append({
            'total dialogues' : self.count_for_recall-1  # Add total number of dialogues processed
        })
        for turn in range(self.turn_num):
            turn_report = {
                "turn": turn + 1,  # Include the turn number for readability
                "metrics": {}
            }
            for k in self.k_list:
                avg_recall = self.turn_metrics[turn][f'recall@{k}'] / self.count_for_recall
                # print(self.count)
                avg_precision = self.turn_metrics[turn][f'precision@{k}'] / self.count_for_recall
            
                turn_report["metrics"][f"recall@{k}"] = avg_recall
                # turn_report["metrics"][f"precision@{k}"] = avg_precision
        
            report.append(turn_report)

        return report

    # For Preference Coverage Computation
    def get_average_Preference_Coverage(self):
        # Average recall and precision per turn across all dialogues
        report = []  # Create a list to store turn-level metrics
        report.append({
            'total dialogues' : self.count,  # Add total number of dialogues processed
            #'Average NDCG' : self.ndcg_all / self.count
        })
        for turn in range(self.turn_num):
            turn_report = {
                "turn": turn + 1,  # Include the turn number for readability
                "Preference Coverage": self.turn_metrics[turn+1]['Preference Coverage'] / self.count
            }
        
            report.append(turn_report)

        return report


def rec_eval(turn_num:int, core_num:int):
    """
        CRS Evaluation Method
    """
    for dataset in datasets:
        for model in models:
            metric = RecMetric(turn_num, [1, 5, 10, 20, 50])  #  K
            # save_path = f"results/dialogue_{turn_num}_{core_num}_{setting}_{ratio}/{model}/{dataset}/eval"  # Evaluation data path
            # result_path = f"results/result_{turn_num}_{core_num}_{setting}_{ratio}/{model}/"                # Save path
            
            result_path = f'experiments/5_2_main/results/dialogue_20_10_main/{model}'
            save_path = f"experiments/5_2_main/results/dialogue_20_10_main/{model}/{dataset}/eval"
            
            # result_path = f'results/main_experiment/results/{model}'
            # save_path = f"results/main_experiment/results/{model}/{dataset}/eval"
            os.makedirs(result_path, exist_ok=True)
 
            if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
                path_list = os.listdir(save_path)
                print(f"turn_num: {turn_num}, model: {model}, dataset: {dataset}, num_files: {len(path_list)}")

                for path in tqdm(path_list):
                    with open(f"{save_path}/{path}", 'r', encoding="utf-8") as f:
                        data = json.load(f)

                        # gt_labels = data[0]['seen'] + [data[0]['target']]  # gold labels (seen + target)
                        # gt_labels = data[0]['target_movies_sampled'] # gold labels
                        # gt_labels = data[0]['target_movies_not_sampled'] # gold labels
                        gt_labels = data[0]['target'] # gold labels
                        
                        # random.shuffle(gt_labels)
                        # if (len(gt_labels) > 1 and len(gt_labels) % 2 != 0):
                        #     gt_labels = gt_labels[:-1]
                        # half = math.ceil(len(gt_labels) / 2)
                        # # gt_labels = gt_labels[:half] # upper
                        # gt_labels = gt_labels[half:] # lower

                        
                        # def normalize_title(title_with_year: str) -> str:
                        #     return re.sub(r'\s*\(\d{4}\)$', '', title_with_year).strip()
                        # if dataset == 'opendialkg':
                        #     gt_labels = [normalize_title(movie) for movie in gt_labels]
                            
                        # print(gt_labels)
                        # For saving turn-level metrics
                        all_turn_metrics = []
                        
                        # for CLR
                        all_candidates = [] 
                        
                        # ########################### For Convention Recall###############################
                        # for turn_data in data[1:]:  # Skip the first entry (seen + target)
                        #     recommended = turn_data['recommended']  # Get recommended list for the current turn

                        #     # Evaluate recall and precision for this turn
                        #     turn_metrics = metric.evaluate_turn(recommended, gt_labels)
                            
                        #     # Save metrics for each turn
                        #     all_turn_metrics.append(turn_metrics)
                        # # print(all_turn_metrics)
                        # metric.accumulate_turn_metrics(all_turn_metrics)
                        
                        ######################## For PC ###############################
                        best_ranks = {gt_label: -1 for gt_label in gt_labels}
                        for turn_data in data[1:]:  # Skip the first entry (seen + target)
                            recommended = turn_data['recommended'][:50]  # Get recommended list for the current turn
                            # print(f'recommended count: {len(recommended)}')
                            
                            # accumulate
                            all_candidates.extend(recommended)
                            all_candidates = list(set(all_candidates))
                            # print(len(all_candidates))
                            
                            metric.compute_Preference_Coverage(all_candidates, gt_labels, turn_data['turn num'])
                            #metric.compute_best_rank(recommended, gt_labels, best_ranks)
                        
                        ########################## For PC ###############################   
                        
                        #metric.compute_ndcg(best_ranks)
                        metric.count+=1 # Count instances, Warning: Notice that accumulate_turn_metrics also increases count.
                    
                # ########## For PC ###########
                report = metric.get_average_Preference_Coverage() 
                
                # ###########Conventional Recall###########
                # report = metric.average_metrics()
                

                # Save the result for this dataset
                with open(f"{result_path}/{dataset}.json", 'w', encoding="utf-8") as w:
                    json.dump(report, w, indent=4)
                        
                
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--turn_num', default=5, type=int)
    # parser.add_argument('--core_num', default=10)
    # args = parser.parse_args()
    rec_eval(turn_num, core_num)
