import json
import os
from datasets import DatasetDict, Dataset
from ragas import evaluate
from dotenv import load_dotenv, find_dotenv
from tvalmetrics import RagScoresCalculator
import openai
load_dotenv(find_dotenv())
import pandas as pd
import logging
# Increase the maximum number of columns displayed (adjust to your preference)
pd.set_option('display.max_columns', None)

# Increase the maximum number of rows displayed (adjust to your preference)
pd.set_option('display.max_rows', None)

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

openai.api_key = os.getenv('OPENAI_API_KEY')
gpt_with_documentation_evaluation_data_path = "evaluation_files/evaluation_1024_openai.json"
evaluation_data = {}
if os.path.exists(gpt_with_documentation_evaluation_data_path):
    with open(gpt_with_documentation_evaluation_data_path, "r") as evaluation_data_file:
        evaluation_data = json.load(evaluation_data_file)

logging.info(evaluation_data)

#RAGAS EVALUATIONS FOR CONTEXT COUNT = 4
questions = [item["question"] for item in evaluation_data]
contexts = [item["contexts"] for item in evaluation_data]
answer = [item["answer"] for item in evaluation_data]
ground_truths = [[item["ground_truths"]] for item in evaluation_data]
data_length = len(evaluation_data)

dataset = Dataset.from_dict({
    "question": questions,
    "contexts": contexts,
    "answer": answer,
    "ground_truths": ground_truths,
})

results = evaluate(dataset)
logging.info("RAGAS EVALUATIONS FOR CONTEXT COUNT = 4")
logging.info(results)
#{'ragas_score': 0.4601, 'answer_relevancy': 0.9457, 'context_precision': 0.2153, 'faithfulness': 0.8029, 'context_recall': 0.5726}

#TONIC METRIC FOR CONTEXT COUNT = 4
llm_evaluator = "gpt-4"
score_calculator = RagScoresCalculator(llm_evaluator)

question_list = [item["question"] for item in evaluation_data]
retrieved_context_list_list = [item["contexts"] for item in evaluation_data]
reference_answer_list = [item["answer"] for item in evaluation_data]
llm_answer_list = [item["ground_truths"] for item in evaluation_data]


batch_scores = score_calculator.score_batch(
    question_list, reference_answer_list, llm_answer_list, retrieved_context_list_list
)

# mean of each score over the batch of question
mean_scores = batch_scores.mean_scores

# dataframe that has the input data and the scores for each question
scores_df = batch_scores.to_dataframe()
logging.info("TONIC METRIC FOR CONTEXT COUNT = 4")
logging.info(scores_df)

#     answer_similarity_score  retrieval_precision  augmentation_precision  \
# 0                       0.0                 0.00                0.000000
# 1                       5.0                 0.25                1.000000
# 2                       4.0                 0.75                0.333333
# 3                       5.0                 0.50                1.000000
# 4                       4.0                 1.00                0.500000
# 5                       0.0                 0.25                1.000000
# 6                       5.0                 0.25                1.000000
# 7                       4.0                 0.50                0.500000
# 8                       4.0                 0.00                0.000000
# 9                       2.0                 0.75                0.000000
# 10                      5.0                 0.25                1.000000
# 11                      5.0                 1.00                0.750000
# 12                      3.0                 0.50                0.500000
# 13                      5.0                 0.50                1.000000
# 14                      4.0                 0.25                1.000000
# 15                      4.0                 0.25                1.000000
# 16                      4.0                 0.50                1.000000
# 17                      2.0                 1.00                0.500000
# 18                      4.0                 0.00                0.000000
# 19                      4.0                 0.25                1.000000
# 20                      5.0                 0.75                0.333333
# 21                      0.0                 0.25                0.000000
# 22                      0.0                 0.25                0.000000
# 23                      4.0                 1.00                0.750000
# 24                      4.0                 0.75                0.666667
# 25                      2.0                 0.50                1.000000
# 26                      4.0                 0.50                1.000000
# 27                      5.0                 0.75                0.666667
# 28                      5.0                 0.25                1.000000
# 29                      5.0                 1.00                0.750000
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                    0.00            0.500000       0.100000
# 1                    0.25            1.000000       0.700000
# 2                    0.25            1.000000       0.626667
# 3                    0.50            1.000000       0.800000
# 4                    0.50            0.500000       0.660000
# 5                    0.25            1.000000       0.500000
# 6                    0.25            1.000000       0.700000
# 7                    0.25            0.333333       0.476667
# 8                    0.25            1.000000       0.410000
# 9                    0.00            0.333333       0.296667
# 10                   0.25            1.000000       0.700000
# 11                   0.75            1.000000       0.900000
# 12                   0.25            0.333333       0.436667
# 13                   0.50            1.000000       0.800000
# 14                   0.25            1.000000       0.660000
# 15                   0.50            1.000000       0.710000
# 16                   0.50            1.000000       0.760000
# 17                   0.50            0.500000       0.580000
# 18                   0.75            0.333333       0.376667
# 19                   0.50            1.000000       0.710000
# 20                   0.25            1.000000       0.666667
# 21                   0.00            0.090909       0.068182
# 22                   0.00            0.000000       0.050000
# 23                   0.75            0.250000       0.710000
# 24                   0.50            1.000000       0.743333
# 25                   0.50            1.000000       0.680000
# 26                   0.50            1.000000       0.760000
# 27                   0.50            1.000000       0.783333
# 28                   0.25            1.000000       0.700000
# 29                   0.75            1.000000       0.900000



#RAGAS EVALUATIONS FOR CONTEXT COUNT = 1
#
retrieved_single_context_list = [item["single_context"] for item in evaluation_data]

llm_single_context_answer_list = [item["single_context_answer"] for item in evaluation_data]
single_context_ground_truths = [[item["ground_truths"]] for item in evaluation_data]

questions = [item["question"] for item in evaluation_data]
ground_truths = [[item["ground_truths"]] for item in evaluation_data]
data_length = len(evaluation_data)
dataset = Dataset.from_dict({
    "question": questions,
    "contexts": retrieved_single_context_list,
    "answer": llm_single_context_answer_list,
    "ground_truths": ground_truths,
})
#
# # Print the dataset
logging.info(dataset)
results = evaluate(dataset)
logging.info("RAGAS EVALUATIONS FOR CONTEXT COUNT = 1")
logging.info(results)

#{'ragas_score': 0.5185, 'answer_relevancy': 0.9270, 'context_precision': 0.2589, 'faithfulness': 0.7603, 'context_recall': 0.6857}


#
#TONIC METRIC FOR CONTEXT COUNT = 1

llm_evaluator = "gpt-4"
score_calculator = RagScoresCalculator(llm_evaluator)

question_list = [item["question"] for item in evaluation_data]
retrieved_single_context_list = [item["single_context"] for item in evaluation_data]
reference_single_context_answer_list = [item["single_context_answer"] for item in evaluation_data]
llm_answer_list = [item["ground_truths"] for item in evaluation_data]


single_context_batch_scores = score_calculator.score_batch(
    question_list, reference_single_context_answer_list, llm_answer_list, retrieved_single_context_list
)


# mean of each score over the batch of question
single_context_mean_scores = single_context_batch_scores.mean_scores

# dataframe that has the input data and the scores for each question
single_context_scores_df = single_context_batch_scores.to_dataframe()
logging.info("TONIC METRIC FOR CONTEXT COUNT = 1")
logging.info(single_context_scores_df)

#     answer_similarity_score  retrieval_precision  augmentation_precision  \
# 0                       0.0                  0.0                     0.0
# 1                       5.0                  1.0                     1.0
# 2                       4.0                  0.0                     0.0
# 3                       5.0                  1.0                     1.0
# 4                       5.0                  1.0                     1.0
# 5                       0.0                  0.0                     0.0
# 6                       0.0                  0.0                     0.0
# 7                       4.0                  1.0                     1.0
# 8                       1.0                  0.0                     0.0
# 9                       3.0                  1.0                     0.0
# 10                      5.0                  1.0                     1.0
# 11                      5.0                  1.0                     1.0
# 12                      2.0                  1.0                     0.0
# 13                      5.0                  1.0                     1.0
# 14                      3.0                  0.0                     0.0
# 15                      4.0                  1.0                     1.0
# 16                      4.0                  1.0                     1.0
# 17                      3.0                  1.0                     0.0
# 18                      4.5                  0.0                     0.0
# 19                      4.0                  1.0                     1.0
# 20                      5.0                  1.0                     1.0
# 21                      0.0                  0.0                     0.0
# 22                      0.0                  0.0                     0.0
# 23                      4.0                  1.0                     1.0
# 24                      5.0                  1.0                     1.0
# 25                      5.0                  1.0                     1.0
# 26                      4.0                  1.0                     1.0
# 27                      5.0                  1.0                     1.0
# 28                      5.0                  1.0                     1.0
# 29                      5.0                  1.0                     1.0
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                     0.0            0.500000       0.100000
# 1                     1.0            1.000000       1.000000
# 2                     0.0            0.500000       0.260000
# 3                     1.0            1.000000       1.000000
# 4                     1.0            0.333333       0.866667
# 5                     0.0            0.400000       0.080000
# 6                     0.0            0.000000       0.000000
# 7                     1.0            0.333333       0.826667
# 8                     0.0            0.000000       0.040000
# 9                     0.0            0.333333       0.386667
# 10                    1.0            1.000000       1.000000
# 11                    1.0            1.000000       1.000000
# 12                    0.0            0.000000       0.280000
# 13                    1.0            1.000000       1.000000
# 14                    0.0            0.250000       0.170000
# 15                    1.0            1.000000       0.960000
# 16                    1.0            1.000000       0.960000
# 17                    0.0            0.500000       0.420000
# 18                    1.0            0.200000       0.420000
# 19                    1.0            1.000000       0.960000
# 20                    1.0            1.000000       1.000000
# 21                    0.0            0.000000       0.000000
# 22                    0.0            0.000000       0.000000
# 23                    1.0            0.250000       0.810000
# 24                    1.0            1.000000       1.000000
# 25                    1.0            1.000000       1.000000
# 26                    1.0            0.666667       0.893333
# 27                    1.0            1.000000       1.000000
# 28                    1.0            1.000000       1.000000
# 29                    1.0            1.000000       1.000000


#RAGAS EVALUATIONS FOR CONTEXT COUNT = 2
retrieved_two_context_list = [item["two_context"] for item in evaluation_data]

llm_two_context_answer_list = [item["two_context_answer"] for item in evaluation_data]
two_context_ground_truths = [[item["ground_truths"]] for item in evaluation_data]

questions = [item["question"] for item in evaluation_data]
ground_truths = [[item["ground_truths"]] for item in evaluation_data]
dataset = Dataset.from_dict({
    "question": questions,
    "contexts": retrieved_two_context_list,
    "answer": llm_two_context_answer_list,
    "ground_truths": ground_truths,
})
#
# # Print the dataset
logging.info(dataset)
results = evaluate(dataset)
logging.info("RAGAS EVALUATIONS FOR CONTEXT COUNT = 2")
logging.info(results)
#{'ragas_score': 0.4816, 'answer_relevancy': 0.9405, 'context_precision': 0.2362, 'faithfulness': 0.7669, 'context_recall': 0.5867}



#TONIC METRIC FOR CONTEXT COUNT = 2
llm_evaluator = "gpt-4"
score_calculator = RagScoresCalculator(llm_evaluator)

question_list = [item["question"] for item in evaluation_data]
retrieved_two_context_list = [item["two_context"] for item in evaluation_data]
reference_two_context_answer_list = [item["two_context_answer"] for item in evaluation_data]
llm_answer_list = [item["ground_truths"] for item in evaluation_data]


two_context_batch_scores = score_calculator.score_batch(
    question_list, reference_two_context_answer_list, llm_answer_list, retrieved_two_context_list
)

# mean of each score over the batch of question
two_context_mean_scores = two_context_batch_scores.mean_scores

# dataframe that has the input data and the scores for each question
two_context_scores_df = two_context_batch_scores.to_dataframe()
logging.info("TONIC METRIC FOR CONTEXT COUNT = 2")
logging.info(two_context_scores_df)

#     answer_similarity_score  retrieval_precision  augmentation_precision  \
# 0                       0.0                  0.0                     0.0
# 1                       4.0                  0.5                     1.0
# 2                       4.0                  1.0                     1.0
# 3                       5.0                  0.5                     1.0
# 4                       4.5                  1.0                     0.5
# 5                       0.0                  0.0                     0.0
# 6                       0.0                  0.0                     0.0
# 7                       4.0                  0.5                     0.0
# 8                       1.0                  0.0                     0.0
# 9                       3.0                  1.0                     0.0
# 10                      5.0                  0.5                     1.0
# 11                      5.0                  1.0                     0.5
# 12                      2.0                  1.0                     0.5
# 13                      5.0                  1.0                     1.0
# 14                      2.0                  0.0                     0.0
# 15                      5.0                  0.5                     1.0
# 16                      4.0                  0.5                     1.0
# 17                      2.0                  1.0                     0.5
# 18                      4.5                  0.0                     0.0
# 19                      5.0                  0.5                     1.0
# 20                      5.0                  1.0                     0.5
# 21                      0.0                  0.0                     0.0
# 22                      0.0                  0.5                     0.0
# 23                      5.0                  1.0                     0.5
# 24                      5.0                  1.0                     0.5
# 25                      5.0                  1.0                     1.0
# 26                      4.0                  1.0                     1.0
# 27                      5.0                  1.0                     0.5
# 28                      5.0                  0.5                     1.0
# 29                      5.0                  1.0                     0.5
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                     0.0            0.000000       0.000000
# 1                     0.5            1.000000       0.760000
# 2                     1.0            1.000000       0.960000
# 3                     0.5            1.000000       0.800000
# 4                     0.5            0.500000       0.680000
# 5                     0.0            0.800000       0.160000
# 6                     0.5            0.000000       0.100000
# 7                     0.0            0.333333       0.326667
# 8                     0.0            0.000000       0.040000
# 9                     0.0            0.333333       0.386667
# 10                    0.5            1.000000       0.800000
# 11                    0.5            1.000000       0.800000
# 12                    0.5            0.333333       0.546667
# 13                    1.0            1.000000       1.000000
# 14                    0.0            0.250000       0.130000
# 15                    0.5            1.000000       0.800000
# 16                    0.5            1.000000       0.760000
# 17                    0.5            0.500000       0.580000
# 18                    1.0            0.200000       0.420000
# 19                    0.5            1.000000       0.800000
# 20                    0.5            1.000000       0.800000
# 21                    0.0            0.000000       0.000000
# 22                    0.0            0.000000       0.100000
# 23                    0.5            0.000000       0.600000
# 24                    0.5            1.000000       0.800000
# 25                    1.0            1.000000       1.000000
# 26                    1.0            1.000000       0.960000
# 27                    0.5            1.000000       0.800000
# 28                    0.5            1.000000       0.800000
# 29                    0.5            1.000000       0.800000


#
# #RAGAS EVALUATIONS FOR CONTEXT COUNT = 8
retrieved_eight_context_list = [item["eight_context"] for item in evaluation_data]

llm_eight_context_answer_list = [item["eight_context_answer"] for item in evaluation_data]

questions = [item["question"] for item in evaluation_data]
ground_truths = [[item["ground_truths"]] for item in evaluation_data]
dataset = Dataset.from_dict({
    "question": questions,
    "contexts": retrieved_eight_context_list,
    "answer": llm_eight_context_answer_list,
    "ground_truths": ground_truths,
})
#
# # Print the dataset
logging.info(dataset)
results = evaluate(dataset)
logging.info("RAGAS EVALUATIONS FOR CONTEXT COUNT = 8")
logging.info(results)
# {'ragas_score': 0.3987, 'answer_relevancy': 0.9521, 'context_precision': 0.1633, 'faithfulness': 0.8007, 'context_recall': 0.6213}


#TONIC METRIC FOR CONTEXT COUNT = 8
llm_evaluator = "gpt-4"
score_calculator = RagScoresCalculator(llm_evaluator)

question_list = [item["question"] for item in evaluation_data]
retrieved_eight_context_list = [item["eight_context"] for item in evaluation_data]
reference_eight_context_answer_list = [item["eight_context_answer"] for item in evaluation_data]
llm_answer_list = [item["ground_truths"] for item in evaluation_data]


eight_context_batch_scores = score_calculator.score_batch(
    question_list, reference_eight_context_answer_list, llm_answer_list, retrieved_eight_context_list
)

# mean of each score over the batch of question
eight_context_mean_scores = eight_context_batch_scores.mean_scores

# dataframe that has the input data and the scores for each question
eight_context_scores_df = eight_context_batch_scores.to_dataframe()
logging.info("TONIC METRIC FOR CONTEXT COUNT = 8")
logging.info(eight_context_scores_df)

#     answer_similarity_score  retrieval_precision  augmentation_precision  \
# 0                       5.0                0.125                0.000000
# 1                       5.0                0.125                1.000000
# 2                       4.0                0.500                0.500000
# 3                       5.0                0.375                0.666667
# 4                       4.0                0.875                0.428571
# 5                       5.0                0.250                0.500000
# 6                       5.0                0.125                1.000000
# 7                       4.0                0.250                1.000000
# 8                       4.0                0.000                0.000000
# 9                       4.0                0.875                0.000000
# 10                      5.0                0.125                1.000000
# 11                      5.0                1.000                0.625000
# 12                      3.0                0.375                0.333333
# 13                      4.0                0.375                0.666667
# 14                      4.0                0.125                1.000000
# 15                      4.0                0.125                1.000000
# 16                      5.0                0.375                0.666667
# 17                      2.0                1.000                0.500000
# 18                      4.5                0.375                0.666667
# 19                      5.0                0.125                1.000000
# 20                      5.0                0.500                0.500000
# 21                      0.0                0.375                0.000000
# 22                      5.0                0.500                0.000000
# 23                      4.0                0.500                0.500000
# 24                      4.0                0.625                0.600000
# 25                      2.0                0.250                1.000000
# 26                      3.0                0.250                1.000000
# 27                      5.0                0.625                0.600000
# 28                      5.0                0.125                1.000000
# 29                      5.0                0.875                0.714286
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                   0.000            1.000000       0.425000
# 1                   0.125            1.000000       0.650000
# 2                   0.375            1.000000       0.635000
# 3                   0.250            0.500000       0.558333
# 4                   0.375            1.000000       0.695714
# 5                   0.125            1.000000       0.575000
# 6                   0.125            1.000000       0.650000
# 7                   0.250            0.333333       0.526667
# 8                   0.125            1.000000       0.385000
# 9                   0.000            0.333333       0.401667
# 10                  0.125            1.000000       0.650000
# 11                  0.625            1.000000       0.850000
# 12                  0.125            0.333333       0.353333
# 13                  0.250            1.000000       0.618333
# 14                  0.125            1.000000       0.610000
# 15                  0.500            1.000000       0.685000
# 16                  0.250            1.000000       0.658333
# 17                  0.500            0.500000       0.580000
# 18                  0.500            1.000000       0.688333
# 19                  0.375            1.000000       0.700000
# 20                  0.375            1.000000       0.675000
# 21                  0.000            0.181818       0.111364
# 22                  0.000            0.153846       0.330769
# 23                  0.250            0.250000       0.460000
# 24                  0.375            1.000000       0.680000
# 25                  0.250            1.000000       0.580000
# 26                  0.250            1.000000       0.620000
# 27                  0.375            1.000000       0.720000
# 28                  0.125            1.000000       0.650000
# 29                  0.625            1.000000       0.842857
