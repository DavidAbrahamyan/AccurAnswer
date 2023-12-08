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
gpt_with_documentation_evaluation_data_path = "evaluation_files/evaluation_1024_openai_rerank.json"
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
logging.info("((("*100)

results = evaluate(dataset)
logging.info("RAGAS EVALUATIONS FOR CONTEXT COUNT = 4")
logging.info(results)
#{'ragas_score': 0.4209, 'answer_relevancy': 0.9569, 'context_precision': 0.1675, 'faithfulness': 0.8329, 'context_recall': 0.7750}

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
# 0                       5.0                 0.25                1.000000
# 1                       5.0                 0.25                1.000000
# 2                       4.0                 0.75                0.666667
# 3                       5.0                 0.75                0.666667
# 4                       5.0                 0.50                0.500000
# 5                       5.0                 1.00                0.500000
# 6                       5.0                 0.25                1.000000
# 7                       4.0                 1.00                0.250000
# 8                       4.0                 1.00                0.500000
# 9                       4.0                 1.00                1.000000
# 10                      5.0                 0.50                0.500000
# 11                      5.0                 1.00                0.500000
# 12                      3.0                 0.50                1.000000
# 13                      5.0                 0.50                0.500000
# 14                      5.0                 0.25                1.000000
# 15                      4.0                 0.25                1.000000
# 16                      4.0                 0.75                0.666667
# 17                      2.0                 1.00                0.500000
# 18                      5.0                 0.50                1.000000
# 19                      5.0                 0.25                1.000000
# 20                      5.0                 1.00                0.500000
# 21                      4.0                 0.75                0.333333
# 22                      0.0                 0.75                0.000000
# 23                      5.0                 0.75                0.333333
# 24                      4.0                 1.00                1.000000
# 25                      2.0                 0.25                1.000000
# 26                      4.0                 0.25                1.000000
# 27                      4.0                 1.00                0.750000
# 28                      5.0                 1.00                0.500000
# 29                      5.0                 0.75                0.666667
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                    0.25            1.000000       0.700000
# 1                    0.25            1.000000       0.700000
# 2                    0.50            1.000000       0.743333
# 3                    0.50            1.000000       0.783333
# 4                    0.25            0.666667       0.583333
# 5                    0.50            1.000000       0.800000
# 6                    0.25            1.000000       0.700000
# 7                    0.25            1.000000       0.660000
# 8                    0.50            1.000000       0.760000
# 9                    1.00            1.000000       0.960000
# 10                   0.25            1.000000       0.650000
# 11                   0.50            1.000000       0.800000
# 12                   0.50            1.000000       0.720000
# 13                   0.25            1.000000       0.650000
# 14                   0.25            1.000000       0.700000
# 15                   0.50            1.000000       0.710000
# 16                   0.50            1.000000       0.743333
# 17                   0.50            0.500000       0.580000
# 18                   0.50            1.000000       0.800000
# 19                   0.50            1.000000       0.750000
# 20                   0.50            1.000000       0.800000
# 21                   0.25            1.000000       0.626667
# 22                   0.00            0.307692       0.211538
# 23                   0.50            1.000000       0.716667
# 24                   1.00            1.000000       0.960000
# 25                   0.25            1.000000       0.580000
# 26                   0.25            1.000000       0.660000
# 27                   0.75            1.000000       0.860000
# 28                   0.50            1.000000       0.800000
# 29                   0.50            1.000000       0.783333



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


#{'ragas_score': 0.6364, 'answer_relevancy': 0.9648, 'context_precision': 0.3511, 'faithfulness': 0.8688, 'context_recall': 0.8006}


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
# 0                       5.0                  1.0                     1.0
# 1                       4.0                  1.0                     1.0
# 2                       4.0                  1.0                     1.0
# 3                       5.0                  1.0                     1.0
# 4                       5.0                  1.0                     1.0
# 5                       5.0                  1.0                     1.0
# 6                       5.0                  1.0                     1.0
# 7                       5.0                  1.0                     1.0
# 8                       5.0                  1.0                     1.0
# 9                       4.0                  1.0                     1.0
# 10                      5.0                  1.0                     1.0
# 11                      5.0                  1.0                     1.0
# 12                      4.0                  1.0                     1.0
# 13                      5.0                  1.0                     1.0
# 14                      5.0                  1.0                     1.0
# 15                      4.5                  1.0                     1.0
# 16                      2.0                  1.0                     1.0
# 17                      2.0                  1.0                     1.0
# 18                      4.5                  1.0                     1.0
# 19                      4.0                  1.0                     1.0
# 20                      5.0                  1.0                     1.0
# 21                      1.0                  0.0                     0.0
# 22                      0.0                  1.0                     0.0
# 23                      3.0                  1.0                     1.0
# 24                      4.0                  1.0                     1.0
# 25                      2.0                  1.0                     1.0
# 26                      5.0                  1.0                     1.0
# 27                      5.0                  1.0                     1.0
# 28                      5.0                  1.0                     1.0
# 29                      5.0                  1.0                     1.0
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                     1.0            1.000000       1.000000
# 1                     1.0            1.000000       0.960000
# 2                     1.0            1.000000       0.960000
# 3                     1.0            1.000000       1.000000
# 4                     1.0            0.666667       0.933333
# 5                     1.0            1.000000       1.000000
# 6                     1.0            1.000000       1.000000
# 7                     1.0            1.000000       1.000000
# 8                     1.0            1.000000       1.000000
# 9                     1.0            1.000000       0.960000
# 10                    1.0            1.000000       1.000000
# 11                    1.0            1.000000       1.000000
# 12                    1.0            1.000000       0.960000
# 13                    1.0            1.000000       1.000000
# 14                    1.0            1.000000       1.000000
# 15                    1.0            1.000000       0.980000
# 16                    1.0            1.000000       0.880000
# 17                    1.0            1.000000       0.880000
# 18                    1.0            1.000000       0.980000
# 19                    1.0            1.000000       0.960000
# 20                    1.0            1.000000       1.000000
# 21                    0.0            0.000000       0.040000
# 22                    0.0            0.076923       0.215385
# 23                    1.0            1.000000       0.920000
# 24                    1.0            1.000000       0.960000
# 25                    1.0            1.000000       0.880000
# 26                    1.0            1.000000       1.000000
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
#{'ragas_score': 0.5410, 'answer_relevancy': 0.9593, 'context_precision': 0.2568, 'faithfulness': 0.8474, 'context_recall': 0.7833}



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
# 0                       5.0                  0.5                     1.0
# 1                       5.0                  0.5                     1.0
# 2                       4.0                  1.0                     1.0
# 3                       5.0                  1.0                     1.0
# 4                       4.0                  0.5                     1.0
# 5                       5.0                  1.0                     0.5
# 6                       5.0                  0.5                     1.0
# 7                       4.0                  1.0                     1.0
# 8                       2.0                  1.0                     0.5
# 9                       4.0                  1.0                     1.0
# 10                      5.0                  1.0                     0.5
# 11                      5.0                  1.0                     1.0
# 12                      3.0                  1.0                     1.0
# 13                      5.0                  0.5                     1.0
# 14                      4.0                  0.5                     1.0
# 15                      4.0                  0.5                     1.0
# 16                      3.0                  1.0                     0.5
# 17                      4.0                  1.0                     1.0
# 18                      5.0                  1.0                     1.0
# 19                      5.0                  0.5                     1.0
# 20                      5.0                  1.0                     0.5
# 21                      1.0                  0.5                     0.0
# 22                      0.0                  0.5                     0.0
# 23                      5.0                  1.0                     1.0
# 24                      4.0                  1.0                     1.0
# 25                      2.0                  0.5                     1.0
# 26                      5.0                  0.5                     1.0
# 27                      5.0                  1.0                     1.0
# 28                      5.0                  0.5                     1.0
# 29                      5.0                  1.0                     0.5
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                     0.5            1.000000       0.800000
# 1                     0.5            1.000000       0.800000
# 2                     1.0            1.000000       0.960000
# 3                     1.0            1.000000       1.000000
# 4                     0.5            0.666667       0.693333
# 5                     0.5            1.000000       0.800000
# 6                     0.5            1.000000       0.800000
# 7                     1.0            1.000000       0.960000
# 8                     0.5            0.666667       0.613333
# 9                     1.0            1.000000       0.960000
# 10                    0.5            1.000000       0.800000
# 11                    1.0            1.000000       1.000000
# 12                    1.0            1.000000       0.920000
# 13                    0.5            1.000000       0.800000
# 14                    0.5            1.000000       0.760000
# 15                    1.0            1.000000       0.860000
# 16                    0.5            1.000000       0.720000
# 17                    1.0            1.000000       0.960000
# 18                    1.0            1.000000       1.000000
# 19                    0.5            1.000000       0.800000
# 20                    0.5            1.000000       0.800000
# 21                    0.0            0.000000       0.140000
# 22                    0.0            0.076923       0.115385
# 23                    1.0            1.000000       1.000000
# 24                    1.0            1.000000       0.960000
# 25                    0.5            1.000000       0.680000
# 26                    1.0            1.000000       0.900000
# 27                    1.0            1.000000       1.000000
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
#{'ragas_score': 0.3154, 'answer_relevancy': 0.9565, 'context_precision': 0.1095, 'faithfulness': 0.8578, 'context_recall': 0.7461}


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
# 0                       5.0                0.125                1.000000
# 1                       5.0                0.125                1.000000
# 2                       4.0                0.625                0.400000
# 3                       5.0                0.500                0.750000
# 4                       5.0                0.500                0.250000
# 5                       5.0                0.625                0.200000
# 6                       5.0                0.125                1.000000
# 7                       4.0                0.625                0.200000
# 8                       4.0                1.000                0.625000
# 9                       4.0                1.000                1.000000
# 10                      5.0                0.250                0.500000
# 11                      5.0                1.000                0.625000
# 12                      3.0                0.375                0.666667
# 13                      5.0                0.375                0.666667
# 14                      5.0                0.125                1.000000
# 15                      4.0                0.125                1.000000
# 16                      4.0                0.500                0.500000
# 17                      2.0                1.000                0.500000
# 18                      4.5                0.375                1.000000
# 19                      4.5                0.125                1.000000
# 20                      5.0                0.750                0.333333
# 21                      4.0                0.625                0.400000
# 22                      0.0                0.375                0.000000
# 23                      5.0                0.625                0.800000
# 24                      4.0                0.875                0.571429
# 25                      2.0                0.125                1.000000
# 26                      4.0                0.250                0.500000
# 27                      4.0                0.500                0.750000
# 28                      5.0                0.375                0.666667
# 29                      5.0                0.625                0.600000
#
#     augmentation_accuracy  answer_consistency  overall_score
# 0                   0.125            1.000000       0.650000
# 1                   0.125            1.000000       0.650000
# 2                   0.250            1.000000       0.615000
# 3                   0.375            1.000000       0.725000
# 4                   0.125            0.833333       0.541667
# 5                   0.125            1.000000       0.590000
# 6                   0.125            1.000000       0.650000
# 7                   0.125            1.000000       0.550000
# 8                   0.625            0.666667       0.743333
# 9                   1.000            1.000000       0.960000
# 10                  0.125            1.000000       0.575000
# 11                  0.625            1.000000       0.850000
# 12                  0.250            1.000000       0.578333
# 13                  0.250            1.000000       0.658333
# 14                  0.125            1.000000       0.650000
# 15                  0.250            1.000000       0.635000
# 16                  0.250            1.000000       0.610000
# 17                  0.500            1.000000       0.680000
# 18                  0.375            1.000000       0.730000
# 19                  0.375            1.000000       0.680000
# 20                  0.375            1.000000       0.691667
# 21                  0.250            1.000000       0.615000
# 22                  0.000            0.307692       0.136538
# 23                  0.625            1.000000       0.810000
# 24                  0.500            1.000000       0.749286
# 25                  0.250            1.000000       0.555000
# 26                  0.125            1.000000       0.535000
# 27                  0.375            1.000000       0.685000
# 28                  0.250            1.000000       0.658333
# 29                  0.375            1.000000       0.720000

