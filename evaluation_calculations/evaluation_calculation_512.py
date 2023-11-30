import json
import os
from datasets import DatasetDict, Dataset
from ragas import evaluate
from dotenv import load_dotenv, find_dotenv
from tvalmetrics import RagScoresCalculator
import openai
load_dotenv(find_dotenv())
import pandas as pd

# Increase the maximum number of columns displayed (adjust to your preference)
pd.set_option('display.max_columns', None)

# Increase the maximum number of rows displayed (adjust to your preference)
pd.set_option('display.max_rows', None)

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
gpt_with_documentation_evaluation_data_path = "evaluation_files/evaluation_512_openai.json"
evaluation_data = {}
if os.path.exists(gpt_with_documentation_evaluation_data_path):
    with open(gpt_with_documentation_evaluation_data_path, "r") as evaluation_data_file:
        evaluation_data = json.load(evaluation_data_file)

print(evaluation_data)

#RAGAS EVALUATIONS FOR CONTEXT COUNT = 4
# questions = [item["question"] for item in evaluation_data]
# contexts = [item["contexts"] for item in evaluation_data]
# answer = [item["answer"] for item in evaluation_data]
# ground_truths = [[item["ground_truths"]] for item in evaluation_data]
# data_length = len(evaluation_data)
#
# dataset = Dataset.from_dict({
#     "question": questions,
#     "contexts": contexts,
#     "answer": answer,
#     "ground_truths": ground_truths,
# })
#
# results = evaluate(dataset)
# print(results)
#{'ragas_score': 0.4059, 'answer_relevancy': 0.9737, 'context_precision': 0.1646, 'faithfulness': 0.6655, 'context_recall': 0.8000}

#TONIC METRIC FOR CONTEXT COUNT = 4
# llm_evaluator = "gpt-4"
# score_calculator = RagScoresCalculator(llm_evaluator)
#
# question_list = [item["question"] for item in evaluation_data]
# retrieved_context_list_list = [item["contexts"] for item in evaluation_data]
# reference_answer_list = [item["answer"] for item in evaluation_data]
# llm_answer_list = [item["ground_truths"] for item in evaluation_data]
#
#
# batch_scores = score_calculator.score_batch(
#     question_list, reference_answer_list, llm_answer_list, retrieved_context_list_list
# )
#
# # mean of each score over the batch of question
# mean_scores = batch_scores.mean_scores
#
# # dataframe that has the input data and the scores for each question
# scores_df = batch_scores.to_dataframe()
#
# print(scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [‘instruct’ models that have received further ...                      5.0
# 1  [corpus. They are made available under the Apa...                      5.0
# 2  [g) FalconConfig configuration class: FalconFo...                      4.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [supported in the Transformers library. If you...                      4.5
# 5  [ma - Instruct variants support infilling base...                      5.0
# 6  [s/2203.13474) CodeLlama (from MetaAI) release...                      5.0
# 7  [h and commercial use. Check out all Code Llam...                      2.0
# 8  [tokenizer = LlamaTokenizer.from_pretrained("/...                      4.0
# 9  [en_type_ids_from_sequences.example) BERT sequ...                      3.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                 0.25                0.000000                   0.00
# 1                 0.25                1.000000                   0.25
# 2                 0.75                0.333333                   0.25
# 3                 0.50                0.500000                   0.25
# 4                 1.00                0.250000                   0.25
# 5                 0.25                1.000000                   0.25
# 6                 0.25                1.000000                   0.25
# 7                 0.50                1.000000                   0.50
# 8                 0.00                0.000000                   0.25
# 9                 1.00                0.000000                   0.00
#
#    answer_consistency  overall_score
# 0            1.000000       0.450000
# 1            1.000000       0.700000
# 2            1.000000       0.626667
# 3            0.750000       0.600000
# 4            0.333333       0.546667
# 5            1.000000       0.700000
# 6            1.000000       0.700000
# 7            0.333333       0.546667
# 8            1.000000       0.410000
# 9            0.333333       0.386667


#RAGAS EVALUATIONS FOR CONTEXT COUNT = 1
#
# retrieved_single_context_list = [item["single_context"] for item in evaluation_data]
#
# llm_single_context_answer_list = [item["single_context_answer"] for item in evaluation_data]
# single_context_ground_truths = [[item["ground_truths"]] for item in evaluation_data]
#
# questions = [item["question"] for item in evaluation_data]
# ground_truths = [[item["ground_truths"]] for item in evaluation_data]
# data_length = len(evaluation_data)
# dataset = Dataset.from_dict({
#     "question": questions,
#     "contexts": retrieved_single_context_list,
#     "answer": llm_single_context_answer_list,
#     "ground_truths": ground_truths,
# })
# #
# # # Print the dataset
# print(dataset)
# results = evaluate(dataset)
# print(results)

#{'ragas_score': 0.2554, 'answer_relevancy': 0.9105, 'context_precision': 0.0917, 'faithfulness': 0.4600, 'context_recall': 0.6750}

#
#TONIC METRIC FOR CONTEXT COUNT = 1

# llm_evaluator = "gpt-4"
# score_calculator = RagScoresCalculator(llm_evaluator)
#
# question_list = [item["question"] for item in evaluation_data]
# retrieved_single_context_list = [item["single_context"] for item in evaluation_data]
# reference_single_context_answer_list = [item["single_context_answer"] for item in evaluation_data]
# llm_answer_list = [item["ground_truths"] for item in evaluation_data]
#
#
# single_context_batch_scores = score_calculator.score_batch(
#     question_list, reference_single_context_answer_list, llm_answer_list, retrieved_single_context_list
# )
#
#
# # mean of each score over the batch of question
# single_context_mean_scores = single_context_batch_scores.mean_scores
# # print(single_context_mean_scores)
#
# # dataframe that has the input data and the scores for each question
# single_context_scores_df = single_context_batch_scores.to_dataframe()
#
# print(single_context_scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [‘instruct’ models that have received further ...                      0.0
# 1  [corpus. They are made available under the Apa...                      5.0
# 2  [g) FalconConfig configuration class: FalconFo...                      4.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [supported in the Transformers library. If you...                      5.0
# 5  [ma - Instruct variants support infilling base...                      0.0
# 6  [s/2203.13474) CodeLlama (from MetaAI) release...                      0.0
# 7  [h and commercial use. Check out all Code Llam...                      1.0
# 8  [tokenizer = LlamaTokenizer.from_pretrained("/...                      0.0
# 9  [en_type_ids_from_sequences.example) BERT sequ...                      4.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                  0.0                     0.0                    0.0
# 1                  1.0                     1.0                    1.0
# 2                  0.0                     0.0                    0.0
# 3                  1.0                     1.0                    1.0
# 4                  1.0                     1.0                    1.0
# 5                  0.0                     0.0                    0.0
# 6                  0.0                     0.0                    0.0
# 7                  1.0                     1.0                    1.0
# 8                  0.0                     0.0                    0.0
# 9                  1.0                     0.0                    0.0
#
#    answer_consistency  overall_score
# 0            0.500000       0.100000
# 1            1.000000       1.000000
# 2            0.500000       0.260000
# 3            1.000000       1.000000
# 4            0.500000       0.900000
# 5            0.400000       0.080000
# 6            0.000000       0.000000
# 7            0.333333       0.706667
# 8            0.000000       0.000000
# 9            0.333333       0.426667


#RAGAS EVALUATIONS FOR CONTEXT COUNT = 2
# retrieved_two_context_list = [item["two_context"] for item in evaluation_data]
#
# llm_two_context_answer_list = [item["two_context_answer"] for item in evaluation_data]
# two_context_ground_truths = [[item["ground_truths"]] for item in evaluation_data]
#
# questions = [item["question"] for item in evaluation_data]
# ground_truths = [[item["ground_truths"]] for item in evaluation_data]
# data_length = len(evaluation_data)
# dataset = Dataset.from_dict({
#     "question": questions,
#     "contexts": retrieved_two_context_list,
#     "answer": llm_two_context_answer_list,
#     "ground_truths": ground_truths,
# })
# #
# # # Print the dataset
# print(dataset)
# results = evaluate(dataset)
# print(results)
#{'ragas_score': 0.4333, 'answer_relevancy': 0.9478, 'context_precision': 0.2111, 'faithfulness': 0.5433, 'context_recall': 0.6250}



#TONIC METRIC FOR CONTEXT COUNT = 2
# llm_evaluator = "gpt-4"
# score_calculator = RagScoresCalculator(llm_evaluator)
#
# question_list = [item["question"] for item in evaluation_data]
# retrieved_two_context_list = [item["two_context"] for item in evaluation_data]
# reference_two_context_answer_list = [item["two_context_answer"] for item in evaluation_data]
# llm_answer_list = [item["ground_truths"] for item in evaluation_data]
#
#
# two_context_batch_scores = score_calculator.score_batch(
#     question_list, reference_two_context_answer_list, llm_answer_list, retrieved_two_context_list
# )
#
# # mean of each score over the batch of question
# two_context_mean_scores = two_context_batch_scores.mean_scores
#
# # dataframe that has the input data and the scores for each question
# two_context_scores_df = two_context_batch_scores.to_dataframe()
#
# print(two_context_scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [‘instruct’ models that have received further ...                      0.0
# 1  [corpus. They are made available under the Apa...                      5.0
# 2  [g) FalconConfig configuration class: FalconFo...                      4.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [supported in the Transformers library. If you...                      4.5
# 5  [ma - Instruct variants support infilling base...                      0.0
# 6  [s/2203.13474) CodeLlama (from MetaAI) release...                      0.0
# 7  [h and commercial use. Check out all Code Llam...                      4.0
# 8  [tokenizer = LlamaTokenizer.from_pretrained("/...                      1.0
# 9  [en_type_ids_from_sequences.example) BERT sequ...                      4.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                  0.0                     0.0                    0.0
# 1                  0.5                     1.0                    0.5
# 2                  0.5                     1.0                    0.5
# 3                  0.5                     1.0                    0.5
# 4                  1.0                     0.5                    0.5
# 5                  0.5                     0.0                    0.5
# 6                  0.0                     0.0                    0.0
# 7                  0.5                     1.0                    0.5
# 8                  0.0                     0.0                    0.0
# 9                  1.0                     0.0                    0.0
#
#    answer_consistency  overall_score
# 0            0.500000       0.100000
# 1            1.000000       0.800000
# 2            1.000000       0.760000
# 3            1.000000       0.800000
# 4            0.333333       0.646667
# 5            0.400000       0.280000
# 6            0.000000       0.000000
# 7            0.000000       0.560000
# 8            0.000000       0.040000
# 9            0.333333       0.426667


#
# #RAGAS EVALUATIONS FOR CONTEXT COUNT = 8
# retrieved_eight_context_list = [item["eight_context"] for item in evaluation_data]
#
# llm_eight_context_answer_list = [item["eight_context_answer"] for item in evaluation_data]
#
# questions = [item["question"] for item in evaluation_data]
# ground_truths = [[item["ground_truths"]] for item in evaluation_data]
# data_length = len(evaluation_data)
# dataset = Dataset.from_dict({
#     "question": questions,
#     "contexts": retrieved_eight_context_list,
#     "answer": llm_eight_context_answer_list,
#     "ground_truths": ground_truths,
# })
# #
# # # Print the dataset
# print(dataset)
# results = evaluate(dataset)
# print(results)
#{'ragas_score': 0.2322, 'answer_relevancy': 0.9686, 'context_precision': 0.0757, 'faithfulness': 0.5917, 'context_recall': 0.7741}

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

print(eight_context_scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [‘instruct’ models that have received further ...                      5.0
# 1  [corpus. They are made available under the Apa...                      5.0
# 2  [g) FalconConfig configuration class: FalconFo...                      4.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [supported in the Transformers library. If you...                      5.0
# 5  [ma - Instruct variants support infilling base...                      5.0
# 6  [s/2203.13474) CodeLlama (from MetaAI) release...                      5.0
# 7  [h and commercial use. Check out all Code Llam...                      4.0
# 8  [tokenizer = LlamaTokenizer.from_pretrained("/...                      2.0
# 9  [en_type_ids_from_sequences.example) BERT sequ...                      2.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                0.125                0.000000                  0.000
# 1                0.125                1.000000                  0.125
# 2                0.500                0.500000                  0.375
# 3                0.250                0.500000                  0.125
# 4                0.875                0.285714                  0.250
# 5                0.375                0.333333                  0.125
# 6                0.125                1.000000                  0.125
# 7                0.375                0.666667                  0.250
# 8                0.000                0.000000                  0.000
# 9                1.000                0.000000                  0.000
#
#    answer_consistency  overall_score
# 0            1.000000       0.425000
# 1            1.000000       0.650000
# 2            1.000000       0.635000
# 3            0.500000       0.475000
# 4            1.000000       0.682143
# 5            1.000000       0.566667
# 6            1.000000       0.650000
# 7            0.333333       0.485000
# 8            1.000000       0.280000
# 9            0.333333       0.346667
