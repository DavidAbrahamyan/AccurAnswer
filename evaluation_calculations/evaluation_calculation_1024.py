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
gpt_with_documentation_evaluation_data_path = "evaluation_files/evaluation_1024_openai.json"
evaluation_data = {}
if os.path.exists(gpt_with_documentation_evaluation_data_path):
    with open(gpt_with_documentation_evaluation_data_path, "r") as evaluation_data_file:
        evaluation_data = json.load(evaluation_data_file)

print(evaluation_data)

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
print("RAGAS EVALUATIONS FOR CONTEXT COUNT = 4")
print(results)
#{'ragas_score': 0.3098, 'answer_relevancy': 0.9778, 'context_precision': 0.1144, 'faithfulness': 0.7850, 'context_recall': 0.5333}

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
print("TONIC METRIC FOR CONTEXT COUNT = 4")
print(scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [Link: https://huggingface.co/docs/transformer...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  [tiiuae/falcon-7b architecture. FalconModel (/...                      4.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [and consistently rank highly in the OpenLLM l...                      4.5
# 5  [following ability for programming tasks. We p...                      5.0
# 6  [following ability for programming tasks. We p...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      4.0
# 8  ['Whether or not the default system prompt for...                      4.0
# 9  [in a unified manner using the exact same mode...                      4.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                 0.25                1.000000                   0.25
# 1                 0.25                1.000000                   0.25
# 2                 0.25                0.000000                   0.25
# 3                 0.75                0.666667                   0.50
# 4                 0.50                0.500000                   0.25
# 5                 0.75                0.666667                   0.50
# 6                 0.25                1.000000                   0.25
# 7                 0.25                1.000000                   0.50
# 8                 0.25                1.000000                   0.25
# 9                 0.50                0.500000                   0.25
#
#    answer_consistency  overall_score
# 0            1.000000       0.700000
# 1            1.000000       0.700000
# 2            1.000000       0.460000
# 3            1.000000       0.783333
# 4            0.666667       0.563333
# 5            1.000000       0.783333
# 6            1.000000       0.700000
# 7            1.000000       0.710000
# 8            1.000000       0.660000
# 9            0.666667       0.543333


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
print(dataset)
results = evaluate(dataset)
print("RAGAS EVALUATIONS FOR CONTEXT COUNT = 1")
print(results)

#{'ragas_score': 0.3655, 'answer_relevancy': 0.9486, 'context_precision': 0.1393, 'faithfulness': 0.6250, 'context_recall': 0.9000}


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
# print(single_context_mean_scores)

# dataframe that has the input data and the scores for each question
single_context_scores_df = single_context_batch_scores.to_dataframe()
print("TONIC METRIC FOR CONTEXT COUNT = 1")
print(single_context_scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [Link: https://huggingface.co/docs/transformer...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      4.0
# 2  [tiiuae/falcon-7b architecture. FalconModel (/...                      3.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [and consistently rank highly in the OpenLLM l...                      5.0
# 5  [following ability for programming tasks. We p...                      5.0
# 6  [following ability for programming tasks. We p...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      5.0
# 8  ['Whether or not the default system prompt for...                      1.0
# 9  [in a unified manner using the exact same mode...                      0.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                  1.0                     1.0                    1.0
# 1                  1.0                     1.0                    1.0
# 2                  0.0                     0.0                    0.0
# 3                  1.0                     1.0                    1.0
# 4                  1.0                     1.0                    1.0
# 5                  1.0                     1.0                    1.0
# 6                  1.0                     1.0                    1.0
# 7                  1.0                     1.0                    1.0
# 8                  0.0                     0.0                    0.0
# 9                  0.0                     0.0                    0.0
#
#    answer_consistency  overall_score
# 0                 1.0           1.00
# 1                 1.0           0.96
# 2                 0.5           0.22
# 3                 1.0           1.00
# 4                 0.8           0.96
# 5                 1.0           1.00
# 6                 1.0           1.00
# 7                 1.0           1.00
# 8                 0.0           0.04
# 9                 0.0           0.00


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
print(dataset)
results = evaluate(dataset)
print("RAGAS EVALUATIONS FOR CONTEXT COUNT = 2")
print(results)
#{'ragas_score': 0.2880, 'answer_relevancy': 0.9785, 'context_precision': 0.0978, 'faithfulness': 0.8633, 'context_recall': 0.6750}



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
print("TONIC METRIC FOR CONTEXT COUNT = 2")
print(two_context_scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [Link: https://huggingface.co/docs/transformer...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  [tiiuae/falcon-7b architecture. FalconModel (/...                      3.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [and consistently rank highly in the OpenLLM l...                      4.5
# 5  [following ability for programming tasks. We p...                      5.0
# 6  [following ability for programming tasks. We p...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      5.0
# 8  ['Whether or not the default system prompt for...                      5.0
# 9  [in a unified manner using the exact same mode...                      1.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                  0.5                     1.0                    0.5
# 1                  0.5                     1.0                    0.5
# 2                  0.5                     0.0                    0.0
# 3                  1.0                     1.0                    1.0
# 4                  1.0                     0.5                    0.5
# 5                  1.0                     1.0                    1.0
# 6                  0.5                     1.0                    0.5
# 7                  0.5                     1.0                    0.5
# 8                  0.5                     1.0                    0.5
# 9                  0.5                     0.0                    0.0
#
#    answer_consistency  overall_score
# 0            1.000000       0.800000
# 1            1.000000       0.800000
# 2            1.000000       0.420000
# 3            1.000000       1.000000
# 4            0.666667       0.713333
# 5            1.000000       1.000000
# 6            1.000000       0.800000
# 7            1.000000       0.800000
# 8            1.000000       0.800000
# 9            0.000000       0.140000


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
print(dataset)
results = evaluate(dataset)
print("RAGAS EVALUATIONS FOR CONTEXT COUNT = 8")
print(results)
#{'ragas_score': 0.2084, 'answer_relevancy': 0.9833, 'context_precision': 0.0651, 'faithfulness': 0.8783, 'context_recall': 0.6000}


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
print("TONIC METRIC FOR CONTEXT COUNT = 8")
print(eight_context_scores_df)

#                                    retrieved_context  answer_similarity_score  \
# 0  [Link: https://huggingface.co/docs/transformer...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  [tiiuae/falcon-7b architecture. FalconModel (/...                      3.0
# 3  [Vocabulary size of the Falcon model. Defines ...                      5.0
# 4  [and consistently rank highly in the OpenLLM l...                      5.0
# 5  [following ability for programming tasks. We p...                      5.0
# 6  [following ability for programming tasks. We p...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      5.0
# 8  ['Whether or not the default system prompt for...                      4.0
# 9  [in a unified manner using the exact same mode...                      2.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                0.125                1.000000                  0.125
# 1                0.125                1.000000                  0.125
# 2                0.250                1.000000                  0.250
# 3                0.500                0.500000                  0.250
# 4                0.250                0.500000                  0.125
# 5                0.625                0.400000                  0.250
# 6                0.125                1.000000                  0.125
# 7                0.625                0.600000                  0.375
# 8                0.375                0.333333                  0.125
# 9                0.500                0.750000                  0.375
#
#    answer_consistency  overall_score
# 0            1.000000       0.650000
# 1            1.000000       0.650000
# 2            1.000000       0.620000
# 3            1.000000       0.650000
# 4            0.666667       0.508333
# 5            1.000000       0.655000
# 6            1.000000       0.650000
# 7            1.000000       0.720000
# 8            1.000000       0.526667
# 9            1.000000       0.605000
