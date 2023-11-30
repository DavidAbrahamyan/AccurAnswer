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
gpt_with_documentation_evaluation_data_path = "evaluation_files/evaluation_1024_openai_rerank.json"
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
#{'ragas_score': 0.3816, 'answer_relevancy': 0.9647, 'context_precision': 0.1446, 'faithfulness': 0.8067, 'context_recall': 0.7750}

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
# 0  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  ['experimental feature, subject to breaking AP...                      4.0
# 3  ['Vocabulary size of the Falcon model. Defines...                      5.0
# 4  ['and consistently rank highly in the OpenLLM ...                      4.5
# 5  ['following ability for programming tasks. We ...                      5.0
# 6  ['following ability for programming tasks. We ...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      4.0
# 8  ["is that when decoding a sequence, if the fir...                      4.0
# 9  ['List of IDs.  (#transformers.AlbertTokenizer...                      4.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                 0.25                1.000000                   0.25
# 1                 0.25                1.000000                   0.25
# 2                 0.50                1.000000                   0.50
# 3                 0.75                0.666667                   0.50
# 4                 0.25                1.000000                   0.25
# 5                 0.75                0.333333                   0.25
# 6                 0.25                1.000000                   0.25
# 7                 1.00                0.500000                   0.50
# 8                 1.00                0.250000                   0.25
# 9                 1.00                1.000000                   1.00
#
#    answer_consistency  overall_score
# 0            1.000000       0.700000
# 1            1.000000       0.700000
# 2            1.000000       0.760000
# 3            1.000000       0.783333
# 4            0.666667       0.613333
# 5            1.000000       0.666667
# 6            1.000000       0.700000
# 7            1.000000       0.760000
# 8            1.000000       0.660000
# 9            1.000000       0.960000


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
#{'ragas_score': 0.4842, 'answer_relevancy': 0.9712, 'context_precision': 0.1988, 'faithfulness': 0.8500, 'context_recall': 0.9750}


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
# 0  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  ['experimental feature, subject to breaking AP...                      4.0
# 3  ['Vocabulary size of the Falcon model. Defines...                      5.0
# 4  ['and consistently rank highly in the OpenLLM ...                      5.0
# 5  ['following ability for programming tasks. We ...                      5.0
# 6  ['following ability for programming tasks. We ...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      5.0
# 8  ["is that when decoding a sequence, if the fir...                      5.0
# 9  ['List of IDs.  (#transformers.AlbertTokenizer...                      4.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                  1.0                     1.0                    1.0
# 1                  1.0                     1.0                    1.0
# 2                  1.0                     1.0                    1.0
# 3                  1.0                     1.0                    1.0
# 4                  1.0                     1.0                    1.0
# 5                  1.0                     1.0                    1.0
# 6                  1.0                     1.0                    1.0
# 7                  1.0                     1.0                    1.0
# 8                  1.0                     1.0                    1.0
# 9                  1.0                     1.0                    1.0
#
#    answer_consistency  overall_score
# 0                 1.0           1.00
# 1                 1.0           1.00
# 2                 1.0           0.96
# 3                 1.0           1.00
# 4                 0.8           0.96
# 5                 1.0           1.00
# 6                 1.0           1.00
# 7                 1.0           1.00
# 8                 1.0           1.00
# 9                 1.0           0.96


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
#{'ragas_score': 0.4031, 'answer_relevancy': 0.9663, 'context_precision': 0.1547, 'faithfulness': 0.8517, 'context_recall': 0.8000}



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
# 0  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  ['experimental feature, subject to breaking AP...                      4.0
# 3  ['Vocabulary size of the Falcon model. Defines...                      4.0
# 4  ['and consistently rank highly in the OpenLLM ...                      4.0
# 5  ['following ability for programming tasks. We ...                      5.0
# 6  ['following ability for programming tasks. We ...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      4.0
# 8  ["is that when decoding a sequence, if the fir...                      2.0
# 9  ['List of IDs.  (#transformers.AlbertTokenizer...                      4.0
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                  0.5                     1.0                    0.5
# 1                  0.5                     1.0                    0.5
# 2                  1.0                     1.0                    1.0
# 3                  1.0                     1.0                    1.0
# 4                  0.5                     1.0                    0.5
# 5                  1.0                     0.5                    0.5
# 6                  0.5                     1.0                    0.5
# 7                  1.0                     1.0                    1.0
# 8                  1.0                     0.5                    0.5
# 9                  1.0                     1.0                    1.0
#
#    answer_consistency  overall_score
# 0            1.000000       0.800000
# 1            1.000000       0.800000
# 2            1.000000       0.960000
# 3            1.000000       0.960000
# 4            0.666667       0.693333
# 5            1.000000       0.800000
# 6            1.000000       0.800000
# 7            1.000000       0.960000
# 8            0.666667       0.613333
# 9            1.000000       0.960000


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
#{'ragas_score': 0.2670, 'answer_relevancy': 0.9699, 'context_precision': 0.0875, 'faithfulness': 0.7917, 'context_recall': 0.7972}


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
# 0  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 1  ['Link: https://huggingface.co/docs/transforme...                      5.0
# 2  ['experimental feature, subject to breaking AP...                      4.0
# 3  ['Vocabulary size of the Falcon model. Defines...                      5.0
# 4  ['and consistently rank highly in the OpenLLM ...                      5.0
# 5  ['following ability for programming tasks. We ...                      2.0
# 6  ['following ability for programming tasks. We ...                      5.0
# 7  ['odellama org. here (https://huggingface.co/m...                      4.0
# 8  ["is that when decoding a sequence, if the fir...                      4.0
# 9  ['List of IDs.  (#transformers.AlbertTokenizer...                      4.5
#
#    retrieval_precision  augmentation_precision  augmentation_accuracy  \
# 0                0.125                1.000000                  0.125
# 1                0.125                1.000000                  0.125
# 2                0.375                0.666667                  0.250
# 3                0.500                0.750000                  0.375
# 4                0.250                0.500000                  0.125
# 5                0.750                0.166667                  0.125
# 6                0.125                1.000000                  0.125
# 7                0.625                0.200000                  0.125
# 8                1.000                0.500000                  0.500
# 9                1.000                1.000000                  1.000
#
#    answer_consistency  overall_score
# 0            1.000000       0.650000
# 1            1.000000       0.650000
# 2            1.000000       0.618333
# 3            1.000000       0.725000
# 4            0.833333       0.541667
# 5            1.000000       0.488333
# 6            1.000000       0.650000
# 7            1.000000       0.550000
# 8            1.000000       0.760000
# 9            1.000000       0.980000

