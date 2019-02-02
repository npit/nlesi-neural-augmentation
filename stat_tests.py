from scipy.stats import f_oneway
import os
import pickle
""" Module to perform statistical tests on classification results.
"""

# list of lists of directories, each containing a classification a results pickle
# each nests list should represent a group for which to test significance
directory_list = [
    ["/home/nik/work/iit/submissions/NLE-special/experiments/test/bag_pos_replace_wordnet_2-0.5_concept_frequencies_4_lda",
    "/home/nik/work/iit/submissions/NLE-special/experiments/test/tfidf_pos_replace_wordnet_2-0.5_concept_frequencies_4_lda",
    "/home/nik/work/iit/submissions/NLE-special/experiments/test/pos_replace_wordnet_2-0.5_concept_frequencies_4_lda",
    "/home/nik/work/iit/submissions/NLE-special/experiments/test/bag_pos_concat_wordnet_2-0.5_concept_frequencies_4_lda",
    "/home/nik/work/iit/submissions/NLE-special/experiments/test/tfidf_pos_concat_wordnet_2-0.5_concept_frequencies_4_lda",
     "/home/nik/work/iit/submissions/NLE-special/experiments/test/pos_concat_wordnet_2-0.5_concept_frequencies_4_lda"],

    [
"/home/nik/work/iit/submissions/NLE-special/experiments/test/tfidf_pos_concat_wordnet_2-0.5_concept_frequencies_4_lsa",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/bag_pos_concat_wordnet_2-0.5_concept_frequencies_4_lsa",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/pos_replace_wordnet_2-0.5_concept_frequencies_4_lsa",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/pos_concat_wordnet_2-0.5_concept_frequencies_4_lsa",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/tfidf_pos_replace_wordnet_2-0.5_concept_frequencies_4_lsa",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/bag_pos_replace_wordnet_2-0.5_concept_frequencies_4_lsa"
        ],
[
"/home/nik/work/iit/submissions/NLE-special/experiments/test/tfidf_pos_concat_wordnet_2-0.5_concept_frequencies_4_lida",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/tfidf_pos_replace_wordnet_2-0.5_concept_frequencies_4_lida",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/bag_pos_concat_wordnet_2-0.5_concept_frequencies_4_lida",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/pos_replace_wordnet_2-0.5_concept_frequencies_4_lida",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/bag_pos_replace_wordnet_2-0.5_concept_frequencies_4_lida",
"/home/nik/work/iit/submissions/NLE-special/experiments/test/pos_concat_wordnet_2-0.5_concept_frequencies_4_lida"

    ]
]
# an evaluation measure against which to extract significance scores
evaluation_metric = "f1-score"
# multiclass aggregation type, for single-label classification results.
# is ignored for multilabel metrics (e.g. mean average precision)
multiclass_aggregation = "macro"
# which performance statistic to use. Will probably always use the default.
evaluation_stat = "mean"
# list of which run type(s) to parse, e.g. 'run', 'random' or 'majority'. Will probably always use the default.
run_types = ['run']
# significance tests to use
tests = [f_oneway]
# kruskal
# wilcoxon
# use statsmodels package?


def load_group_results(dir_list, metric, evaluation_stat, aggr=None, run_type='run'):
    group_results = []
    for direc in dir_list:
        results_file = os.path.join(direc, "results", "results.pickle")
        print("Reading results:", results_file)
        with open(results_file, "rb") as f:
            df = pickle.load(f)

        score = df[run_type][metric]
        if aggr is not None:
            score = score[aggr]
        score = score[evaluation_stat]
        print("Incorporating score:", score)
        if type(score) == list:
            group_results.extend(score)
        else:
            group_results.append(score)
    return group_results


if __name__ == '__main__':
    results = []
    for dir_group in directory_list:
        if evaluation_metric in ["ap", "auc"]:
            multiclass_aggregation = None
        results.append(load_group_results(dir_group, evaluation_metric, evaluation_stat, multiclass_aggregation))

    if not results:
        print("No results loaded")
        exit(1)
    print("Applying tests on data:", results)
    print()
    for test in tests:
        print(test.__name__,":", test(*results))

