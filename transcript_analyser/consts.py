VALIDATION_ERR = "Could not validate credentials!"
N_GRAMS_MIN_DESC = 'Minimum number of the words that the keyphrases should have.'
N_GRAMS_MAX_DESC = 'Maximum number of the words that the keyphrases should have.'
N_KEYPHRASES_DESC = 'Number of the keyphrases to be returned.'
TYPE_NOT_SUPPORTED = 'Instance provided by user is not supported!'

FORBIDDEN_STATUS_CODE = 403
N_GRAMS_MIN = 1
N_GRAMS_MAX = 3
N_KEYPHRASES = 5
MIN_WORD_LEN = 1
CACHING_TIME_TOLERANCE = 10

PER_CLUSTER_RESULTS_DESC = 'Whether to get the results per cluster, per_cluster_results = [true, false].'
CLUSTERING_ALGORITHM_DESC = "The clustering algorithm that you want to be used for the clustering phase, clustering_algorithm = ['kmeans', 'louvain']."
DO_CLUSTER_DESC = 'Whether to cluster the results, do_cluster = [false, true].'
GET_GRAPH_BACKBONE_DESC = "Whether to get the sentence graph backbone when analyzing the sentences and their most important ones (which are nodes in a sense). get_graph_backbone = [false, true]"
REMOVE_ENTAILED_SENTENCES_DESC = "Whether to remove the sentences which can be entailed from the other sentences or not, it can be assigned by one of the following values, remove_entailed_sentences = [false, true]."
FILTER_BACKCHANNELS_DESC = "Whether to filter the backchannels or not as a preprocessing step, it can be set to one of the following values, filter_backchannels = [false, true]."
OUTPUT_TYPE_DESC = "The type of the output you want to get. it can be one of the following optionts, output_types = ['SENTENCE', 'WORD']."
JOB_DOES_NOT_EXISTS = "There is no job with this job id."
