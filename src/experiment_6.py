# coding=utf-8

from Expreriment import Experiment6

HEALTHY_STRING = "HEALTHY"
DISEASE_STRING = "DISEASE"

# Utilizzando questo script è possibile eseguire la combinazione tra i migliori task di categoria
# Il pratica utilizzando i risultati ottenuti dall'esperimento 4 inizialmente, incrementando man mano i task selezionati
# come combinazione inizale è possibile selezionare la miglior combinazione, ad esempio:
# selezioniamo le matrici come task iniziale:
# 1) combination = {'matrix': {HEALTHY_STRING:  ['_m1.'],
#                               DISEASE_STRING: ['_m2.']}}
# 2) combination = {'matrix': {HEALTHY_STRING:  ['_m1.', '_cs.'],
#                               DISEASE_STRING: ['_m2.', '_ts.']}}
# 3) combination = {'matrix': {HEALTHY_STRING:  ['_m1.', '_cs.', '_s2.'],
#                               DISEASE_STRING: ['_m2.', '_ts.', '_w.']}}
# etc... alla fine si otterranno tutte le combinazioni di tutti i task migliori per ogni categoria con i rispettivi risultati.
# Ovviamente bisogna avere l'accortezza di modificare il file di salvataggio ad ogni test, così da mantenere i risultati
# precedenti in memoria: matrix_test, matrix_test_2, matrix_test_3, ...


def main():
    path_dictionary = {'drawing': 'experiment_result/experiment_2/drawing.txt',
                       'matrix': 'experiment_result/experiment_2/matrix.txt',
                       'solid_lines': 'experiment_result/experiment_2/solid_lines.txt',
                       'trial': 'experiment_result/experiment_2/trial.txt',
                       'word_easy': 'experiment_result/experiment_2/word_easy.txt',
                       'word_hard': 'experiment_result/experiment_2/word_hard.txt'}
    saving_path = 'resource/test_dictionary_file.csv'
    combination = {'matrix': {HEALTHY_STRING: ['_s2.', '_m1.'],
                              DISEASE_STRING: ['_w.', '_m2.']}}
    experiment = Experiment6(None, path_dictionary, 2500, 50)
    experiment.start_shift_selection(combination, "matrix_test")


if __name__ == '__main__':
    main()
