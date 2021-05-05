import sys
sys.path.append("..")

from Expreriment.Experiment_5 import Experiment5


# Questo esperimento effettua un test leave one out completo includendo tutti i task per generare e testare il modello.

def main():
    leave_one_out_experiment = Experiment5("Dataset")
    leave_one_out_experiment.start_experiment()


if __name__ == "__main__":
    main()
