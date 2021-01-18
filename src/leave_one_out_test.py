from Expreriment.LeaveOneOutExperiment import LeaveOneOutExperiment


def main():
    leave_one_out_experiment = LeaveOneOutExperiment("Dataset")
    leave_one_out_experiment.start_experiment(10, 1)
    print(leave_one_out_experiment.get_metrics())


if __name__ == "__main__":
    main()
