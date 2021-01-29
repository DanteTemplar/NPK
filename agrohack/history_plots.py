import pickle
from matplotlib import pyplot as plt

all_models = ["NASNetMobile", "ResNet50V2", "ResNet101V2", "Xception"]
metrics = {}
for model in all_models:
    logs = pickle.load(open(f"stories/history-{model}.pkl", 'rb'))
    for metric, values in logs.items():
        if metric not in metrics:
            metrics[metric] = {}
        metrics[metric][model] = values
for metric, models in metrics.items():
    plt.xlabel("Итерация")
    plt.ylabel(metric)
    for model_name, data in models.items():
        plt.plot(data, label=model_name)
    plt.legend()
    plt.savefig(f"plots/{metric}.png")
    plt.show()
