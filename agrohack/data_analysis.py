import os
import shutil
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

all_models = ["NASNetMobile", "ResNet50V2", "ResNet101V2",  # мы не берём аутсайдеров! но пусть будут...
              "Xception"]
sns.set_theme()
sns.set_style("white")
datagen = ImageDataGenerator(rescale=1 / 255)
for model_name, shape in zip(all_models, [224, 331, 331,
                                          331]):
    model_path = f"models/model-{model_name}.h5"
    test_generator = datagen.flow_from_directory(
        "dataset",
        target_size=(shape, shape),
        shuffle=False,
        class_mode='binary',
    )

    plots_dir = f"plots/{model_name}"
    os.makedirs(plots_dir, exist_ok=True)
    pred_path = f"predictions/predictions-{model_name}.json"
    if os.path.exists(pred_path):
        predictions = json.load(open(pred_path))
    else:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        # необходимо добавить, чтобы программа работала на локальном компьютере
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        model = load_model(model_path, custom_objects={'f1': metrics.f1_score})
        predictions = [float(x) for x in model.predict(test_generator, verbose=True)]
        json.dump(predictions, open(pred_path, 'w'))

    ts = []
    f1s = []
    best_f1 = 0
    best_threshold = 0
    for i in range(100):
        threshold = i / 100
        ts.append(threshold)
        y = [int(x > threshold) for x in predictions]
        f1 = metrics.f1_score(y, test_generator.classes)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        f1s.append(f1)
    plt.plot(ts, f1s)
    plt.savefig(f"{plots_dir}/threshold.png")

    conf = metrics.confusion_matrix([int(x > best_threshold) for x in predictions], test_generator.classes)
    conf = metrics.ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[0, 1])
    conf.plot()
    plt.axis("off")
    plt.gca().images[-1].colorbar.remove()
    plt.savefig(f"{plots_dir}/confmatrix.png")

    model_predictions = np.array([int(x > best_threshold) for x in predictions])
    indices = lambda z: [(y, i) for i, y in enumerate(z)]
    unindices = lambda z: [i for x, i in z]
    most_confident_positive = unindices(sorted(indices(predictions))[-5:])
    most_confident_negative = unindices(sorted(indices(predictions))[:5])
    wrong = model_predictions != test_generator.classes
    false_positive = np.where(np.logical_and(test_generator.classes == 1, wrong))[0].tolist()[:5]
    false_negative = np.where(np.logical_and(test_generator.classes == 0, wrong))[0].tolist()[:5]


    def write_images(out_dir, indexes):
        for index in indexes:
            fn = test_generator.filenames[index]
            in_file = f"dataset/{fn}"
            os.makedirs(f"{plots_dir}/{out_dir}", exist_ok=True)
            out_file = f"{plots_dir}/{out_dir}/{os.path.basename(fn)}"
            shutil.copy2(in_file, out_file)


    write_images("most_confident/positive", most_confident_positive)
    write_images("most_confident/negative", most_confident_negative)
    write_images("wrong/positive", false_positive)
    write_images("wrong/negative", false_negative)
