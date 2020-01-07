import itertools
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix


matplotlib_styles = {
    'axes.axisbelow': True,
    'axes.edgecolor': '0',
    'axes.facecolor': 'white',
    'axes.formatter.useoffset': False,
    'axes.grid': True,
    'axes.labelcolor': '.15',
    'axes.linewidth': 1,
    'axes.titlepad': 15,
    'axes.titlesize': 16,
    'figure.figsize': [6, 6],
    'figure.facecolor': 'white',
    'font.family': ['Source Code Pro', 'monospace'],
    'grid.color': '.95',
    'grid.linestyle': '-',
    'image.cmap': 'Greys',
    'legend.frameon': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'lines.solid_capstyle': 'round',
    'lines.linewidth': 2,
    'lines.scale_dashes': False,
    'lines.dashed_pattern': [6, 6],
    'text.color': '.15',
    'xtick.color': '.15',
    'xtick.direction': 'out',
    'xtick.major.size': 0,
    'xtick.minor.size': 0,
    'ytick.color': '.15',
    'ytick.direction': 'out',
    'ytick.major.size': 0,
    'ytick.minor.size': 0
}
plt.rcParams.update(matplotlib_styles)


def get_features(model, encoder_step_label):
    features = []
    for col, enc in model.named_steps[encoder_step_label].encodings.items():
        for c in enc.classes_:
            features.append(f'{col}={c}')
    return features


def performance(model, x_test, y_test, encoder_step_label, cross_validate=True, x_train=None, y_train=None, pos_label=0, average='binary', beta=1):
    encoder = LabelEncoder()
    y_test = encoder.fit_transform(y_test)
    y_pred = encoder.transform(model.predict(x_test))
    if cross_validate and x_train and y_train:
        scores = cross_val_score(model, x_train, y_train, cv=5)
    precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=pos_label, average=average, beta=beta)

    features = get_features(model, encoder_step_label)
    line_len = max(len(f) for f in features) + 1
    feature_importances = model.named_steps['classifier'].classifier.feature_importances_
    print('{:<11}{:0.6f}'.format('Accuracy:', accuracy_score(y_test, y_pred)))
    print('{:<11}{:0.6f}'.format('Recall:', recall))
    print('{:<11}{:0.6f}'.format('F-beta:', fbeta_score))
    print('{:<11}{:0.6f}'.format('Precision:', precision))
    if cross_validate:
        print('{:<20}{:0.6f} (+/- {:0.2f})'.format('Cross Val Accuracy:', scores.mean(), scores.std() * 2))
    print('')
    print('-' * 40)
    print('Feature Importances:')
    feature_importances = sorted(zip(features, feature_importances), key=lambda x: -x[1])
    for feature, importance in feature_importances:
        print(f'{feature:<{line_len}}{importance:0.6f}')
    return


def roc_curve(model, x_test, y_test):
    encoder = LabelEncoder()
    y_test = encoder.fit_transform(y_test)
    preds = [list(response.values())[1] for response in model.predict_proba(x_test)]
    false_positive_rate, true_positive_rate, threshold = roc(y_test, preds)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots()
    ax.plot(false_positive_rate, true_positive_rate, 'black', label='AUC = {:0.3f}'.format(roc_auc))
    ax.plot([0, 1], [0, 1], color='gray', ls='--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_title('ROC curve', loc='left')
    ax.legend(loc='lower right')
    return plt.show()


def plot_confusion_matrix(y_test, y_pred, class_names, normalize=False, fig_size=(6, 6)):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(cm, interpolation='nearest')
    plt.colorbar(ax.matshow(cm), fraction=0.046, pad=0.04)
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, '{:0.4f}'.format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > threshold else 'black')
        else:
            ax.text(j, i, '{:,}'.format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > threshold else 'black')

    accuracy = np.trace(cm) / float(np.sum(cm))
    ax.set_title('Confusion matrix', loc='left')
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([1.5, -0.5])
    ax.set_ylabel('True label')
    ax.set_xlabel(f'Predicted label\naccuracy={accuracy:0.4f}; misclass={ 1 - accuracy:0.4f}')
    ax.grid(b=None)
    plt.tight_layout()
    return plt.show()
