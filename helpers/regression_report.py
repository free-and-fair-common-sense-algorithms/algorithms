import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


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


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(np.log(y_pred[i] + 1) - np.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def adj_r2(model, x_test, y_test):
    return 1 - (1 - model.score(x_test, y_test)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)


def evaluate(model, x_test, y_test, encoder_step_label):
    preds = model.predict(x_test)
    errors = abs(preds - y_test.values.ravel())
    mape = 100 * (errors / y_test.values.ravel())
    print('{:<30}{:0.5f}'.format('Accuracy:', 100 - np.mean(mape)))
    print('{:<30}{:0.6f}'.format('MSE', mean_squared_error(y_test, preds)))
    print('{:<30}{:0.6f}'.format('RMSE', np.sqrt(mean_squared_error(y_test, preds))))
    print('{:<30}{:0.6f}'.format('RMSLE', rmsle(y_test.values.ravel(), model.predict(x_test))))
    print('{:<30}{:0.6f}'.format('MAE', mean_absolute_error(y_test, preds)))
    print('{:<30}{:0.6f}'.format('R2', r2_score(y_test, preds)))
    print('{:<30}{:0.6f}'.format('Adj R2', adj_r2(model, x_test, y_test)))
    print('{:<30}{:0.6f}'.format('Median', pd.Series(preds).median()))
    print('{:<30}{:0.6f}'.format('Avg absolute error (degrees)', np.mean(errors)))
    print('')
    print('-' * 40)
    print('Feature Importances:')
    feature_importances = list(model.named_steps['estimator'].regressor.feature_importances_)
    features = get_features(model, encoder_step_label)
    line_len = max(len(f) for f in features) + 1
    feature_importances = sorted(zip(features, feature_importances), key=lambda x: -x[1])
    for feature, importance in feature_importances:
        print(f'{feature:<{line_len}}{importance:0.6f}')

    pd.Series(preds).hist(bins=50, color='black')
    plt.title('Predictions', loc='left')
    return plt.show()
