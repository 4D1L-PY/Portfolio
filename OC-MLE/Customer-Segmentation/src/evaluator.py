from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import *
from sklearn.dummy import DummyRegressor, DummyClassifier
from src.datacleaner import format_run_time
from src.analyzer.univariate import pd, np, plt, percentage_change
import datetime
import joblib
import pickle


########################################################################################################################
#                                                  Metrics                                                             #
########################################################################################################################


def print_score_results(scores, set_type='test'):
    if set_type is 'test':
        print('--------------------------')
        print('Testing set performances :')
    elif set_type is 'train':
        print('--------------------------------')
        print('Training set performances (CV) :')
    else:
        print('-----------------------')
        print('Training performances :')        
    for score_label, score_value in scores.items():
        print(f'- {score_label.upper()} = {score_value}')
        
        
def get_model_scores(model_fit, x_test, y_test, scorer, precision=3, verbose=True): # TO DO : Add other metrics
    """
    """
    # Model name
    model_label = model_name(model_fit)
    # Predict on testing set
    y_pred = model_fit.predict(x_test)
    # Build scorer dictionary
    testing_set_scores = {scorer_label: 0 for scorer_label in scorer}
    # Compute metric
    for scorer_label in testing_set_scores.keys():
        # Regression metrics
        if scorer_label == 'mse':
            model_score = round(mean_squared_error(y_test, y_pred, squared=True), precision)
        elif scorer_label == 'rmse':
            model_score = round(mean_squared_error(y_test, y_pred, squared=False), precision)
        elif scorer_label == 'r2':
            model_score = round(r2_score(y_test, y_pred), precision)
        # Classification metrics
        elif scorer_label == 'accuracy':
            model_score = round(accuracy_score(y_test, y_pred), precision)
        # Clustering metrics
        elif scorer_label == 'adjusted_rand_score':
            model_score = round(adjusted_rand_score(y_test, y_pred), precision)
        # Update scorer dictionary with matched metric(s)
        testing_set_scores[scorer_label] = model_score
    # Display result for each scorer
    if verbose:
        print(f'{model_label}')
        print_score_results(testing_set_scores, set_type='test')
    return testing_set_scores


def train_gridsearch(data, model, param_grid, metric, k=10, p=3, v=True):
    # Model name
    model_label = model_name(model)
    # Get training & testing data
    x_train, y_train = data['train']
    x_test, y_test = data['test']
    # Define refit condition (first metric if evaluationg multiple metrics else False)
    refit_cond = metric[0] if type(metric) is list else True
    # Build grid search
    gridsearch = GridSearchCV(model, param_grid, cv=k, scoring=metric, refit=refit_cond)
    # Time the model training
    start_training = datetime.datetime.now()
    # Train model with grid search
    gridsearch.fit(x_train, y_train)
    end_training = datetime.datetime.now()
    # Compute training time
    training_time = end_training - start_training
    # Format training time
    training_time_str = format_run_time(training_time)
    # Trained_model
    trained_model = gridsearch.best_estimator_
    # Get scores from cross validation
    cv_scores = {}
    if type(metric) is list:
        for scorer_label in metric:
            if scorer_label.startswith('neg'):
                formatted_label = "".join([w[0] for w in scorer_label.replace('neg_', '').split('_')])
                formatted_score = round(np.abs(gridsearch.cv_results_[f'mean_test_{scorer_label}'])[0], p)
                cv_scores[formatted_label] = formatted_score
            else:
                cv_scores[scorer_label] = round(gridsearch.cv_results_[f'mean_test_{scorer_label}'][0], p)
    else:
        cv_scores[metric] = round(gridsearch.cv_results_[f'mean_test_score'][0], p)
    # Get scores from testing set
    testing_set_scores = get_model_scores(trained_model, x_test, y_test, list(cv_scores.keys()), p, v)
    # Display cross validation mean scores
    if v:
        print_score_results(cv_scores, set_type='train')
    # Build model dictionary which contains GridSearchCV & model instances (with model name)
    model_data = {'gs': gridsearch,                                 # GridSearchCV trained instance
                  'model': trained_model,                           # Model trained instance
                  'model_name': model_label}                        # Model name
    # Build additional evaluation data dictionary
    additional_evaluation_data = {'time': training_time_str,        # Training time
                                  'n_features': x_train.shape[1],   # Selected features
                                  'learning_potential': None}       # Learning potential
    # Build results dictionary (merge dictionnaries)
    results = dict(**model_data, **testing_set_scores, **additional_evaluation_data)
    return results


def gram_matrix(x_train, gamma):
    kmatrix = pairwise.rbf_kernel(x_train, gamma)
    return kmatrix


def plot_gram_matrix(x_train, gamma, n_reduce=None):
    kmatrix = gram_matrix(x_train, gamma)
    if n_reduce is not None:
        kmatrix = kmatrix[:n_reduce, :n_reduce]
    plt.pcolor(kmatrix, cmap=plt.cm.PuRd)
    plt.colorbar()
    plt.xlim([0, kmatrix.shape[0]])
    plt.ylim([0, kmatrix.shape[0]])
    plt.gca().invert_yaxis()
    plt.gca().xaxis.tick_top()
    plt.show()

    
def get_confusion_matrix(y_test, y_predicted, class_labels=None):
    """
    cf :
    - https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/
      4308256-evaluez-un-algorithme-de-classification-qui-retourne-des-valeurs-binaires
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    cm = confusion_matrix(y_test, y_predicted, labels=class_labels)
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp, cm


def roc_decision_rule(decision_rate, thr, decision_threshold):
    """
    cf :
    - https://openclassrooms.com/fr/courses/4297211-evaluez-les-performances-dun-modele-de-machine-learning/
      4308261-evaluez-un-algorithme-de-classification-qui-retourne-des-scores
    """
    idx = np.min(np.where(decision_rate > decision_threshold))
    threshold = thr[idx]
    return threshold, idx


def elbow_criterion(total_inertia, threshold=0.25):
    """
    Find total components/clusters number based on Elbow criterion :
    (cf : https://en.wikipedia.org/wiki/Elbow_method_(clustering))
    """
    features_nb = len(total_inertia)
    var_cumsum = total_inertia.cumsum()
    # Compute variations ratio from cumulated explained variance values
    variations = [abs(percentage_change(var_cumsum[i + 1], x)) for i, x in enumerate(var_cumsum) if i + 1 < features_nb]
    # Get total components selected
    if threshold is 'min':
        n_selected = variations.index(min(variations)) + 1
    elif type(threshold) is float:
        variations = np.array(variations)
        n_selected = np.min(np.where(variations <= np.quantile(variations, q=threshold))) + 1
    return n_selected
    
    
########################################################################################################################
#                                              Visualizations                                                          #
########################################################################################################################


# Regularization path

def plot_regularization_path(alphas, coefs, features_labels, n_features_labels=None, legend_size='medium'):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Coefficients')
    plt.legend(features_labels[:n_features_labels] if n_features_labels is not None else features_labels,
               loc='upper right',
               fontsize=legend_size)
    plt.show()


# ROC curve
    
def plot_roc_curve(y_test, y_predicted, curve_color='coral', font_size=14, x_lim=(0, 1), y_lim=(0, 1)):
    """
    cf :
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    """
    fpr, tpr, thr = roc_curve(y_test, y_predicted)
    auroc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUROC : {auroc:.2f}', color=curve_color, lw=2)
    # Random classifier line plot
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('1 - specificity (FPR)', fontsize=font_size)
    plt.ylabel('Recall (TPR)', fontsize=font_size)
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


# LIFT curve
    
def plot_lift_curve(y_test, y_predicted, curve_color='blue', font_size=14, x_lim=(0, 1), y_lim=(0, 1)):
    fpr, tpr, thr = roc_curve(y_test, y_predicted)
    percentage_sample = [(n + 1) / len(tpr) for n in range(len(tpr))]
    plt.plot(fpr, percentage_sample, color=curve_color, lw=2)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Percentage of sample', fontsize=font_size)
    plt.ylabel('Recall (TPR)', fontsize=font_size)
    plt.title('LIFT curve')

    
# Training curve    

def plot_validation_curve(model, x_train, y_train, h_param, h_range, k=10, log_scale=False, scorer=None):
    train_score, val_score = validation_curve(model,
                                              x_train,
                                              y_train,
                                              h_param,
                                              h_range,
                                              cv=k,
                                              scoring=scorer)
    validation_curve_data = {'train': train_score,
                             'test': val_score}
    for legend, scores in validation_curve_data.items():
        if log_scale:
            plt.semilogx(h_range, np.abs(scores.mean(axis=1)), label=legend)
        else:
            plt.plot(h_range, np.abs(scores.mean(axis=1)), label=legend)
    plt.xlabel(h_param, labelpad=20)
    plt.ylabel('score', labelpad=20)
    plt.legend()
    plt.show()
    

# Learning curve 
    
def plot_learning_curve(model, x_train, y_train, train_sizes_ratio=np.linspace(0.1, 1.0, 10), k=10, scorer=None):
    N, train_score, val_score = learning_curve(model,
                                               x_train,
                                               y_train,
                                               train_sizes=train_sizes_ratio,
                                               cv=k,
                                               scoring=scorer)
    learning_curve_data = {'train': train_score,
                           'test': val_score}
    for legend, scores in learning_curve_data.items():
        plt.plot(N, np.abs(scores.mean(axis=1)), label=legend)
    plt.xlabel('train_sizes', labelpad=20)
    plt.legend()
    plt.show()


########################################################################################################################
#                                         Dummy estimators                                                             #
########################################################################################################################


def dummy_regression(x_train, y_train, x_test, y_test, strategy):
    """
    
    Regression strategies :

    case 1 : y_pred_random = np.random.randint(np.min(y), np.max(y), y_test.shape)
    case 2 : strategy is 'mean' or 'median'
    
    cf :
    - https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html
    
    """
    dum = DummyRegressor(strategy)
    dum.fit(x_train, y_train)
    y_predicted_dum = dum.predict(x_test)
    # np.sqrt(metrics.mean_squared_error(y_test, y_predicted_dum)) RMSE
    return y_predicted_dum, y_test


def dummy_classification(x_train, y_train, x_test, y_test, strategy):
    """
    Classifier strategies :

    'most_frequent' => Cette approche naïve permet d’évaluer si le modèle que nous proposons a appris « plus »
                       que simplement quelle est la classe la plus fréquente.
                       C’est intéressant si une des classes est beaucoup plus fréquente que les autres.

    'stratified'    => Cette approche naïve nous permet d’évaluer si les performances que nous observons
                       ne seraient pas simplement dûes aux proportions relatives des classes.

    'uniform'       => Cette méthode est recommandée quand on cherche à interpréter une courbe ROC ou une AUROC,
                       elles-mêmes construites à partir de classifieurs qui retournent des scores.

    cf :
    - https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

    """
    dum = DummyClassifier(strategy)
    dum.fit(x_train, y_train)
    y_predicted_dum = dum.predict(x_test)
    return dum.score(x_test, y_test)


########################################################################################################################
#                                         Models utilities                                                             #
########################################################################################################################


def model_name(model):
    """
    Get machine learning model label by accessing class name attribute
    :params: model: model class instance
    """
    return model.__class__.__name__


# joblib wrappers

def save_model(model, model_label=None, model_path=None):
    """
    Save machine learning model as object with joblib
    """
    if model_label is None:
        model_label = model_name(model)
    if model_path is None:
        joblib.dump(model, "{}.joblib".format(model_label))
    else:
        joblib.dump(model, f"{model_path}{model_label}.joblib")

    
def load_model(model_label, model_path=None):
    """
    Load machine learning model as object with joblib
    """
    if model_path is None:
        return joblib.load(model_label)
    else:
        return joblib.load(model_path+model_label)

    
# pickle wrapper

def pickle_data(filename, data=None, folder=None, method='w'):
    filename = filename if folder is None else f'{folder}/{filename}'
    if method is 'w':
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(data, file)
    elif method is 'r':
        with open(f'{filename}.pkl', 'rb') as file:
            file_data = pickle.load(file)
        return file_data
    