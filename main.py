import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


df_aflow_elastic = pd.read_csv('aflow_data/aflow_elastic_data.csv')
uncleaned_formulae = df_aflow_elastic['ENTRY ']
cleaned_formulae = []

for cell_value in uncleaned_formulae:
   
    split_list = cell_value.split(" [")
    clean_formula = split_list[0]
    cleaned_formulae.append(clean_formula)

df_cleaned = pd.DataFrame()
df_cleaned['formula'] = cleaned_formulae
df_cleaned['bulk_modulus'] = df_aflow_elastic['AEL VRH bulk modulus ']
check_for_duplicates = df_cleaned['formula'].value_counts()
df_cleaned.drop_duplicates('formula', keep='first', inplace=True)
plt.figure(1, figsize=(10, 10))
df_cleaned['bulk_modulus'].hist(bins=20, grid=False, edgecolor='black')
plt.plot()
df_cleaned.columns = ['formula', 'target']
X, y, formulae = composition.generate_features(df_cleaned)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)
scalar = StandardScaler()
normalizer = Normalizer()
X_train_scaled = scalar.fit_transform(X_train)  # get statistics & transform
X_test_scaled = scalar.transform(X_test)  # trandform using 'training' stats.
X_train_scaled = normalizer.fit_transform(X_train_scaled)  # normalize vectors
X_test_scaled = normalizer.transform(X_test_scaled)  # normalize vectors
model = SVR()
cv = KFold(n_splits=5, shuffle=True, random_state=1)
c_parameters = np.logspace(-1, 3, 5)
gamma_parameters = np.logspace(-2, 2, 5)
parameter_candidates = {'C': c_parameters,
                        'gamma': gamma_parameters}
grid = GridSearchCV(estimator=model,
                    param_grid=parameter_candidates,
                    cv=cv)
grid.fit(X_train_scaled, y_train)
best_parameters = grid.best_params_
print(best_parameters)
utils.plot_2d_grid_search(grid, midpoint=0.7, vmin=-0, vmax=1)
plt.plot()
final_model = SVR(**best_parameters)
final_model.fit(X_train_scaled, y_train)
y_test_predicted = final_model.predict(X_test_scaled)
utils.plot_act_vs_pred(y_test, y_test_predicted)
score = r2_score(y_test, y_test_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_test_predicted))

print('r2 score: {:0.3f}, rmse: {:0.2f}'.format(score, rmse))
class MaterialsModel():
    def __init__(self, trained_model, scalar, normalizer):
        self.model = trained_model
        self.scalar = scalar
        self.normalizer = normalizer

    def predict(self, formula):
        '''
        Parameters
        ----------
        formula: str or list of strings
            input chemical formula or list of formulae you want predictions for
    
        Return
        ----------
        prediction: pd.DataFrame()
            predicted values generated from the given data
        '''
        # Store our formula in a dataframe. Give dummy 'taget value'.
        # (we will use composition.generate_features() to get the features)
        if type(formula) is str:
            df_formula = pd.DataFrame()
            df_formula['formula'] = [formula]
            df_formula['target'] = [0]
        if type(formula) is list:
            df_formula = pd.DataFrame()
            df_formula['formula'] = formula
            df_formula['target'] = np.zeros(len(formula))
        # here we get the features associated with the formula
        X, y, formula = composition.generate_features(df_formula)
        # here we scale the data (acording to the training set statistics)
        X_scaled = self.scalar.transform(X)
        X_scaled = self.normalizer.transform(X_scaled)
        y_predicted = self.model.predict(X_scaled)
        # save our predictions to a dataframe
        prediction = pd.DataFrame(formula)
        prediction['predicted value'] = y_predicted
        return prediction
bulk_modulus_model = MaterialsModel(final_model, scalar, normalizer)
formulae_to_predict = ['NaCl', 'Pu2O4', 'NaNO3']
formula = 'NaCl'
bulk_modulus_prediction = bulk_modulus_model.predict(formula)
