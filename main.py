import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Models tested: Decision Tree, Random Forest, KNN, Linear Reg., Log. Reg., SVM
mode = 'cleveland'

def read_data(mode):
	entries = []
	origEntries = []
	with open(os.path.join('data', mode + '.txt'), 'r') as fp:
		lines = [line.strip() for line in fp.readlines()]
		for index in range(0, len(lines), 10):
			entry = ' '.join(lines[index:index+10]).split(' ')
			if len(entry) != 76:
				continue
			origEntries.append(entry)
			entries.append(entry[2:68])
	return origEntries, entries

origEntries, entries = read_data(mode)
X = np.asarray([entry[:55] + entry[56:] for entry in entries], dtype='float64')
y = np.asarray([1 if int(entry[57]) > 0 else 0 for entry in origEntries], dtype='float64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None)

models = [DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier(), LinearRegression(), LogisticRegression(), SVC()]

cross_val_scores = {}
for idx in range(len(models)):
	models[idx] = make_pipeline(StandardScaler(), models[idx])
	#models[idx].__class__(**tuned_parameters[models[idx]]['precision']))
	cv_score = cross_val_score(models[idx], X_train, y_train, cv=5)
	cross_val_scores[models[idx]] = {
		'mean':cv_score.mean(),
		'std':cv_score.std()
	}

cross_val_scores_items = list(cross_val_scores.items())
max_mean_index = 0
max_mean_model = cross_val_scores_items[0][0]
for idx, key in enumerate(cross_val_scores):
	if cross_val_scores_items[idx][1]['mean'] > cross_val_scores_items[max_mean_index][1]['mean']:
		max_mean_index = idx
		max_mean_model = key

print(' ------------ Done ------------ ')
print('Best Model: ')
print(cross_val_scores_items[max_mean_index][1])
