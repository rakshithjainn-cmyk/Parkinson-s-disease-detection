from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier




def build_models():
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
svm = SVC(probability=True, kernel='rbf', random_state=42)


ensemble = VotingClassifier(estimators=[
('rf', rf),
('xgb', xgb),
('svm', svm)
], voting='soft')


return rf, xgb, svm, ensemble
