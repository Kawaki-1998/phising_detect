import joblib

# Load feature names
feature_names = joblib.load('models/feature_names.pkl')
print("Feature names:")
for name in feature_names:
    print(name) 