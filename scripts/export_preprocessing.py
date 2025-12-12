import joblib
import json
import os

PREPPATH = 'datasets/processed/preprocessor.pkl'
OUTDIR = 'datasets/processed'

if not os.path.exists(PREPPATH):
    raise SystemExit(f"Preprocessor not found at {PREPPATH}. Run preprocess.py first.")

preprocessor = joblib.load(PREPPATH)

# Ensure named_transformers_ exists
if not hasattr(preprocessor, 'named_transformers_'):
    raise SystemExit('Loaded object does not look like a ColumnTransformer with named_transformers_.')

num = preprocessor.named_transformers_.get('num')
cat = preprocessor.named_transformers_.get('cat')

if num is None and cat is None:
    raise SystemExit('ColumnTransformer does not contain transformers named "num" or "cat".')

# Export components
if num is not None:
    joblib.dump(num, os.path.join(OUTDIR, 'scaler.joblib'))
    print('Saved scaler ->', os.path.join(OUTDIR, 'scaler.joblib'))
if cat is not None:
    joblib.dump(cat, os.path.join(OUTDIR, 'encoder.joblib'))
    print('Saved encoder ->', os.path.join(OUTDIR, 'encoder.joblib'))

# Collect metadata: we attempt to get feature names if available
metadata = {}
try:
    # numerical and categorical feature names are stored in transformer attribute "feature_names_in_" for sklearn >=1.0
    # For ColumnTransformer we can try to get them from transformers_ information
    num_features = None
    cat_features = None
    if hasattr(preprocessor, 'transformers_'):
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                num_features = cols
            if name == 'cat':
                cat_features = cols
    metadata['num_features'] = list(num_features) if num_features is not None else None
    metadata['cat_features'] = list(cat_features) if cat_features is not None else None

    # If encoder has categories_ we include them
    if cat is not None and hasattr(cat, 'categories_'):
        metadata['encoder_categories'] = [list(c) for c in cat.categories_]

except Exception as e:
    metadata['warning'] = f'Could not extract full metadata: {e}'

with open(os.path.join(OUTDIR, 'preprocessor_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print('Saved metadata ->', os.path.join(OUTDIR, 'preprocessor_metadata.json'))
