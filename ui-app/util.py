
def get_artifacts():
    from io import BytesIO
    import requests
    import joblib
    import json
    import os    

    SCHEMA_FILE = "input_schema.json"
    MODEL_REPO = os.getenv("MODEL_REPO")
    MODEL_FILE = os.getenv("MODEL_FILE")

    r = requests.get(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}")
    r.raise_for_status()
    model = joblib.load(BytesIO(r.content))

    r = requests.get(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{SCHEMA_FILE}")
    r.raise_for_status()
    schema = json.loads(r.text)

    return model, schema
