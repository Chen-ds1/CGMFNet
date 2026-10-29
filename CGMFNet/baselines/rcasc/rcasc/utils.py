import json
from pathlib import Path

def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))
