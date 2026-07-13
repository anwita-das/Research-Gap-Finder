import json
from pathlib import Path
path = Path(r'c:/Users/Anwita.Das/Desktop/Research-Gap-Finder/data/processed/knowledge_graph.json')
with path.open('r', encoding='utf-8') as handle:
    data = json.load(handle)
print(type(data).__name__)
print(list(data.keys())[:20])
print('nodes', len(data.get('nodes', [])))
print('links', len(data.get('links', [])))
print('edges', len(data.get('edges', [])))
print('graph type', type(data.get('graph')).__name__)
print('first node keys', list(data.get('nodes', [{}])[0].keys()) if data.get('nodes') else None)
print('first link keys', list(data.get('links', [{}])[0].keys()) if data.get('links') else None)
print('first edge keys', list(data.get('edges', [{}])[0].keys()) if data.get('edges') else None)
print('graph keys', list(data.get('graph', {}).keys()) if isinstance(data.get('graph'), dict) else None)
