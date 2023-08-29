import json

json_string = '{"key": "value", "key2": "value2" invalid_json}'

data = json.loads(json_string)
