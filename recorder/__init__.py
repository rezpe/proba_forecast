import datetime
import json
import numpy as np

class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def save_result(result,name,folder="results"):
    prefix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
    print(folder+"/"+prefix+name+".json")
    with open(folder+"/"+prefix+name+".json","w") as output:
        output.write(json.dumps(result,cls=NumPyArangeEncoder))