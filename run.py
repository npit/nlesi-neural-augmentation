from os.path import join, exists
from os import makedirs
import subprocess
import yaml
import pickle
import pandas as pd
import itertools

"""
Script to reproduce semantic neural augmentation experiments
"""
# settable parameters
############################################################
# folder to run experiments in
run_dir = "/media/npittaras/B47E380F7E37C8BE/Data/nle/preliminary_exps"
# folder where run scripts are
sources_dir = "/home/npittaras/Documents/project/nle-special/nlp-semantic-augmentation/"
# virtualenv folder
venv_dir = ""
# results csv file
results_file = "results_network_prelim.csv"
# evaluation measures
eval_measures = ["f1-score"]
aggr_measures = ["macro", "micro"]
run_types = ["run", "majority"]

# preliminary experiment params
mlp_params = {
    "hidden_size": [256, 512],
    "num_layers": [1, 2]
    }
lstm_params = {
    "hidden_size": [256, 512, 1024, 2048],
    "num_layers": [1, 2, 3, 4]
   }

#########################################################

def make_configs(variables, variable_names, baseconf):
    conf = baseconf.copy()
    for params in itertools.product:
        for p, param in enumerate(params):
            lconf = baseconf
            for name in variable_names[p][:-1]:
                lconf = lconf[name]
            lconf[variable_names[-1]] = param
        print(lconf)


results = {}

# dir checks
if venv_dir and not exists(venv_dir):
    print("Virtualenv dir {} not found".format(venv_dir))
    exit()
if not exists(run_dir):
    print("Run dir {} not found, creating.".format(run_dir))
    makedirs(run_dir)

# preliminary experiments on network architectures
configs = []
common_conf = {
    "dataset": {"name" : "20newsgroups"},
    "embedding": {"name": "glove", "dimension": 50},
    "train": {"epochs": 50, "folds": 5, "batch_size": 50},
    "folders": {"serialization": join(sources_dir, "serialization"), "embeddings": "embeddings"},
    "log_level": "info",
      }

hidden_size = [256, 512, 1024, 2048]
num_layers = [1, 2, 3, 4]
names = [["learner", "hidden_dim"], ["learner", "layers"]]
mlp_conf = common_conf.copy()
mlp_conf["learner"] = {"name":"mlp"}
mlp_conf["embedding"] = {"aggregation":"avg"}

configs.extend(make_configs([hidden_size, num_layers], names, mlp_conf))


configs.extend(mp_conf)
lstm_conf = common_conf.copy()
lstm_conf["learner"] = {"name":"mlp"}
lstm_conf["embedding"] = {"aggregation":["pad", 10, "first"]}
configs.append(lstm_conf) 

# prelim experiments
for name, network in zip(["lstm", "mlp"],[lstm_params, mlp_params]):
    print("Running experimens for {} learner".format(name))
    for lsize in network["hidden_size"]:
        for nl in network["num_layers"]:
            print("\tHidden / num. layers : {},{}.".format(lsize, nl))
            run_id = "{}_{}_{}".format(name, lsize, nl)
            experiment_dir = join(run_dir, run_id)
            completed_file = join(experiment_dir, "completed")
            error_file = join(experiment_dir, "error")

            if exists(completed_file):
                print("Skipping completed experiment {}".format(run_id))
            else:
                makedirs(experiment_dir, exist_ok=True)
                # amend the configuration file
                conf["run_id"] = run_id
                heads = ["name", "hidden_dim", "layers", "sequence_length"]
                conf["learner"] =  {n:x for (n,x) in zip(heads, [name, lsize, nl, 10] )} 

                conf["folders"]["logs"] = join(experiment_dir, "logs")
                conf["folders"]["results"] = join(experiment_dir, "results")
                conf["embedding"]["aggregation"] =  "avg" if name == "mlp" else "pad,10,first"

                conf_path = join(experiment_dir, "config.yml")
                with open(conf_path, "w") as f:
                    yaml.dump(conf, f)
                # write the run script file
                script_path = join(experiment_dir, "run.sh")
                with open(script_path, "w") as f:
                    if venv_dir:
                        f.write("source \"{}/bin/activate\"".format(venv_dir))

                    f.write("cd \"{}\"\n".format(sources_dir))
                    f.write("python3 \"{}\" --config_file \"{}\" && touch \"{}\" &&  exit 0\n".format(join(sources_dir, "main.py"), conf_path, completed_file))
                    f.write("touch '{}' && exit 1\n".format(error_file))
                subprocess.run(["/usr/bin/env", "bash", script_path])
                if exists(error_file):
                    exit(1)
            # read experiment results
            res_file = join(experiment_dir,"results", run_id, "results.pickle")
            with open(res_file, "rb") as f:
                res_data = pickle.load(f)
            results[run_id] = res_data

# show results
print_vals = {}
for run_id in results:
    print_vals[run_id] = {}
    for m in eval_measures:
        for run in run_types:
            for ag in aggr_measures:
                header = "{}.{}.{}".format(run, m, ag)
                val = results[run_id].loc[m][run][ag][-1]
                print_vals[run_id][header] = val
print(pd.DataFrame.from_dict(print_vals, orient='index'))
