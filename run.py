from os.path import join, exists
from os import makedirs
import subprocess
import yaml
import pandas as pd

"""
Script to reproduce semantic neural augmentation experiments
"""
# settable parameters

# folder to run experiments in
run_dir = "/home/nik/work/iit/submissions/NLE-special/experiments/test/"
# folder where run scripts are
sources_dir = "/home/nik/work/iit/submissions/NLE-special/code/"
# virtualenv folder
venv_dir = ""
# results csv file
results_file = "results.csv"

# preliminary experiment params
mlp_params = {
    "hidden_size": [256, 512, 1024, 2048, 4096],
    "num_layers": [1, 2, 3, 4]
    }
lstm_params = {
    "hidden_size": [256, 512, 1024, 2048, 4096],
    "num_layers": [1, 2, 3, 4]
    }

#########################################################

# dir checks
if venv_dir and not exists(venv_dir):
    print("Virtualenv dir {} not found".format(venv_dir))
    exit()
if not exists(run_dir):
    print("Run dir {} not found, creating.".format(run_dir))
    makedirs(run_dir)


# preliminary experiments on network architectures
conf = {"dataset": "20newsgroups",
        "embedding": "glove,50",
        "aggregation": "avg",
        "train": {
            "epochs": 50,
            "folds": 3,
            "batch_size": 50
        },
        "log_level": "info",
        "options":{"data_limit": 100},
        "serialization_dir": join(sources_dir, "serializations_prelim")
}

# prelim experiments
for name, network in zip(["mlp", "lstm"],[mlp_params, lstm_params]):
    print("Running experimens for {} learner".format(name))
    for lsize in network["hidden_size"]:
        for nl in network["num_layers"]:
            print("\tHidden / num. layers : {},{}.".format(lsize, nl))
            run_id = "{}_{}_{}".format(name, lsize, nl)
            experiment_dir = join(run_dir, run_id)
            completed_file = join(experiment_dir, "completed")
            if exists(completed_file):
                print("Skipping completed experiment {}".format(run_id))
            makedirs(experiment_dir, exist_ok=True)
            # amend the configuration file
            conf["learner"] = "mlp,{},{}".format(lsize, nl)
            conf["log_dir"] = join(experiment_dir, "logs")
            conf["results_folder"] = join(experiment_dir, "results")
            conf_path = join(experiment_dir, "config.yml")
            with open(conf_path, "w") as f:
                yaml.dump(conf, f)
            # write the run script file
            script_path = join(experiment_dir, "run.sh")
            with open(script_path, "w") as f:

                if venv_dir:
                    f.write("source \"{}/bin/activate\"".format(venv_dir))

                f.write("cd \"{}\"\n".format(sources_dir))
                f.write("python3 \"{}\" --config_file \"{}\"".format(join(sources_dir, "main.py"), conf_path))

            # subprocess.run(["/usr/bin/env", "bash", script_path])
            subprocess.run(["touch", completed_file])
            # read results
            res_file = join(experiment_dir,"logs","results.txt")
            results = pd.read_csv(res_file)


