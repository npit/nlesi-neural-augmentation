from os.path import join, exists, isabs
from os import makedirs
import subprocess
import yaml
import pickle
import pandas as pd
from functools import reduce
import itertools
from copy import deepcopy
from numpy import round

"""
Script to reproduce semantic neural augmentation experiments
"""
def traverse_dict(ddict, key, prev_keys):

    res = []
    if key is None:
        for key in ddict:
            rres = traverse_dict(ddict, key, prev_keys)
            res.append(rres)
        return res
    if type(ddict[key]) == dict:
        prev_keys.append(key)
        res =  traverse_dict(ddict[key], None, prev_keys)
    else:
        val = ddict[key]
        if type(val) != list:
            val = [val]
        res = (val, prev_keys + [key])
    return res


def make_configs(base_config, run_dir, sources_dir):
    vars = []
    params = base_config["params"]
    base_raw_folder = base_config["folders"]["raw_data"]
    base_serialization_folder = base_config["folders"]["serialization"]
    for val in sorted(params.keys()):
        seqs = traverse_dict(params, val, [])
        vars.extend(seqs)

    configs, run_ids = [], []
    vars = sorted(vars, key = lambda x : str(x[1]))

    values = [v[0] for v in vars]
    names =  [v[1] for v in vars]

    for combo in itertools.product(*values):
        conf = deepcopy(base_config)
        name_components = []
        for v, value in enumerate(combo):
            lconf = conf
            name_components.append(str(value))
            key_chain = names[v]
            for key in key_chain[:-1]:
                if key not in lconf:
                    lconf[key] = {}
                lconf = lconf[key]
            lconf[key_chain[-1]] = value
        # dirs
        run_id = "_".join(name_components)
        conf["folders"]["run"] = join(run_dir, run_id)
        if isabs(base_serialization_folder):
            conf["folders"]["serialization"] = base_serialization_folder
        else:
            conf["folders"]["serialization"] = join(sources_dir, base_serialization_folder)
        if isabs(base_raw_folder):
            conf["folders"]["raw_data"] = base_raw_folder
        else:
            conf["folders"]["raw_data"] = join(sources_dir, base_raw_folder)
        configs.append(conf)
        run_ids.append(run_id)
    return configs, run_ids

# make a run id name out of a list of nested dict keys and a configuration dict
def make_run_ids(keychains, confs):
    names = []
    for conf in confs:
        name_components = []
        for keychain in keychains:
            name_components.append(reduce(dict.get, keychain, conf))
        names.append("_".join(map(str,name_components)))
    return names


def main():
    # settable parameters
    ############################################################

    # config file
    config_file = "config.yml"

    ############################################################

    # set the expeirment parameters via a configuration list
    conf = yaml.load(open(config_file))
    # evaluation measures
    exps = conf["experiments"]
    eval_measures = exps["measures"] if "measures" in exps else ["f1-score", "accuracy"]
    aggr_measures = exps["aggregation"] if "aggregation" in exps else ["macro", "micro", "fold-var"]
    stat_functions = exps["stat_funcs"] if "stat_funcs" in exps else ["mean"]
    run_types = exps["run_types"] if "run_types" in exps else "run"

    # folder to run experiments in
    run_dir = exps["run_folder"]
    # folder where run scripts are
    sources_dir = exps["sources_dir"]

    configs, run_ids = make_configs(conf, run_dir, sources_dir)

    # virtualenv folder
    venv_dir = conf["experiments"]["venv"] if "venv" in conf["experiments"] else None
    # results csv file
    results_file = conf["experiments"]["results_file"]

    configs, run_ids = make_configs(conf)

    results = {}

    # dir checks
    if venv_dir and not exists(venv_dir):
        print("Virtualenv dir {} not found".format(venv_dir))
        exit()
    if not exists(run_dir):
        print("Run dir {} not found, creating.".format(run_dir))
        makedirs(run_dir)

    #################################################################################

    # prelim experiments
    for conf_index, (conf, run_id) in enumerate(zip(configs, run_ids)):
        print("Running experimens for configuration {}/{}: {}".format(conf_index+1, len(configs), run_id))
        experiment_dir = conf["folders"]["run"]
        completed_file = join(experiment_dir, "completed")
        error_file = join(experiment_dir, "error")
        # results to run folders, if not specified otherwise
        respath = conf["folders"]["results"]
        if not isabs(respath):
            conf["folders"]["results"] = join(experiment_dir, respath)

        if exists(completed_file):
            print("Skipping completed experiment {}".format(run_id))
        else:
            makedirs(experiment_dir, exist_ok=True)

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
        exp_res_file = join(experiment_dir,"results", "results.pickle")
        with open(exp_res_file, "rb") as f:
            res_data = pickle.load(f)
        results[run_id] = res_data

    # show results
    print_vals = {}
    for run_id in results:
        print_vals[run_id] = {}
        for m in eval_measures:
            for run in run_types:
                for ag in aggr_measures: # classwise
                    for stat in stat_functions:
                        header = "{}.{}.{}.{}".format(run[:3], m[:3], ag[:3], stat)
                        if stat == "var":
                            val = round(results[run_id].loc[m][run][ag]["var"], decimals=4)
                        elif stat == "mean":
                            val = round(results[run_id].loc[m][run][ag]["mean"], decimals=4)
                        elif stat == "std":
                            val = round(results[run_id].loc[m][run][ag]["std"], decimals=4)
                        print_vals[run_id][header] = val
    df = pd.DataFrame.from_dict(print_vals, orient='index')
    print(df.to_string())

if __name__ == "__main__":
    main()
