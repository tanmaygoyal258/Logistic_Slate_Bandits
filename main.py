import numpy as np
import os
import json
import argparse
from GLMEnv import GLMEnv
from Ordered_Slot_Bandit import Ordered_Slot_Bandit
from ETC_Slate import ETC_Slate
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type = str, default = 'ada_ofu_ecolog', help = 'Name of algorithm to run: \
                        ["ada_ofu_ecolog", "slate_glm_ofu", "mps" , "ordered_bandit_linear_loss , ordered_bandit_logistic_loss , TS_ecolog , slate_glm_TS , ETC_slate"]')
    parser.add_argument('--warmup' , action = "store_true")
    parser.add_argument("--reward_type" , type = str , default = "logistic" , help = "reward type : [logistic , probit]")
    parser.add_argument("--num_contexts" , type = int , default = None , help = "number of contexts: 1 for non-contextual setting and None for infinite context setting")
    parser.add_argument('--seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--theta_star', type = str, default = "random", help = 'file containing optimal parameter')
    parser.add_argument('--normalize_theta_star', action = "store_true")
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--arm_dim', type = int, default = 5, help = 'arm dimensions')
    parser.add_argument('--slot_count', type = int, default = 5, help = 'number of slots')
    parser.add_argument('--item_count', type = int, default = 4, help = 'number of items per slot')
    return parser.parse_args()

def generate_slot_arms(params):
    num_contexts = params["horizon"] if params["num_contexts"] is None else params["num_contexts"]
    all_arms = []
    for c in range(num_contexts):
        context_arms = []
        for s in range(params["slot_count"]):
            slot_arms = []
            for _ in range(args.item_count):
                slot_arms.append([np.random.random()*2-1 for i in range(args.arm_dim)])
            slot_arms = [arm/np.linalg.norm(arm) * 1/np.sqrt(params["slot_count"]) for arm in slot_arms]
            context_arms.append(slot_arms)
        all_arms.append(context_arms)
    return all_arms

    
def generate_theta_star(args):
    # generate theta_star
    if args.theta_star != "random" and "npy" in args.theta_star:
        theta_star = np.load(args.theta_star)
        assert len(theta_star) == args.arm_dim * args.slot_count
    elif args.theta_star == "random":
        theta_star = np.array([np.random.random()*2-1 for i in range(args.arm_dim * args.slot_count)])
        if args.normalize_theta_star:
            theta_star /= np.linalg.norm(theta_star)
    return theta_star


if __name__ ==  "__main__":

    alg_names = ["ada_ofu_ecolog", "slate_glm_ofu", "mps" , "ordered_bandit_linear_loss" , "ordered_bandit_logistic_loss" , "TS_ecolog" , "slate_glm_TS" , "ETC_slate"]

    # read the arguments
    args = parse_args()
    assert args.alg_name in alg_names , f"Incorrect algorithm name, should be one of {alg_names}"
    assert args.reward_type in ["logistic" , "probit"]

    # set the seed before any randomization occurs
    np.random.seed(args.seed)
    
    # create the params dictionary
    params = {}
    params["alg_name"] = args.alg_name
    params["warmup"] = args.warmup
    params["num_contexts"] = args.horizon if args.num_contexts is None else args.num_contexts
    params["reward_type"] = args.reward_type
    params["horizon"] = args.horizon
    params["failure_level"] = args.failure_level
    params["arm_dim"] = args.arm_dim
    params["slot_count"] = args.slot_count
    params["item_count"] = args.item_count
    params["seed"] = args.seed

    if params["alg_name"] == "slate_glm_TS" and params["warmup"]:
        params["alg_name"] = "slate_glm_TS_Fixed"

    theta_star = generate_theta_star(args)
    params["thetastar"] = theta_star.tolist()
    params["param_norm_ub"] = int(np.linalg.norm(theta_star)) + 1

    print(params)

    # generate the arms for each slot
    slot_arms = generate_slot_arms(params)

    # check validity of the data path
    data_path = f"Results"
    if not os.path.exists(data_path):
            os.makedirs(data_path)
    data_path_with_alg = f"{data_path}/{args.alg_name}" if args.alg_name in ["ada_ofu_ecolog", "slate_glm_ofu", "TS_ecolog" , "slate_glm_TS"] else data_path
    if not os.path.exists(data_path_with_alg):
        os.makedirs(data_path_with_alg)
    suffix = f"{args.reward_type}_contexts={args.num_contexts}_N={args.slot_count}_K={args.item_count}" if args.alg_name in ["ada_ofu_ecolog", "slate_glm_ofu", "TS_ecolog" , "slate_glm_TS"] else \
        f"{args.alg_name}_{args.reward_type}_contexts={args.num_contexts}_N={args.slot_count}_K={args.item_count}"
    data_path_with_details = f"{data_path_with_alg}/{suffix}"
    if not os.path.exists(data_path_with_details):
        os.makedirs(data_path_with_details)
    params["data_path"] = data_path_with_details

    # dump the json file with the params
    with open(data_path_with_details + "/params.json", "w") as outfile:
        json.dump(params, outfile)

    # set the environment
    if args.alg_name in ["ada_ofu_ecolog", "slate_glm_ofu", "TS_ecolog" , "slate_glm_TS" , "mps" , "slate_glm_TS_Fixed"]:
        env = GLMEnv(params , slot_arms , theta_star)
    elif args.alg_name in ["ordered_bandit_linear_loss" , "ordered_bandit_logistic_loss"]:
        env = Ordered_Slot_Bandit(params , slot_arms , theta_star)
    elif args.alg_name in ["ETC_slate"]:
        env = ETC_Slate(params , slot_arms , theta_star)
    env.play_algorithm()

    # obtain the regret, reward, and time arrays, and save them
    regret_arr = env.regret_arr
    np.save(data_path_with_details + "/regret.npy", regret_arr)
    try:
        reward_arr = env.reward_arr
        np.save(data_path_with_details + "/reward.npy", reward_arr)
        pull_time_arr = env.pull_time_arr
        update_time_arr = env.update_time_arr
        np.save(data_path_with_details + "/pull_time.npy" , pull_time_arr)
        np.save(data_path_with_details + "/update_time.npy" , update_time_arr)
    except: 
        # the time and reward arrays were not instantiated for the algorithms
        pass
    

    # print("Test Command")
    # print("Another Test Command")