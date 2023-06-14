import os, simpy, math, json
import numpy as np
import pandas as pd
from environment.test_simulation import *

atcs_k1 = 3.5
atcs_k2 = 1.4
covert_k = 7.6
weight = {"ATCS": [atcs_k1, atcs_k2], "COVERT": covert_k}


def test(num_job=200, num_m=5, ddt=None, sample_data=None, routing_rule=None, file_path=None, pt_var=None):
    tard_list = list()
    setup_list = list()
    makespan_list = list()
    for episode in range(20):
        iat = 15 / num_m
        model = dict()
        env = simpy.Environment()
        monitor = Monitor()
        monitor.reset()
        job_list = list()
        for j in range(num_job):
            job_name = "Job {0}".format(j)
            processing_time = np.random.uniform(10, 20) if sample_data is None else sample_data['processing_time'][j]
            feature = random.randint(0, 5) if sample_data is None else sample_data['feature'][j]
            job_list.append(Job(name=job_name, processing_time=processing_time, feature=feature))

        routing = Routing(env, model, monitor, end_num=num_job, routing_rule=routing_rule, weight=weight)
        routing.reset()
        sink = Sink(env, monitor)
        sink.reset()
        source = Source(env, job_list, iat, ddt, routing, monitor)

        for m in range(num_m):
            machine_name = "Machine {0}".format(m)
            model[machine_name] = Process(env, machine_name, routing, sink, monitor, pt_var=pt_var)
            model[machine_name].reset()

        env.run()
        monitor.get_logs(file_path=file_path + '/log_{0}_{1}.csv'.format(round(ddt, 1), episode))

        tard_list.append(monitor.tardiness / num_job)
        setup_list.append(monitor.setup / num_job)
        makespan_list.append(sink.makespan)

    avg_tardiness = np.mean(tard_list)
    avg_setup = np.mean(setup_list)
    avg_makespan = np.mean(makespan_list)

    return avg_tardiness, avg_setup, avg_makespan


if __name__ == "__main__":
    rule_list = ["SSPT", "ATCS", "MDD", "COVERT"]
    num_job = 100
    with open('../sample100.json', 'r') as f:
        sample_data = json.load(f)
    ddt_list = [1.0 + 0.2 * i for i in range(6)]
    output_dict = dict()
    for test_i in sample_data.keys():
        test_data = sample_data[test_i]
        output_dict["Test {0}".format(int(test_i))] = dict()
        for ddt in ddt_list:
            output_dict["Test {0}".format(int(test_i))][ddt] = dict()
            for rule in rule_list:
                tard, setup, makespan = test(num_job=num_job, ddt=ddt, sample_data=test_data, routing_rule=rule, pt_var=0.1, file_path=".")
                output_dict["Test {0}".format(int(test_i))][ddt][rule] = {"Tardiness": tard,
                                                                          "Setup": setup,
                                                                          "Makespan": makespan}

    output = dict()
    for ddt in ddt_list:
        output[ddt] = {model: {"Tardiness": 0.0, "Setup": 0.0, "Makespan": 0.0} for model in rule_list}
        for test in output_dict.keys():
            for model in rule_list:
                output[ddt][model]["Tardiness"] += output_dict[test][ddt][model]["Tardiness"] * 0.1
                output[ddt][model]["Setup"] += output_dict[test][ddt][model]["Setup"] * 0.1
                output[ddt][model]["Makespan"] += output_dict[test][ddt][model]["Makespan"] * 0.1

    data = pd.DataFrame()
    for ddt in ddt_list:
        temp_df = pd.DataFrame()
        temp_df["DDT={0}".format(ddt)] = rule_list
        temp_df["Tardiness"] = [output[ddt][model]["Tardiness"] for model in rule_list]
        temp_df["Setup"] = [output[ddt][model]["Setup"] for model in rule_list]
        temp_df["Makespan"] = [output[ddt][model]["Makespan"] for model in rule_list]

        data = pd.concat([data, temp_df], 1)
    data.to_excel("../Test Result(100)_heuristic.xlsx")
    print(0)