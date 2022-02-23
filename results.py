import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from mpl_toolkits import mplot3d
import os
import glob

plt.rcParams.update({"font.size": 12})

def std_err(x):
    return np.std(x) / np.sqrt(len(x))



########### for noisy data ##############
def aggr(data, stat, aggregate, tag="time", kind="line"):
    mean_table = pd.pivot_table(
        data, aggregate, index=stat, aggfunc=np.mean
    )
    line_mean_df = pd.DataFrame(mean_table.to_records())
    # print(line_mean_df)

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    tmp_data = line_mean_df.loc[line_mean_df["filter"] == False]
    tmp_data.plot(x=stat[1], y=aggregate, ax=ax[0], kind=kind)
    ax[0].get_legend().remove()
    ax[0].set_title("No Redundancy Filter")

    tmp_data = line_mean_df.loc[line_mean_df["filter"] == True]
    tmp_data.plot(x=stat[1], y=aggregate, kind="line", ax=ax[1])
    ax[1].get_legend().remove()
    ax[1].set_title("With Redundancy Filter")

    handles, labels = ax[1].get_legend_handles_labels()

    lgd = fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=3,
    )
    plt.savefig(
        "new_results/"+tag+"_1.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        # pad_inches=0.35,
    )
    plt.savefig(
        "new_results/"+tag+"_1.pdf",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        # pad_inches=0.35,
    )
    # plt.show()

    # count_table = pd.pivot_table(
    #     data, aggregate, index=stat, aggfunc="count"
    # )
    # count_table_df = pd.DataFrame(count_table.to_records())
    # print(count_table_df)

    # std_table = pd.pivot_table(
    #     data, aggregate, index=stat, aggfunc=std_err
    # )
    # line_std_df = pd.DataFrame(std_table.to_records())
    # print(line_std_df)
##########################################
def aggr_acc(data, stat, aggregate, tag="time", kind="bar"):
    mean_table = pd.pivot_table(
        data, aggregate, index=stat, aggfunc=np.mean
    )
    line_mean_df = pd.DataFrame(mean_table.to_records())

    fig, ax = plt.subplots(1, 1, sharey="row", figsize=(18, 6))

    tmp_data = line_mean_df.loc[line_mean_df["filter"] == True]
    tmp_data.plot(x=stat[1], y=aggregate, ax=ax, kind=kind)
    ax.get_legend().remove()
    ax.set_title("No Redundancy Filter")

    handles, labels = ax.get_legend_handles_labels()

    lgd = fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=2,
    )
    plt.savefig(
        "new_results/"+tag+"_2.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        # pad_inches=0.35,
    )
    plt.savefig(
        "new_results/"+tag+"_2.pdf",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        # pad_inches=0.35,
    )
##########################################

def aggr_competition_data(path, types, stat, aggregate, tag="time", kind="bar"):
    all_csv_files = []
    for t in types:
        # print(path + f"type{t:02d}.csv")
        if os.path.exists(path + f"type_{t:02d}_propositional.csv"):
            # path + f"type{t:02d}.csv"
            all_csv_files.append(path + f"type_{t:02d}_propositional.csv")
    data = pd.concat((pd.read_csv(f) for f in all_csv_files))

    mean_table = pd.pivot_table(
        data, aggregate, index=stat, aggfunc=np.mean
    )
    line_mean_df = pd.DataFrame(mean_table.to_records())
    print(line_mean_df)

    fig, ax = plt.subplots(1, 1, sharey="row", figsize=(18, 6))

    line_mean_df.plot(x=stat[0], y=aggregate, ax=ax, kind=kind)

    plt.savefig(
        path+tag+".png",
        bbox_inches="tight",
        # pad_inches=0.35,
    )


if __name__ == "__main__":
    path = "results/type_level_neg_details/"
    types = [l for l in range(1, 17) if l not in [9]]
    aggr_competition_data(path, types, ["type"], ["precision", "recall"])
    # aggr_competition_data(path, types, ["type"], ["perc_pos", "perc_neg"])

    # all_csv_files = []
    # for t in types:
    #     if os.path.exists(path+f"type{t:02d}_filter_True.csv") and os.path.exists(path+f"type{t:02d}_filter_False.csv"):
    #         all_csv_files.append(path + f"type{t:02d}_filter_True_modified.csv")
    #         all_csv_files.append(path + f"type{t:02d}_filter_False_modified.csv")
    #
    # data = pd.concat((pd.read_csv(f) for f in all_csv_files))
    # data.sort_values('number_of_constraints', inplace=True)
    # data['binned'] = pd.qcut(data['number_of_constraints'], 5)
    # data['total_time'] = data["test_time_taken"] + data["time_taken"]
    # aggr(data, ["filter", "binned"], ["time_taken", "test_time_taken", "total_time"], "time")
    # aggr(data, ["filter", "binned"], ["number_of_constraints", "constraints_after_filter"], "filter")
