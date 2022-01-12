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
def aggr(data, stat, aggregate, tag="time"):
    mean_table = pd.pivot_table(
        data, aggregate, index=stat, aggfunc=np.mean
    )
    line_mean_df = pd.DataFrame(mean_table.to_records())

    fig, ax = plt.subplots(1, 2, sharey="row", figsize=(18, 6))

    tmp_data = line_mean_df.loc[line_mean_df["filter"] == False]
    tmp_data.plot(x="binned", y=aggregate, ax=ax[0])
    ax[0].get_legend().remove()
    ax[0].set_title("No Redundancy Filter")

    tmp_data = line_mean_df.loc[line_mean_df["filter"] == True]
    tmp_data.plot(x="binned", y=aggregate, kind="line", ax=ax[1])
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
        "new_results/"+tag+".png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
        # pad_inches=0.35,
    )
    plt.savefig(
        "new_results/"+tag+".pdf",
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



if __name__ == "__main__":
    path = "new_results/"

    types = [l for l in range(1, 17) if l != 9]
    # for t in types:
    #     for filter in [True, False]:
    #         if os.path.exists(path+f"type{t:02d}_filter_{filter}.csv"):
    #             csv_file=pd.read_csv(path+f"type{t:02d}_filter_{filter}.csv", delimiter=",")
    #             csv_file['type'] = t
    #             csv_file['filter'] = filter
    #             csv_file.to_csv(path+f'type{t:02d}_filter_{filter}_modified.csv', encoding='utf-8', index=False)


    # all_csv_files = [
    #     file for file in glob.glob(path+"*modified.csv")
    # ]
    all_csv_files = []
    for t in types:
        if os.path.exists(path+f"type{t:02d}_filter_True.csv") and os.path.exists(path+f"type{t:02d}_filter_False.csv"):
            all_csv_files.append(path + f"type{t:02d}_filter_True_modified.csv")
            all_csv_files.append(path + f"type{t:02d}_filter_False_modified.csv")

    data = pd.concat((pd.read_csv(f) for f in all_csv_files))
    print(min(data['number_of_constraints']), max(data['number_of_constraints']))
    bins=[0, 50, 100, 200, 350, 2000]
    data['binned'] = pd.cut(data['number_of_constraints'], bins, labels=["<= 50","50 - 100", "100 - 200", "200 - 350", "350 <="])
    data['total_time'] = data["test_time_taken"] + data["time_taken"]
    aggr(data, ["filter", "binned"], ["time_taken", "test_time_taken", "total_time"], "time")
    aggr(data, ["filter", "binned"], ["percentage_pos", "percentage_neg"], "accuracy")
    aggr(data, ["filter", "binned"], ["number_of_constraints", "constraints_after_filter"], "filter")
