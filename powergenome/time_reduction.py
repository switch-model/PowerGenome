"""
Functions to cluster or otherwise reduce the number of hours in generation and
load profiles
"""

import datetime
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

logger = logging.getLogger(__name__)


def max_rep_periods(
    resource_profiles: pd.DataFrame,
    load_profiles: pd.DataFrame,
    days_in_group: int,
    num_clusters: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame, pd.DataFrame]:
    """Shortcut clustering when every representative period is assigned once (e.g. 52 rep
    weeks in a year).

    Parameters
    ----------
    resource_profiles : pd.DataFrame
        Hourly profiles of all resources
    load_profiles : pd.DataFrame
        Hourly demand profile in each region
    days_in_group : int
        Number of days in each period
    num_clusters : int
        Numer of representative periods

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[int], pd.DataFrame, pd.DataFrame]
        Load and resource profile dataframes with first N hours, where N is
        num_clusters * days_in_group * 24, the cluster weights (all values are 1),
        a mapping of each period to the representative period and the month,
        and the order of representative periods.
    """
    num_hours = num_clusters * days_in_group * 24
    cluster_weights = [1] * num_clusters
    time_series_mapping = pd.DataFrame(
        data={
            "Period_Index": range(1, 1 + num_clusters),
            "Rep_Period_Index": range(1, 1 + num_clusters),
            "Month": 0,
        }
    )
    for Period_Index in time_series_mapping["Period_Index"]:
        dayOfYear = (days_in_group * Period_Index - 1) % 365 + 1
        d = datetime.datetime.strptime("{} {}".format(dayOfYear, 2011), "%j %Y")
        time_series_mapping["Month"][Period_Index - 1] = d.month
    rep_point = [f"p{i}" for i in range(1, 1 + num_clusters)]
    rep_point_df = pd.DataFrame(data=rep_point, columns=["slot"])

    reduced_load_profiles = load_profiles.iloc[:num_hours, :].reset_index(drop=True)
    reduced_resouce_profiles = resource_profiles.iloc[:num_hours, :]

    return (
        reduced_load_profiles,
        reduced_resouce_profiles,
        cluster_weights,
        time_series_mapping,
        rep_point_df,
    )


def make_time_groups(df, days_in_group):
    """
    Convert df from one row per historical hour, one column per profile (load or
    resource) to one row per profile per hour-in-group, one col per group. This
    drops any fractional groups at the end of df, so annual generation for each
    zone will not exactly match up with raw data.
    """
    hours_in_group = 24 * days_in_group
    num_groups = int(len(df) // hours_in_group)
    num_profiles = len(df.columns)

    shaped_arr = (
        # one row per historical hour, one column per profile
        df.to_numpy()[: num_groups * hours_in_group, :]
        # convert to one layer per group (e.g., week of data), with each layer
        # having one column per profile and one row per hour-in-group
        .reshape(num_groups, hours_in_group, num_profiles)
        # rearrange to one layer per profile, each with one column per group
        # (e.g., week) and one row per hour-in-group
        .transpose(2, 1, 0)
        # convert to one row per profile per hour-in-group, one col per
        # group
        .reshape(num_profiles * hours_in_group, num_groups)
    )

    result = pd.DataFrame(
        shaped_arr, columns=[f"p{j+1}" for j in range(1, num_groups + 1)]
    )
    return result


# import line_profiler


# @line_profiler.profile
def kmeans_time_clustering(
    resource_profiles,
    load_profiles,
    days_in_group,
    num_clusters,
    include_peak_day=True,
    load_weight=1,
    variable_resources_only=True,
    n_init=100,
):
    """Reduce the number of hours in load and resource variability timeseries using
    kmeans clustering.

    This script is adapted from work originally created by Dharik Mallapragada. For more
    information see:
    - Mallapragada, D. S., Papageorgiou, D. J., Venkatesh, A., Lara, C. L., & Grossmann,
    I. E. (2018). Impact of model resolution on scenario outcomes for electricity sector
    system expansion. Energy, 163, 1231–1244.
    https://doi.org/10.1016/j.energy.2018.08.015
    - Mallapragada, D. S., Sepulveda, N. A., & Jenkins, J. D. (2020). Long-run system
    value of battery energy storage in future grids with increasing wind and solar
    generation. Applied Energy, 275, 115390.
    https://doi.org/10.1016/j.apenergy.2020.115390


    Parameters
    ----------
    resource_profiles : DataFrame
        Hourly generation profiles for all resources. Each column is a resource with
        a unique name, each row is a consecutive hour.
    load_profiles : DataFrame
        Hourly demand profiles of load. Each column is a region with a unique name. each
        row is a consecutive hour.
    days_in_group : int
        The number of 24 hour periods included in each group/cluster
    num_clusters : int
        The number of clusters to include in the output
    include_peak_day : bool, optional
        If the days with system peak demand should be included in outputs, by default
        True
    load_weight : int, optional
        A weighting factor for load profiles during clustering, by default 1
    variable_resources_only : bool, optional
        If clustering should only consider resources with variable (non-zero standard
        deviation) profiles, by default True
    n_init : int, optional
        Parameter for k-means clustering.

    Returns
    -------
    (dict, list, list)
        This function returns multiple items. The dict has keys ['load_profiles',
        'resource_profiles', 'ClusterWeights', 'AnnualGenScaleFactor', 'RMSE', and
        'AnnualProfile']

        The first list has strings with the order of periods selected e.g. ['p42','p26',
        'p3', 'p13', 'p32', 'p8'].

        The second list has integer weights of each cluster.
    """
    logger.info(
        f"Reducing time domain from {len(load_profiles)} hours to representative periods"
    )
    # In cases where each cluster is selected exactly once, skip the clustering entirely
    if len(load_profiles) < days_in_group * (num_clusters + 1) * 24:
        (
            load_df,
            resource_df,
            EachClusterWeight,
            time_series_mapping,
            EachClusterRepPoint,
        ) = max_rep_periods(
            resource_profiles=resource_profiles,
            load_profiles=load_profiles,
            days_in_group=days_in_group,
            num_clusters=num_clusters,
        )
        rep_period_map = {
            i + 1: int(p[1:]) for i, p in enumerate(EachClusterRepPoint["slot"])
        }
        time_series_mapping["Rep_Period"] = time_series_mapping["Rep_Period_Index"].map(
            rep_period_map
        )
        return (
            {
                "load_profiles": load_df,  # Scaled Output Load and Renewables profiles for the sampled representative groupings
                "resource_profiles": resource_df,
                "ClusterWeights": EachClusterWeight,  # Weight of each for the representative groupings
                "AnnualGenScaleFactor": 1,  # Scale factor used to adjust load output to match annual generation of original data
                "RMSE": None,  # Root mean square error between full year data and modeled full year data (duration curves)
                "AnnualProfile": None,
                "time_series_mapping": time_series_mapping,
            },
            EachClusterRepPoint,
            EachClusterWeight,
        )

    resource_col_names = resource_profiles.columns
    if variable_resources_only:
        input_std = resource_profiles.describe().loc["std", :]
        var_col_names = input_std[input_std > 0].index.to_list()
        resource_profiles = resource_profiles.loc[:, var_col_names]

    # Initialize dataframes to store final and intermediate data in

    input_data = pd.concat(
        [
            load_profiles.reset_index(drop=True),
            resource_profiles.reset_index(drop=True),
        ],
        axis=1,
    )
    input_data = input_data.reset_index(drop=True)
    original_col_names = input_data.columns.tolist()
    # CAUTION: Load Column lables should be named with the phrase "Load_"
    load_col_names = load_profiles.columns

    # Columns to be reported in output files
    new_col_names = input_data.columns.tolist() + ["GrpWeight"]
    # Dataframe storing final outputs
    final_output_data = pd.DataFrame(columns=new_col_names)

    # Dataframe storing normalized inputs
    norm_tseries = pd.DataFrame(columns=original_col_names)

    # Normalized all load and renewables data 0 and LoadWeight, All Renewables b/w 0
    # and 1
    norm_tseries = pd.DataFrame(
        data=minmax_scale(input_data), columns=input_data.columns
    )
    norm_tseries.loc[:, load_col_names] *= load_weight

    # Identify hour with maximum system wide load
    hr_maxSysLoad = input_data.loc[:, load_col_names].sum(axis=1).idxmax()

    ################################
    # Convert data from one row per historical hour by one column per profile
    # (resource or load) to one column per possible group (defined by
    # days_in_group) with profiles concatenated. The columns are the data points
    # that will be used for kmeans clustering, e.g., each column holds one week of
    # historical data for all resources if the groups are one week long.

    # Variable names for the concatenated column (one row per load or resource,
    # per hour in group)
    ConcatenatedRowNames = pd.Series(
        np.repeat(norm_tseries.columns, days_in_group * 24)
    )
    #  Create a new dataframe storing aggregated load and renewables time series
    ModifiedDataNormalized = make_time_groups(norm_tseries, days_in_group)
    # Original data organized in concatenated column
    ModifiedData = make_time_groups(input_data, days_in_group)

    # Eliminate grouping including the hour with largest system load (GW) - this
    # group will be manually included in the outputs
    if include_peak_day:
        GroupingwithPeakLoad = ["p" + str(int(hr_maxSysLoad / 24 / days_in_group + 1))]
        ClusteringInputDF = ModifiedDataNormalized.drop(GroupingwithPeakLoad, axis=1)
    else:
        ClusteringInputDF = ModifiedDataNormalized

    ################################## k-means clustering process
    # create Kmeans clustering model and specify the number of clusters gathered
    # number of replications =100, squared euclidean distance

    if include_peak_day:  # If peak day in cluster, generate one less cluster
        num_clusters = num_clusters - 1

    # K-means clustering with n_init trials with randomly selected starting values
    if num_clusters >= 1:  # don't cluster if user only wants 1 peak day
        model = KMeans(
            n_clusters=num_clusters, n_init=n_init, init="k-means++", random_state=42
        )
        model.fit(ClusteringInputDF.values.transpose())

    # Store clustered data
    # Create an empty list storing weight of each cluster
    EachClusterWeight = [None] * num_clusters

    # Create an empty list storing name of each data point
    EachClusterRepPoint = [None] * num_clusters

    # Create time_series_mapping dataframe showing the mapping between
    # historical periods and representative periods (e.g., show which
    # historical week/groups are assigned to each 1-week sample cluster):
    # Period_Index   Rep_Period_Index
    # 1              1
    # 2              1
    # ...
    # 7121           7

    period_index = []
    rep_period_index = []

    for k in range(num_clusters):
        # True for all columns (aka groups, points or periods) assigned to
        # cluster k
        mask = model.labels_ == k

        # Number of points in kth cluster (e.g., label=0)
        EachClusterWeight[k] = mask.sum()

        # Names of points belonging to cluster k
        cluster_cols = ClusteringInputDF.columns[mask]

        # Compute Euclidean distance of each point in cluster k from centroid of
        # the cluster
        dists = np.linalg.norm(
            ClusteringInputDF.iloc[:, mask].to_numpy(copy=False).T
            - model.cluster_centers_[k],
            axis=1,
        )

        # Select name of column with the smallest euclidean distance to the mean
        EachClusterRepPoint[k] = cluster_cols[np.argmin(dists)]

        # Create a list that matches each period (e.g., week) in the full dataset
        # to a representative period; this converts column names like 'p121' to
        # period indexes like 121
        period_index_k = [int(c[1:]) for c in cluster_cols]
        rep_period_index_k = [k + 1] * len(cluster_cols)

        period_index.extend(period_index_k)
        rep_period_index.extend(rep_period_index_k)

    if include_peak_day:
        # appending the week representing peak load
        period_index.append(int(GroupingwithPeakLoad[0][1:]))
        rep_period_index.append(num_clusters + 1)

    # same CSV file that will be used in GenX
    time_series_mapping = (
        pd.DataFrame(
            {
                "Period_Index": period_index,
                "Rep_Period_Index": rep_period_index,
            }
        )
        .sort_values(by=["Period_Index"])
        .reset_index(drop=True)
    )

    # extract month corresponding to each time slot
    time_series_mapping["Month"] = 0
    for Period_Index in time_series_mapping["Period_Index"]:
        dayOfYear = (days_in_group * Period_Index - 1) % 365 + 1
        d = datetime.datetime.strptime("{} {}".format(dayOfYear, 2011), "%j %Y")
        time_series_mapping["Month"][Period_Index - 1] = d.month

    # Store selected groupings in a new data frame with appropriate dimensions
    # (E.g. load in GW)
    ClusterOutputDataTemp = ModifiedData[EachClusterRepPoint]

    # Select rows corresponding to Load in excluded subperiods and exclude them from
    # scale factor calculation
    NRowsLoad = len(load_col_names)
    # Excluding grouping with peak hr from scale factor calculation
    if include_peak_day:
        Actualdata = ModifiedData.loc[0 : 24 * days_in_group * NRowsLoad - 1, :].drop(
            GroupingwithPeakLoad, axis=1
        )
    else:
        Actualdata = ModifiedData.loc[0 : 24 * days_in_group * NRowsLoad - 1, :]

    # Scale factor to adjust total generation in original data set to be equal
    # to scaled up total generation in sampled data set
    SampleweeksAnnualTWh = sum(
        [
            ClusterOutputDataTemp.loc[
                0 : 24 * days_in_group * NRowsLoad - 1, EachClusterRepPoint[j]
            ].sum()
            * EachClusterWeight[j]
            for j in range(num_clusters)
        ]
    )
    ScaleFactor = (
        Actualdata.loc[0 : 24 * days_in_group * NRowsLoad - 1, :].sum().sum()
        / SampleweeksAnnualTWh
    )

    # Updated load values in GW
    ClusterOutputDataTemp.loc[0 : 24 * days_in_group * NRowsLoad - 1, :] = (
        ScaleFactor
        * ClusterOutputDataTemp.loc[0 : 24 * days_in_group * NRowsLoad - 1, :]
    )

    # Add the grouping with the peak hour back into the cluster if that was
    # excluded from the clustering
    if include_peak_day:
        EachClusterRepPoint = EachClusterRepPoint + GroupingwithPeakLoad
        EachClusterWeight = EachClusterWeight + [1]
        ClusterOutputData = pd.concat(
            [ClusterOutputDataTemp, ModifiedData[GroupingwithPeakLoad]],
            axis=1,
            sort=False,
        )
    else:
        ClusterOutputData = ClusterOutputDataTemp

    # Store weights for each selected hour  Number of days *24, for each week
    ClusteredWeights = pd.DataFrame(
        EachClusterWeight * np.ones([days_in_group * 24, len(EachClusterWeight)]),
        columns=EachClusterRepPoint,
    )

    # Store weights in final output data column
    final_output_data["GrpWeight"] = ClusteredWeights.melt(id_vars=None)["value"]

    # Regenerate data organized by time series (columns) and representative time
    # periods (hours)
    for i in range(len(new_col_names) - 1):
        final_output_data[new_col_names[i]] = ClusterOutputData.loc[
            ConcatenatedRowNames == new_col_names[i], :
        ].melt(id_vars=None)["value"]

    # Calculate error metrics and annual profile

    # Make FullLengthOutputs dataframe that repeats the selected sample days the
    # number of times specified in EachClusterWeight
    block_ids = np.repeat(range(len(EachClusterWeight)), EachClusterWeight)
    row_idx = (
        block_ids[:, None] * days_in_group * 24 + np.arange(days_in_group * 24)
    ).ravel()
    FullLengthOutputs = final_output_data.iloc[row_idx].reset_index(drop=True)

    # Mean square error between the duration curves of each time series.
    # Only considers the points considered in the k-means clustering
    # - ignoring any days dropped from original data set due to rounding
    in_ldc = np.sort(
        input_data.loc[:, original_col_names]
        .iloc[: len(FullLengthOutputs)]
        .to_numpy(copy=False),
        axis=0,
    )
    out_ldc = np.sort(
        FullLengthOutputs.loc[:, original_col_names].to_numpy(copy=False), axis=0
    )
    # Should this switch to RMSE instead of MSE? e.g., np.sqrt(np.linalg.norm(diff, axis=0))
    RMSE = dict(zip(original_col_names, np.linalg.norm(in_ldc - out_ldc, axis=0)))

    # prepare time-reduced outputs
    load_df = final_output_data.loc[:, load_col_names]
    # if variable_only:
    resource_df = pd.DataFrame(
        columns=resource_col_names, index=final_output_data.index
    )
    resource_df.loc[:, var_col_names] = final_output_data.loc[:, var_col_names]
    resource_df = resource_df.fillna(value=1)

    rep_period_map = {i + 1: int(p[1:]) for i, p in enumerate(EachClusterRepPoint)}
    time_series_mapping["Rep_Period"] = time_series_mapping["Rep_Period_Index"].map(
        rep_period_map
    )
    EachClusterRepPoint = pd.DataFrame(EachClusterRepPoint, columns=["slot"])

    # Calculate weights that add up to 365 days, so these samples add up to
    # a single representative year
    reweight = 365 / (sum(EachClusterWeight) * days_in_group)
    AnnualClusterWeight = [w * reweight for w in EachClusterWeight]
    return (
        {
            "load_profiles": load_df,  # Scaled Output Load and Renewables profiles for the sampled representative groupings
            "resource_profiles": resource_df,
            "ClusterWeights": AnnualClusterWeight,  # Weight of each for the representative groupings
            "AnnualGenScaleFactor": ScaleFactor,  # Scale factor used to adjust load output to match annual generation of original data
            "RMSE": RMSE,  # Mean square error between full year data and modeled full year data (duration curves)
            "AnnualProfile": FullLengthOutputs,
            "time_series_mapping": time_series_mapping,
        },
        EachClusterRepPoint,
        AnnualClusterWeight,
    )  # Modeled duration curves GW
