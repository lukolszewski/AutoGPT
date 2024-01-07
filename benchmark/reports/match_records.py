import glob
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from pydantic import BaseModel, Field

# from agbenchmark.reports.processing.report_types import Report, SuiteTest


class Metrics(BaseModel):
    difficulty: str
    success: bool
    success_percent: float = Field(..., alias="success_%")
    run_time: Optional[str] = None
    fail_reason: Optional[str] = None
    attempted: Optional[bool] = None


class MetricsOverall(BaseModel):
    run_time: str
    highest_difficulty: str
    percentage: Optional[float] = None


class Test(BaseModel):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None


class SuiteTest(BaseModel):
    data_path: str
    metrics: MetricsOverall
    tests: Dict[str, Test]
    category: Optional[List[str]] = None
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None


class Report(BaseModel):
    command: str
    completion_time: str
    benchmark_start_time: str
    metrics: MetricsOverall
    tests: Dict[str, Union[Test, SuiteTest]]
    config: Dict[str, str | dict[str, str]]


def get_reports():
    # Initialize an empty list to store the report data
    report_data = []

    # Get the current working directory
    current_dir = os.getcwd()

    # Check if the current directory ends with 'reports'
    if current_dir.endswith("reports"):
        reports_dir = "/"
    else:
        reports_dir = "reports"

    # Iterate over all agent directories in the reports directory
    for agent_name in os.listdir(reports_dir):
        if agent_name is None:
            continue
        agent_dir = os.path.join(reports_dir, agent_name)

        # Check if the item is a directory (an agent directory)
        if os.path.isdir(agent_dir):
            # Construct the path to the report.json file
            # Get all directories and files, but note that this will also include any file, not just directories.
            run_dirs = glob.glob(os.path.join(agent_dir, "*"))

            # Get all json files starting with 'file'
            # old_report_files = glob.glob(os.path.join(agent_dir, "file*.json"))

            # For each run directory, add the report.json to the end
            # Only include the path if it's actually a directory
            report_files = [
                os.path.join(run_dir, "report.json")
                for run_dir in run_dirs
                if os.path.isdir(run_dir)
            ]
            # old_report_files already contains the full paths, so no need to join again
            # report_files = report_files + old_report_files
            for report_file in report_files:
                # Check if the report.json file exists
                if os.path.isfile(report_file):
                    # Open the report.json file
                    with open(report_file, "r") as f:
                        # Load the JSON data from the file
                        json_data = json.load(f)
                        print(f"Processing {report_file}")
                        report = Report.parse_obj(json_data)

                        for test_name, test_data in report.tests.items():
                            test_json = {
                                "agent": agent_name.lower(),
                                "benchmark_start_time": report.benchmark_start_time,
                            }

                            if isinstance(test_data, SuiteTest):
                                if (
                                    test_data.category
                                ):  # this means it's a same task test
                                    test_json["challenge"] = test_name
                                    test_json["attempted"] = test_data.tests[
                                        list(test_data.tests.keys())[0]
                                    ].metrics.attempted
                                    test_json["categories"] = ", ".join(
                                        test_data.category
                                    )
                                    test_json["task"] = test_data.task
                                    test_json["success"] = test_data.metrics.percentage
                                    test_json[
                                        "difficulty"
                                    ] = test_data.metrics.highest_difficulty
                                    test_json[
                                        "success_%"
                                    ] = test_data.metrics.percentage
                                    test_json["run_time"] = test_data.metrics.run_time
                                    test_json["is_regression"] = test_data.tests[
                                        list(test_data.tests.keys())[0]
                                    ].is_regression
                                else:  # separate tasks in 1 suite
                                    for (
                                        suite_test_name,
                                        suite_data,
                                    ) in test_data.tests.items():
                                        test_json["challenge"] = suite_test_name
                                        test_json[
                                            "attempted"
                                        ] = suite_data.metrics.attempted
                                        test_json["categories"] = ", ".join(
                                            suite_data.category
                                        )
                                        test_json["task"] = suite_data.task
                                        test_json["success"] = (
                                            100.0 if suite_data.metrics.success else 0
                                        )
                                        test_json[
                                            "difficulty"
                                        ] = suite_data.metrics.difficulty
                                        test_json[
                                            "success_%"
                                        ] = suite_data.metrics.success_percentage
                                        test_json[
                                            "run_time"
                                        ] = suite_data.metrics.run_time
                                        test_json[
                                            "is_regression"
                                        ] = suite_data.is_regression

                            else:
                                test_json["challenge"] = test_name
                                test_json["attempted"] = test_data.metrics.attempted
                                test_json["categories"] = ", ".join(test_data.category)
                                test_json["task"] = test_data.task
                                test_json["success"] = (
                                    100.0 if test_data.metrics.success else 0
                                )
                                test_json["difficulty"] = test_data.metrics.difficulty
                                test_json[
                                    "success_%"
                                ] = test_data.metrics.success_percentage
                                test_json["run_time"] = test_data.metrics.run_time
                                test_json["is_regression"] = test_data.is_regression

                            report_data.append(test_json)

    return pd.DataFrame(report_data)

if os.path.exists("raw_reports.pkl"):
    reports_df = pd.read_pickle("raw_reports.pkl")
else:
    reports_df = get_reports()
    reports_df.to_pickle("raw_reports.pkl")

def try_formats(date_str):
    formats = ["%Y-%m-%d-%H:%M", "%Y-%m-%dT%H:%M:%S%z"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    return None

reports_df["benchmark_start_time"] = pd.to_datetime(
    reports_df["benchmark_start_time"].apply(try_formats), utc=True
)
reports_df = reports_df.dropna(subset=["benchmark_start_time"])

assert pd.api.types.is_datetime64_any_dtype(
    reports_df["benchmark_start_time"]
), "benchmark_start_time in reports_df is not datetime"

reports_df["report_time"] = reports_df["benchmark_start_time"]

df = reports_df

df.to_pickle("df.pkl")
print(df.info())
print("Data saved to df.pkl")
print("To load the data use: df = pd.read_pickle('df.pkl')")
