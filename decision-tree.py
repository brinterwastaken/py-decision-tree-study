import argparse
import csv
from dataclasses import dataclass, field
import math
from typing import Literal
from mermaid.graph import Graph
from mermaid import Mermaid

type TrainingData = list[dict[str, str]]


@dataclass
class TreeNode:
    name: str
    node_type: Literal["decision", "leaf"]
    links: dict[str, int] = field(default_factory=dict)


"""
Data Format:
CSV
param1, param2, param3 .... decision
val1,   val2,   val3,  .... True | False
"""

create_md = False


def accept_data(file: str):
    params = []
    raw_data = []
    try:
        with open(file, "r") as f:
            raw_data = [x for x in csv.reader(f)]
            params = [x.lower() for x in raw_data.pop(0)]

            if "decision" not in params:
                raise Exception("Invalid data supplied: No decision parameter.")

            for row in raw_data:
                if len(row) != len(params):
                    raise Exception(
                        "Invalid data supplied: Some row(s) do not have values for all parameters."
                    )

    except FileNotFoundError:
        print("Error: File Not Found!")

    data: TrainingData = [
        {params[n]: row[n] for n in range(len(params))} for row in raw_data
    ]

    for row in data:
        if row["decision"].lower() not in ["true", "false"]:
            raise Exception(
                "Invalid data supplied: Decision parameter has invalid value."
            )

    return data


def unique_values(data: TrainingData, param: str | None = None):
    unique_vals: dict[str, str] = {}

    for param in ([param] if param is not None else data[0].keys()):
        param_unique_vals = []
        for row in data:
            if row[param] not in param_unique_vals:
                param_unique_vals.append(row[param])
        unique_vals[param] = param_unique_vals

    return unique_vals


def subdata(data: TrainingData, filters: dict[str, str]):
    subdata = data
    for param, val in filters.items():
        subdata: TrainingData = [
            {k: v for k, v in row.items() if k != param}
            for row in subdata
            if row[param] == val
        ]
    return subdata


def count_decisions(data: TrainingData):
    true_count = false_count = 0

    for row in data:
        if row["decision"].lower() == "true":
            true_count += 1
        else:
            false_count += 1

    return true_count, false_count, true_count + false_count


def gains(data: TrainingData):
    unique_vals = unique_values(data)

    ovr_counts = count_decisions(data)

    ovr_entropy = -1 * (
        (
            ovr_counts[0] / ovr_counts[2] * math.log2(ovr_counts[0] / ovr_counts[2])
            if ovr_counts[0] > 0
            else 0
        )
        + (
            ovr_counts[1] / ovr_counts[2] * math.log2(ovr_counts[1] / ovr_counts[2])
            if ovr_counts[1] > 0
            else 0
        )
    )

    if create_md:
        print(f"### Overall Entropy: {ovr_entropy:.5f}")

    gains = {}

    for param in [x for x in data[0].keys() if x != "decision"]:
        param_gain = ovr_entropy
        param_total = 0
        entropies = []
        if create_md:
            print(f"### Entropies for `{param}`:")

        for unique_val in unique_vals[param]:
            counts = count_decisions(subdata(data, {param: unique_val}))
            entropy = -1 * (
                (
                    counts[0] / counts[2] * math.log2(counts[0] / counts[2])
                    if counts[0] > 0
                    else 0
                )
                + (
                    counts[1] / counts[2] * math.log2(counts[1] / counts[2])
                    if counts[1] > 0
                    else 0
                )
            )
            entropies.append(
                counts[2] * entropy
            )  # Pre-multiplying entropy with no. of associated entries
            param_total += counts[2]

            if create_md:
                print(f"- {param}: {unique_val} [+{counts[0]}, -{counts[1]}]")
                entropy_eqn = f"$$ \\text{{Entropy}} = "
                entropy_eqn += f"-\\frac{{{counts[0]}}}{{{counts[2]}}}\\log_2 \\frac{{{counts[0]}}}{{{counts[2]}}}"
                entropy_eqn += f"-\\frac{{{counts[1]}}}{{{counts[2]}}}\\log_2 \\frac{{{counts[1]}}}{{{counts[2]}}}"
                entropy_eqn += f" = {-entropy if entropy == 0 else entropy:.5f}$$"
                print(f"    {entropy_eqn}")
        for i in entropies:
            param_gain -= i / param_total
        if create_md:
            print(f"Gain for **{param}** = {param_gain:.5f}\n")
        gains[param] = param_gain

    return gains


def create_tree(og_data: TrainingData):

    nodes: dict[int, TreeNode] = {}
    queue: list[tuple] = []
    node_id = 0

    while queue or not nodes:
        next_item = queue.pop() if queue else None
        filters = next_item[0] if next_item else {}
        data = subdata(og_data, filters) if next_item else og_data

        if create_md:
            print()
            if next_item:
                print(f"## Gain calculation under filter: {next_item[0]}")
            else:
                print(f"## Gain calculation for root node")

        highest_gain_param: str = max(gains(data).items(), key=lambda x: x[1])[0]

        if create_md:
            print(f"### Highest Gain Parameter: `{highest_gain_param}`")

        decision_node_id = next_item[1] if next_item else node_id
        nodes[decision_node_id] = TreeNode(highest_gain_param, "decision")

        for value in unique_values(data, highest_gain_param)[highest_gain_param]:

            decision_counts = count_decisions(
                subdata(og_data, filters | {highest_gain_param: value})
            )

            if decision_counts[0] == decision_counts[2]:
                node_id += 1
                nodes[node_id] = TreeNode("True", "leaf")
                nodes[decision_node_id].links[value] = node_id

            elif decision_counts[1] == decision_counts[2]:
                node_id += 1
                nodes[node_id] = TreeNode("False", "leaf")
                nodes[decision_node_id].links[value] = node_id

            else:
                node_id += 1
                nodes[decision_node_id].links[value] = node_id
                queue += [(filters | {highest_gain_param: value}, node_id)]

    return nodes


def create_mermaid_diagram(
    nodes: dict[int, TreeNode], title: str, output: str | None = None
):
    mermaid_code = f"---\ntitle: {title}\n---\nflowchart TD"

    for node_id, node in nodes.items():
        if node.node_type == "decision":
            mermaid_code += f"\n    n{node_id}[{node.name.title()}]"
            for link, link_to in node.links.items():
                mermaid_code += f"\n    n{node_id} -->|{link}| n{link_to}"
        else:
            mermaid_code += f"\n    n{node_id}([{node.name.title()}])"

    if create_md:
        print("## Decision Tree Diagram")
        print("```mermaid")
        print(mermaid_code)
        print("```")

    else:
        print("Mermaid Code:")
        print(mermaid_code)
        graph = Graph(title=title, script=mermaid_code)
        graph_filename = (
            output
            if output is not None
            else "-".join(title.lower().strip().split(" ")) + ".svg"
        )
        Mermaid(graph).to_svg(graph_filename)

        print("Diagram Saved To: ", graph_filename)


def main():
    parser = argparse.ArgumentParser(
        description="Create a decision tree based on the supplied training data."
    )

    parser.add_argument(
        "-d", "--data-file", help="training data file (csv format)", required=True
    )
    parser.add_argument(
        "-t", "--title", help="title for decision tree diagram", required=True
    )
    parser.add_argument("-o", "--output-file", help="file to save diagram to")
    parser.add_argument(
        "-m",
        "--markdown",
        help="create a markdown file describing the process",
        action="store_true",
    )

    args = parser.parse_args()

    global create_md
    create_md = args.markdown
    if create_md:
        print(f"# Decision Tree for `{args.data_file}`")
        print(
            "Created with: [brinterwastaken/py-decision-tree-study](https://github.com/brinterwastaken/py-decision-tree-study) (GitHub)"
        )

    data = accept_data(args.data_file)

    nodes = create_tree(data)

    if not create_md:
        print("Nodes:", nodes)
    create_mermaid_diagram(nodes, args.title)


if __name__ == "__main__":
    main()
