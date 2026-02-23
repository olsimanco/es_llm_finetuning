# es-llm-finetune



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

* [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
* [Add files using the command line](https://docs.gitlab.com/topics/git/add_files/#add-files-to-a-git-repository) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.uni-bonn.de/pg_ml_1/es-llm-finetune.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

* [Set up project integrations](https://gitlab.uni-bonn.de/pg_ml_1/es-llm-finetune/-/settings/integrations)

## Collaborate with your team

* [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
* [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
* [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
* [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
* [Set auto-merge](https://docs.gitlab.com/user/project/merge_requests/auto_merge/)

## Test and Deploy

Use the built-in continuous integration in GitLab.

* [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/)
* [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
* [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
* [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
* [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.



---------------------ISH_KODI_OLMES_WRAPPER--------------------------------------------------------------------

 import subprocess

import json

import os

import shutil

import glob



class OlmesWrapper:

def __init__(self, base_model_name, task_name, limit=None):

"""

base_model_name: e.g., "Qwen/Qwen2.5-0.5B"

task_name: e.g., "minerva_math_algebra:bpb::olmes"

limit: (Optional) If set to integer (e.g. 50), only tests 50 questions.

CRITICAL for speed during the evolutionary loop!

"""

self.base_model_name = base_model_name

self.task_name = task_name

self.limit = limit


# We will save temporary adapters here

self.temp_dir = os.path.join("results", "temp_es_adapters")

os.makedirs(self.temp_dir, exist_ok=True)


def get_score(self, peft_model, generation_index, candidate_index):

"""

1. Saves the current candidate model (adapter).

2. Runs OLMES via subprocess.

3. Returns the score.

"""

# --- Step A: Define paths for this specific candidate ---

# We use unique paths so parallel runs don't overwrite each other

run_id = f"gen_{generation_index}_cand_{candidate_index}"

adapter_save_path = os.path.join(self.temp_dir, run_id)

output_path = os.path.join(self.temp_dir, f"{run_id}_output")


# Clean up previous run if exists

if os.path.exists(output_path):

shutil.rmtree(output_path)


# --- Step B: Save the Adapter (The "Soft Prompt") ---

# This writes the `adapter_model.bin` and `adapter_config.json` to disk

peft_model.save_pretrained(adapter_save_path)


# --- Step C: Build the OLMES Command ---

# Based on OLMES Readme: we pass the adapter via --model-args "peft=..."

# This is standard for tools built on lm-evaluation-harness

model_args = f"peft={adapter_save_path},trust_remote_code=True"


cmd = [

"olmes",

"--model",

self.base_model_name,

"--model-args",

model_args,

"--task",

self.task_name,

"--output-dir",

output_path,

]


# Add limit for speed (if configured)

if self.limit:

cmd.extend(["--limit", str(self.limit)])


# --- Step D: Execute Command ---

# capture_output=True keeps your terminal clean.

# set text=True to process output as string

try:

result = subprocess.run(cmd, capture_output=True, text=True, check=True)

except subprocess.CalledProcessError as e:

print(f"OLMES Crashed for {run_id}!")

print("Error Log:", e.stderr)

return -999.0 # Return bad score on failure


# --- Step E: Parse Results ---

# OLMES saves results in the output_dir. We need to find the .json file.

score = self._parse_json_results(output_path)


# Cleanup to save disk space (Optional - disable if debugging)

shutil.rmtree(adapter_save_path)

shutil.rmtree(output_path)


return score


def _parse_json_results(self, output_dir):

"""

Finds the JSON file in the output directory and extracts the metric.

"""

try:

# Recursive search for any .json file in the output dir

json_files = glob.glob(f"{output_dir}/**/*.json", recursive=True)


if not json_files:

print(f"No JSON found in {output_dir}")

return -999.0


# Usually the results file is the largest JSON or the most recent

target_file = json_files[0]


with open(target_file, "r") as f:

data = json.load(f)


# --- EXTRACT METRIC ---

# The structure is usually: results -> task_name -> metric

# We need to handle potential slight naming variations in the JSON key


# 1. Find the key that matches our task

result_keys = list(data.get("results", {}).keys())

if not result_keys:

return -999.0


# Just take the first task found (since we only run 1 task at a time)

task_key = result_keys[0]

metrics = data["results"][task_key]


# 2. Select the metric based on task type

# For BPB (Bits Per Byte) - LOWER is better, so we return negative

if "bpb" in metrics:

return -metrics["bpb"]


# For Accuracy (ARC, MMLU) - HIGHER is better

if "acc_norm" in metrics:

return metrics["acc_norm"]

if "acc" in metrics:

return metrics["acc"]


# Fallback

print(f"Unknown metrics found: {metrics.keys()}")

return 0.0


except Exception as e:

print(f"Parsing Error: {e}")

return -999.0 