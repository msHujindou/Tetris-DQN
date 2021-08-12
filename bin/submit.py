"""
Submit job into AML
"""
import os
import tempfile
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
import json

docker_url = ""


def gen_env(requirement_file):
    conda_template = """name: dqn38
channels:
    - defaults
dependencies:
    - python=3.8
    - pip=21.0.1
    - pip:"""
    flag = False
    myenv = None
    try:
        with open(requirement_file, "r") as fr:
            for line in fr:
                if line and line.strip() and not line.strip().startswith("#"):
                    conda_template += f"{os.linesep}        - {line.strip()}"
                    flag = True
    except:
        print("#### error occur for requirement file")
        return None
    if flag:
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".yml")
        tmpf.write(conda_template.encode("utf-8"))
        tmpf.close()
        myenv = Environment.from_conda_specification(
            name="dqn38",
            file_path=tmpf.name,
        )
        myenv.docker.enabled = True
        myenv.docker.base_image = docker_url

        os.unlink(tmpf.name)
    return myenv


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    ws = Workspace.from_config()
    with open(os.path.join(current_path, "../config.json"), "r") as config_file:
        js = json.load(config_file)
    docker_url = js["docker_url"]
    env = gen_env(os.path.join(current_path, "../requirements.txt"))

    experiment = Experiment(workspace=ws, name="test")
    config = ScriptRunConfig(
        source_directory=os.path.join(current_path, "../src"),
        script="train/train_new.py",
        compute_target="test3",
        environment=env,
    )

    run = experiment.submit(
        config, tags={"1": "从Action_List中删除了Rotate_Down操作", "2": "model的output由4变成了3"}
    )
    print("Run Scheduled : ", run)
