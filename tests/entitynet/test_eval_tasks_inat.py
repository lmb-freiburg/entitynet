import pytest

from typedparser.objects import repr_value

from entitynet.datasets.task_tester import run_task_tester

tasks_and_lens = {
    "inat19_val": 3030,
    "inat19lat_val": 3030,
    "inat19_tdev": 10100,
    "inat19lat_tdev": 10100,
    "inat21_val": 100000,
    "inat21lat_val": 100000,
    "inat21_tdev": 100000,
    "inat21lat_tdev": 100000,
    "inat19_val_itemp": 3030,  # with clip imagenet prompt templates
}


@pytest.mark.slow
@pytest.mark.parametrize("task_key", list(tasks_and_lens.keys()))
def test_eval_tasks(task_key):
    print(f"========== {task_key} ==========")
    dataset, loader = run_task_tester(task_key)
    print(f"Created dataset for task {task_key} got length: {len(dataset)}")
    print(repr_value(dataset[0]))
    assert len(dataset) == tasks_and_lens[task_key], str({task_key: len(dataset)})
