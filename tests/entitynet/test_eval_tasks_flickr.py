import pytest

from typedparser.objects import repr_value

from entitynet.datasets.task_tester import run_task_tester

tasks_and_lens = {
    "flickr30k_test": 1000,
    "flickr30k_val": 1014,
}


@pytest.mark.slow
@pytest.mark.parametrize("task_key", list(tasks_and_lens.keys()))
def test_eval_tasks(task_key):
    print(f"========== {task_key} ==========")
    dataset, loader = run_task_tester(task_key)
    print(f"Created dataset for task {task_key} got length: {len(dataset)}")
    print(repr_value(dataset[0]))
    assert len(dataset) == tasks_and_lens[task_key], str({task_key: len(dataset)})
