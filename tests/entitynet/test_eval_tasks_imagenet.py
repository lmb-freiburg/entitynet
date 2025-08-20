import pytest

from typedparser.objects import repr_value

from entitynet.datasets.task_tester import run_task_tester

tasks_and_lens = {
    "imgn_1k_val": 50000,
    "imgn_1k_ts": 50000,  # train small (extra val set using train images)
    "imgn_lt_val": 20500,  # val set, only living things
    "imgn_ot_val": 29500,  # val set, only non-living things
    "imgv2_1k_test_temp": 10000,
    "imgr_200_test_temp": 30000,
    "objn_1k_test_temp": 18574,
    "imgsketch_1k_test_temp": 50889,
    "imga_200_test_temp": 7500,
}


@pytest.mark.slow
@pytest.mark.parametrize("task_key", list(tasks_and_lens.keys()))
def test_eval_tasks(task_key):
    print(f"========== {task_key} ==========")
    dataset, loader = run_task_tester(task_key)
    print(f"Created dataset for task {task_key} got length: {len(dataset)}")
    print(repr_value(dataset[0]))
    assert len(dataset) == tasks_and_lens[task_key], str({task_key: len(dataset)})
