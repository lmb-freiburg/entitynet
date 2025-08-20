import pytest

from typedparser.objects import repr_value

from entitynet.datasets.task_tester import run_task_tester

tasks_and_lens = {
    "rarespecies_train_com_inettemp": 11983,
    "rarespecies_train_com_none": 11983,
    "rarespecies_train_sci_inettemp": 11983,
    "rarespecies_train_sci_none": 11983,
    "rarespecies_train_scicom_inettemp": 11983,
    "rarespecies_train_scicom_none": 11983,
    "rarespecies_train_tax_inettemp": 11983,
    "rarespecies_train_tax_none": 11983,
    "rarespecies_train_taxcom_inettemp": 11983,
    "rarespecies_train_taxcom_none": 11983,
}


@pytest.mark.slow
@pytest.mark.parametrize("task_key", list(tasks_and_lens.keys()))
def test_eval_tasks(task_key):
    print(f"========== {task_key} ==========")
    dataset, loader = run_task_tester(task_key)
    print(f"Created dataset for task {task_key} got length: {len(dataset)}")
    print(repr_value(dataset[0]))
    assert len(dataset) == tasks_and_lens[task_key], str({task_key: len(dataset)})
