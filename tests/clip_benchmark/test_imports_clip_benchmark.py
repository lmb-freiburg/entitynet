import pytest

from packg.testing import ImportFromSourceChecker, apply_visitor, recurse_modules

module_list = list(recurse_modules("clip_benchmark", ignore_tests=True, packages_only=False))
ignore_modules = {"clip_benchmark.datasets.kitti"}


@pytest.mark.parametrize("module", module_list)
def test_imports_from_source(module: str) -> None:
    if module in ignore_modules:
        print(f"Skipping import: {module}")
        return
    print(f"Importing: {module}")
    apply_visitor(
        module=module, visitor=ImportFromSourceChecker(module, ignore_modules=ignore_modules)
    )
