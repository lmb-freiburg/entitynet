import importlib
import importlib.util
import shutil
import sys


def main():
    lib_module_name = __file__.replace("\\", "/").split("/")[-2]
    if len(sys.argv) <= 1:
        print(f"Usage: {lib_module_name} <module> <*args>\n")
        from packg.packaging import get_modules_for_autocomplete
        from packg.strings import format_pseudo_table

        output_modules = get_modules_for_autocomplete(lib_module_name, run_dir=None)
        terminal_width = shutil.get_terminal_size().columns
        print(format_pseudo_table(output_modules, max_width=terminal_width))
        sys.exit(1)
    target_module = sys.argv[1]
    args = sys.argv[2:]
    # load the spec to modify sys.argv before importing, otherwise it's too late
    module_to_load = f"{lib_module_name}.{target_module}"
    spec = importlib.util.find_spec(module_to_load)
    if spec is None:
        raise ValueError(f"Module {module_to_load} not found")
    origin = spec.origin
    sys.argv = [origin] + args
    # import the module and run the main function
    module = importlib.import_module(module_to_load)
    module.main()


if __name__ == "__main__":
    main()
