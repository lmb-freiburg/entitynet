from importlib.resources import files

import pytest

import entitynet
from entitynet.litext.csv_logs_reader import CsvLoggerLogsReader


def test_csv_logs_reader_data_shapes():
    """Test that the CSV logs reader correctly loads and processes data shapes."""
    csvllr = CsvLoggerLogsReader(files(entitynet) / "testdata/example_run")

    # Test train phase data
    assert "train" in csvllr.output_total
    train_df = csvllr.output_total["train"]
    assert train_df is not None
    assert train_df.shape == (35, 5)
    assert set(train_df.columns) == {"epoch", "step", "lr", "lr_g1", "train_loss"}
    assert not train_df.isna().any().any()  # No NA values in any column

    # Test val phase data
    assert "val" in csvllr.output_total
    val_df = csvllr.output_total["val"]
    assert val_df is not None
    assert val_df.shape == (2, 5)
    assert not val_df.isna().any().any()  # No NA values in any column

    # Test test phase data
    assert "test" in csvllr.output_total
    assert csvllr.output_total["test"] is None


def test_csv_logs_reader_metric_retrieval():
    """Test that metrics can be correctly retrieved from the logs."""
    csvllr = CsvLoggerLogsReader(files(entitynet) / "testdata/example_run")

    # Test getting a specific metric value
    phase = "val"
    metric = "val_loss"
    value = csvllr.get_epoch_metric(phase, 14, metric)
    assert value is not None
    assert isinstance(value, float)
    assert abs(value - 2.970) < 1e-2  # Check value matches expected output

    # Test getting non-existent metric
    non_existent_metric = "non_existent_metric"
    value = csvllr.get_epoch_metric(phase, 0, non_existent_metric)
    assert value is None

    # Test getting metric for non-existent phase
    with pytest.raises(KeyError):
        non_existent_phase = "non_existent_phase"
        value = csvllr.get_epoch_metric(non_existent_phase, 0, metric)
        assert value is None
