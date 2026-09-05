import pandas as pd
import great_expectations as gx


EXPECTED_COLUMNS = [
    "timestamp",
    "machine_id",
    "ambient_temperature",
    "humidity",
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "torque",
    "operating_hours",
    "failure",
]


SENSOR_COLUMNS = [
    "ambient_temperature",
    "humidity",
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "torque",
    "operating_hours",
]


def create_expectation_suite():
    """
    Create the Great Expectations suite for the
    predictive-maintenance sensor dataset.
    """

    suite = gx.ExpectationSuite(
        name="predictive_maintenance_sensor_suite"
    )

    # ---------------------------------------------------------
    # Schema validation
    # ---------------------------------------------------------

    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToEqual(
            value=len(EXPECTED_COLUMNS)
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS,
            exact_match=True,
        )
    )

    # ---------------------------------------------------------
    # Missing-value validation
    # ---------------------------------------------------------

    for column in EXPECTED_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=column
            )
        )

    # ---------------------------------------------------------
    # Machine ID validation
    # ---------------------------------------------------------

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="machine_id",
            min_value=1,
        )
    )

    # ---------------------------------------------------------
    # Target validation
    # ---------------------------------------------------------

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="failure",
            value_set=[0, 1],
        )
    )

    # ---------------------------------------------------------
    # Sensor domain validation
    # ---------------------------------------------------------

    # Humidity is represented as a percentage.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="humidity",
            min_value=0,
            max_value=100,
        )
    )

    # These operational measurements cannot be negative.
    non_negative_columns = [
        "vibration",
        "pressure",
        "rotational_speed",
        "torque",
        "operating_hours",
    ]

    for column in non_negative_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=0,
            )
        )

    return suite


def validate_with_great_expectations(
    data: pd.DataFrame,
) -> bool:
    """
    Validate a pandas DataFrame using Great Expectations.

    Returns:
        True if all expectations pass.

    Raises:
        ValueError: If validation fails.
    """

    # Create an in-memory GX Data Context.
    context = gx.get_context(mode="ephemeral")

    # Create a Pandas Data Source.
    data_source = context.data_sources.add_pandas(
        name="predictive_maintenance_source"
    )

    # Create a Data Asset for the DataFrame.
    data_asset = data_source.add_dataframe_asset(
        name="sensor_data_asset"
    )

    # Create a Batch Definition for the complete DataFrame.
    batch_definition = (
        data_asset.add_batch_definition_whole_dataframe(
            "sensor_data_batch"
        )
    )

    # Create the Expectation Suite.
    suite = create_expectation_suite()

    # Supply the DataFrame at runtime.
    batch = batch_definition.get_batch(
        batch_parameters={
            "dataframe": data
        }
    )

    # Validate the DataFrame against the suite.
    validation_result = batch.validate(suite)

    print("\nGreat Expectations Validation")
    print("=" * 40)

    print(
        f"Validation successful: "
        f"{validation_result.success}"
    )

    print(
        f"Expectations evaluated: "
        f"{len(validation_result.results)}"
    )

    passed = sum(
        result.success
        for result in validation_result.results
    )

    failed = len(validation_result.results) - passed

    print(f"Expectations passed: {passed}")
    print(f"Expectations failed: {failed}")

    if not validation_result.success:
        raise ValueError(
            "Great Expectations validation failed."
        )

    print("Great Expectations validation passed.")

    return True