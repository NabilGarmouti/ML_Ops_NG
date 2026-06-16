from __future__ import annotations

import pandas as pd

from config import TARGET
from features import clean_data, expected_columns, validate_clean_data


def test_clean_data_and_validate_clean_data():
    raw_df = pd.DataFrame(
        [
            {
                "Response": "1",
                "Age": "44",
                "Region_Code": "28.0",
                "Annual_Premium": "40454",
                "Policy_Sales_Channel": "26",
                "Vintage": "217",
                "Gender": " Male ",
                "Driving_License": "1",
                "Previously_Insured": "0",
                "Vehicle_Age": "> 2 Years",
                "Vehicle_Damage": "Yes",
            }
        ]
    )

    clean_df = clean_data(raw_df)
    validate_clean_data(clean_df)

    assert list(clean_df.columns) == expected_columns()
    assert clean_df.loc[0, "Gender"] == "Male"
    assert clean_df.loc[0, TARGET] == 1

