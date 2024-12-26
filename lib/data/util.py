import pandas as pd
from pandas import DataFrame, Series
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from lib.data.panda_ds import PandasClsDs
from torch.utils.data import DataLoader


def dataset_split(
    x: DataFrame,
    y: Series,
    test_train_ratio: float,
    valid_train_ratio: float,
    random_seed: int,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    all_x_train, x_test, all_y_train, y_test = train_test_split(
        x, y, test_size=test_train_ratio, random_state=random_seed
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        all_x_train, all_y_train, test_size=valid_train_ratio, random_state=random_seed
    )

    return (
        DataLoader(PandasClsDs(x_train, y_train), batch_size=batch_size),
        DataLoader(PandasClsDs(x_valid, y_valid), batch_size=batch_size),
        DataLoader(PandasClsDs(x_test, y_test), batch_size=batch_size),
    )


def scale_df(df: DataFrame, ignore_columns: list[str]) -> DataFrame:
    scale = StandardScaler()

    ignored = df[ignore_columns]
    to_scale = df.drop(columns=ignore_columns)

    scaled = scale.fit_transform(to_scale)
    scaled = DataFrame(scaled, columns=to_scale.columns)

    concat = pd.concat([scaled, ignored], axis=1)

    return concat[df.columns], scale
