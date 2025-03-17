from onedata.lakehouse import IcebergManager
import numpy as np
import pandas as pd
import base64
import gzip
from array import array

# pip install --extra-index-url https://gitea.onepredict.net/api/packages/guardione/pypi/simple/ onedata
# pip install pandas

iceberg_manager = IcebergManager(name="nvme")

fault_info = iceberg_manager.to_pandas(namespace="g_motor", table_name="fault_info")
fault_info

from pandas import to_datetime
from datetime import timedelta


def strptime_iso(s, tz="UTC"):
    """
    s: str - input date string
    tz: str - timezone to localize
    """
    dt = to_datetime(s)
    dt_iso = dt.tz_localize(tz).isoformat()
    return dt_iso


def strptime_iso_with_tz(
    date: str,
    from_tz: str = "Asia/Seoul",
    to_tz: str = "UTC",
    return_iso_format=True,
    delta: timedelta = None,
):
    """
    date: str - input date
    from_tz: str - timezone of the input date
    to_tz: str - timezone to convert to
    return_iso_format: bool - return the date in iso format
    delta: timedelta - add a timedelta to the date
    """
    date_converted = to_datetime(strptime_iso(date, tz=from_tz)).tz_convert(to_tz)
    if return_iso_format:
        date_converted = date_converted.isoformat()
    return date_converted


def decode_and_parse(cur_arr):
    gzipped_data = base64.b64decode(cur_arr)
    decompressed_data = gzip.decompress(gzipped_data)
    parsed_data = list(array("f", decompressed_data))
    return np.array(parsed_data, dtype=np.float32)


## 모든 파일 불러오기


def all_load_data(iceberg_manager, MOTOR_ID, start_date, end_date):

    FROM_TIME, TO_TIME = strptime_iso_with_tz(start_date), strptime_iso_with_tz(
        end_date
    )

    if isinstance(MOTOR_ID, str):
        query = f"product_motor_id == '{MOTOR_ID}'"
        print(query)
    else:
        query = f"motor_id == {MOTOR_ID}  and acq_time > '{FROM_TIME}' and acq_time < '{TO_TIME}' and is_valid=='true'"
        print(query)

    downsampled_length = 1000

    if isinstance(MOTOR_ID, str):
        signal_df = iceberg_manager.to_pandas(
            namespace="g_motor",
            # table_name="waveform",
            table_name="deeplearning_framework_motor_waveforms",
            row_filter=query,
            limit=3000,  # limit the number of rows to return
        )
        signal_df = signal_df.assign(
            raw_signal=signal_df.apply(
                lambda row: {
                    "current_u": decode_and_parse(row["current_u"]),
                    "current_v": decode_and_parse(row["current_v"]),
                    "current_w": decode_and_parse(row["current_w"]),
                },
                axis=1,
            )
        ).sort_values(by="acq_time")

    else:

        signal_df = iceberg_manager.to_pandas(
            namespace="g_motor",
            table_name="waveform",
            # table_name = "deeplearning_framework_motor_waveforms",
            row_filter=query,
            limit=3000,  # limit the number of rows to return
        )

        signal_df = signal_df.assign(
            raw_signal=signal_df.raw_signal.transform(
                lambda x: dict(x)  # tuple list -> dict
            ).transform(
                lambda d: {
                    cur_name: np.frombuffer(
                        cur_arr, dtype=np.float32
                    )  # binary -> np.ndarray
                    for cur_name, cur_arr in d.items()
                }
            )
        ).sort_values(
            by="acq_time"
        )  # sort by acq_time

    numpy_data = []
    extracted_signal_df = signal_df.iloc[-15 * 24 * 6 :]
    for i in range(len(extracted_signal_df)):

        processed_data = np.stack(
            [
                extracted_signal_df.iloc[i]["raw_signal"][phase]
                for phase in ["current_u", "current_v", "current_w"]
            ],
            axis=0,
        )
        numpy_data.append(processed_data)

    numpy_data = np.array(numpy_data)

    np.save(f"./all_data/{MOTOR_ID}_{end_date}.npy", numpy_data)
    print("Saved file shape: ", numpy_data.shape)


all_dict = {
    "id": [
        84,
        #    131, 136, 87, 12, 128, 135, 6,
        "a003071a-a4c1-449a-a1fb-c7056dba4742",
        #    'ddb03a78-354f-447c-b79d-b7ea49e2012a',
        #    '744bca46-359e-4342-89be-2ebee1fc94ab',
        #    '2d98c879-880d-4af4-a549-65b778fb60fe0',
        #    '358acf8d-2b93-43ea-91b8-7291ab67be2a',
        #    'e399cd4c-fafa-44a0-9451-e20e665ae346',
        #    'd4d27015-375c-4a81-97d7-18806ba517b7',
        #    'ad36b0f0-318e-49ce-850a-164831968ae6',
        #    'd8b50d80-6a00-4279-870d-01fd7023fef1',
        #    '1f26443c-5250-4a73-b4e2-49238f57a25f',
        #    'e9200b4a-34d6-4742-b95e-def8c1f61fff',
        #    'aef36dec-2560-44e0-bef4-7296329970e2',
        #    'b2aae7c2-8791-448e-b300-bd0c161fb277',
        #    '7be1c6af-ab3e-424a-b663-af31b57a23f4',
        #    '5764667d-a08c-40d0-aea1-59fea1a87dc1'
    ],
    "start_date": [
        "2023-01",
        #    "2023-11", "2023-12", "2023-02", "2022-08", "2021-09", "2023-12", "2021-05",
        "2023-01",
        # "2023-11", "2023-12", "2023-02", "2022-08", "2021-09", "2023-12",
        #    "2021-05", "2023-01", "2023-11", "2023-12", "2023-02", "2022-08", "2021-09",
        #    "2023-12"
    ],
    "end_date": [
        "2023-04",
        #  "2024-02", "2024-03", "2023-05", "2022-11", "2021-12", "2024-03", "2023-08",
        "2023-01",
        # "2023-11", "2023-12", "2023-02", "2022-08", "2021-09", "2023-12",
        #  "2021-05", "2023-01", "2023-11", "2023-12", "2023-02", "2022-08", "2021-09",
        #  "2023-12"
    ],
}


dict_data = all_dict
for ix in range(len(dict_data["id"])):
    MOTOR_ID = dict_data["id"][ix]
    start_date = dict_data["start_date"][ix]
    end_date = dict_data["end_date"][ix]
    try:
        all_load_data(iceberg_manager, MOTOR_ID, start_date, end_date)
    except Exception as e:
        print(f"오류 발생. {e}")
