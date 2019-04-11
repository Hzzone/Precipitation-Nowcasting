HKO-7 Dataset Files
-------------------
The HKO-7 dataset used in the benchmark contains radar echo data from 2009 to 2015 collected by
HKO.

| File Name | Description |
| --------- | :----------- |
| hko7_rainy_train_days.txt | Stores the frame names that belong to the training set of HKO-7 data |
| hko7_rainy_valid_days.txt | Stores the frame names that belong to the validation set of HKO-7 data |
| hko7_rainy_test_days.txt  | Stores the frame names that belong to the testing set of HKO-7 data |
| intensity_day.pkl | The daily intensity of the HKO-7 data |

In the pd directory, we will include all the Pandas DataFrames. The data should be downloaded by `python download_all.py`. To load it, refer to the [documentation of pandas](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_pickle.html).

| File Name | Description |
| --------- | :----------- |
| hko7_all.pkl | Datetimes from 09-15 |
| hko7_all_09_14.pkl | Datetimes from 09-14 |
| hko7_all_15.pkl  | Datetimes in year 2015 |
| hko7_rainy_train.pkl | Datetimes of the training set |
| hko7_rainy_valid.pkl | Datetimes of the validation set |
| hko7_rainy_test.pkl | Datetimes of the test set |
