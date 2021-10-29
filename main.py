import streamlit as st
import pandas as pd
import webbrowser as web

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
data_set = st.container()

data_sam_container = st.container()
data_desc_container = st.container()

show_features_target = st.container()

show_features = st.container()

show_model = st.container()

show_data_desc = False
show_data_sam = False

available_features = [('MSSubClass', 'The building class'), ('LotArea', 'Lot size in square feet'),
                      ('OverallQual', 'Overall material and finish quality'),
                      ('OverallCond', 'Overall condition rating'), ('YearBuilt', 'Original construction date'),
                      ('YearRemodAdd', 'Remodel date'), ('1stFlrSF', 'First floor square feet'),
                      ('2ndFlrSF', 'Second floor square feet'),
                      ('LowQualFinSF', 'Low quality finished square feet (all floors)'),
                      ('FullBath', 'Full bathrooms above grade'), ('Fireplaces', 'Number of fireplaces'),
                      ('WoodDeckSF', 'Wood deck area in square feet'),
                      ('OpenPorchSF', 'Open porch area in square feet'),
                      ('EnclosedPorch', 'Enclosed porch area in square feet'),
                      ('3SsnPorch', 'Three season porch area in square feet'), ('PoolArea', 'Pool area in square feet'),
                      ('MiscVal', '$Value of miscellaneous feature')]

features = list()

with header:
    st.title("Welcome to my house price prediction project!")
    st.text("This is the house price prediction for residential homes in Ames, Iowa")


@st.cache
def read_data():
    return pd.read_csv("data/train.csv")


def select_feature(feature_name):
    features.append(feature_name)


def deselect_feature(feature_name):
    if feature_name in features:
        features.remove(feature_name)


with data_set:
    st.header("House price dataset")
    st.text("I download this dataset from Kaggle.com")

    house_price_data = read_data()
    target_data = house_price_data['SalePrice']

    left_column, right_column = st.columns(2)
    with left_column:
        if st.button("Download house price data description"):
            web.open_new_tab(
                "https://storage.googleapis.com/kagglesdsdata/competitions/10211/111096/data_description.txt"
                "?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1635604855&Signature"
                "=Faww5C9yQ8reIT38SDGcv%2FZHG21SQc6jxFgJNj0pf6ZQOkln4hQjM%2B6%2FClqudwBpfNnckvltvkbCr8%2FU"
                "%2FUhhCbCsEcN2%2FDAPgzX2vAan0QTVCb0DRICeUiG5lvvD8y%2FS6nLNGZMJS5eEhsWDeav7S%2Btr3eAIu3l"
                "%2BAJP4SbJoJjKPbosHQdptOy7TcmnFVnnzS6vU1SQhokjKwvZ1uVFC5XAEIZdJca0hwJCnAOw%2B6dHXPbYGJ%2Bh"
                "%2FMsdDFVSsogW0gU8lv9jW4y0SYQvywnB7wVdCg731bQtqNVr%2BJkBQQWI6lx9nCpYDracYXcur"
                "%2Fh1Q8nq98vpJeHOM3iVuxeZXcQ%3D%3D&response-content-disposition=attachment%3B+filename"
                "%3Ddata_description.txt")
    with right_column:
        if st.button("Download house price dataset"):
            web.open_new_tab(
                "https://storage.googleapis.com/kagglesdsdata/competitions/10211/111096/train.csv?GoogleAccessId=web"
                "-data@kaggle-161607.iam.gserviceaccount.com&Expires=1635596514&Signature"
                "=BkHjjzx6egRH7CyvTH6DiCz7JX1AFrSxWzPBsWGI32Tni3SVg0C%2FG2p5%2FJ7eouDcd0oZohOL1zaC%2F2XxDpWfr%2B%2B"
                "%2BKjOrorPE2P37zF%2F8r%2Fcg6r"
                "%2BdcTK5raTalDJgFBSwDtmCtVMC8hD9q9dMed7Cja2X3NmAAJCtfsbpDvK76y4AtzAmeMOLxTGVfXscl6iPwmBRG"
                "%2FpqLANj4zZ24lvJUQi%2Fkx9W9pWc6WD3eyLVZU1DNQg3k3rjHb9qVh0QwzvqTIsO"
                "%2Fn71hF4NRQIbjFR3iTsBNduxyLpvBq0R3xYdSnFyftMY0Qg9%2Bvjl41eVInAhW5I8sxBgN4opilmR%2Bjk1Lg%3D%3D"
                "&response-content-disposition=attachment%3B+filename%3Dtrain.csv")

    left_column, right_column = st.columns(2)
    with left_column:
        if st.checkbox("Show data description"):
            show_data_desc = True
    with right_column:
        if st.checkbox("Show data sample"):
            show_data_sam = True

with data_sam_container:
    if show_data_sam:
        st.write(house_price_data.head())
with data_desc_container:
    if show_data_desc:
        text_file = open("data/data_description.txt", "r")
        description = text_file.read()
        text_file.close()
        st.write(description)

with show_features_target:
    show_features, show_target = st.columns(2)

    with show_features:
        st.header("Features")

        st.markdown("* **These are the usable features to predict sale price of the house**")

        checkboxes = {feature: st.checkbox(f"{name}", True) for feature, name in available_features}

        for key, value in checkboxes.items():
            if value:
                select_feature(key)
            else:
                deselect_feature(key)

    with show_target:
        st.header("Target Data")
        st.markdown("* **This is the target data to predict**")
        st.table(target_data.head(20))

with show_model:
    max_depth = show_model.slider("What should be the maximum depth of the model?", min_value=10, max_value=30,
                                  value=18,
                                  step=1)
    n_estimators = show_model.selectbox("How many trees should there be?",
                                        options=[100, 150, 200, 250, 300, 350, 400], index=3)

    st.header("Model Training")
    st.subheader("You can check the performance of the model")

    X = house_price_data[features].copy()

    # Preprocessing data
    original_train_X, original_val_X, train_y, val_y = train_test_split(X, target_data, random_state=1)

    # Define a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=1)

    has_null_value = False

    for col in original_train_X.columns:
        if original_train_X[col].isnull().any():
            has_null_value = True
            break

    if has_null_value:
        # Make copy to avoid changing original data (when imputing)
        train_X_copy = original_train_X.copy()
        val_X_copy = original_val_X.copy()

        # Get names of columns with missing values
        cols_with_missing = [col for col in train_X_copy.columns
                             if train_X_copy[col].isnull().any()]

        # Make new columns indicating what will be imputed
        for col in cols_with_missing:
            train_X_copy[col + '_was_missing'] = train_X_copy[col].isnull()
            val_X_copy[col + '_was_missing'] = val_X_copy[col].isnull()

        # Imputation
        my_imputer = SimpleImputer()
        train_X = pd.DataFrame(my_imputer.fit_transform(train_X_copy))
        val_X = pd.DataFrame(my_imputer.transform(val_X_copy))

        # Imputation removed column names; put them back
        train_X.columns = train_X_copy.columns
        val_X.columns = val_X_copy.columns
    else:
        train_X = original_train_X
        val_X = original_val_X

    model.fit(train_X, train_y)

    prediction = model.predict(val_X)
    model_mae = mean_absolute_error(prediction, val_y)
    model_mse = mean_squared_error(prediction, val_y)
    model_rss = r2_score(prediction, val_y)

    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("* **Mean absolute error of the model is: **")
        st.markdown("* **Mean squared error of the model is: **")
        st.markdown("* **R squared error of the model is: **")
    with right_column:
        st.write(model_mae)
        st.write(model_mse)
        st.write(model_rss)
