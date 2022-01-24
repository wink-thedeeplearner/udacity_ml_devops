"""
The unit test will run on churn_predictor.py via Pytest.
Main script will automatically launched Pylint.
Artifact will be saved in figures, models and logs folders.
"""

import os
import logging
import sys
import glob

import pytest
import joblib

import churn_predictor

#Import customized features.
from features import plot_features, cat_features

os.environ['QT_QPA_PLATFORM']='offscreen'
logging.basicConfig(
    filename="logs/churn_predictor.log",
    level=logging.INFO,
    filemode='w',
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
    force=True)


@pytest.fixture(name='dataframe_raw')
def dataframe_raw_():
    """
    raw data import  fixture - returns the raw dataframe from initial csv file
    """
    try:
        raw_dataframe = churn_predictor.import_data(
            "data/bank_data.csv")
        logging.info("Fixture creation for importing raw data: PASSED")
    except FileNotFoundError as err:
        logging.error("Fixture creation for importing raw data: missing data file")
        raise err
    return raw_dataframe


@pytest.fixture(name='dataframe_encoded')
def dataframe_encoded_(dataframe_raw):
    """
    encoded dataframe fixture - returns the encoded dataframe on some specific column
    """
    try:
        dataframe_encoded = churn_predictor.encoder_helper(
            dataframe_raw, cat_features)
        logging.info("Fixture creation for dataframe_encoded: PASSED")
    except KeyError as err:
        logging.error(
            "Fixture creation for dataframe_encoded: missing columns to encode")
        raise err
    return dataframe_encoded


@pytest.fixture(name='train_test_split')
def train_test_split_(dataframe_encoded):
    """
    train_test_split fixture - returns 4 series containing x and y
    """
    try:
        x_train, x_test, y_train, y_test = churn_predictor.feature_engineering(
            dataframe_encoded)

        logging.info("train_test_split fixture creation: PASSED")
    except BaseException:
        logging.error(
            "train_test_split fixture creation: mismatched length for x & y")
        raise
    return x_train, x_test, y_train, y_test


def test_import_data(dataframe_raw):
    """
    test import function - test initial dataset import for raw data
    """
    try:
        assert dataframe_raw.shape[0] > 0
        assert dataframe_raw.shape[1] > 0
        logging.info("Test import_data method: PASSED")
    except AssertionError as err:
        logging.error(
            "Test import_data method: The file doesn't appear to have rows and columns")
        raise err


def test_visualize_eda(dataframe_raw):
    """
    Test visualize_eda method test
    """
    churn_predictor.visualize_eda(dataframe_raw)
    for figure_name in plot_features:
        try:
            with open("plot_figures/eda/%s.jpg" % figure_name, 'r'):
                logging.info("Test visualize_eda method: PASSED")
        except FileNotFoundError as err:
            logging.error("Test visualize_eda method: missing images")
            raise err


def test_encoder_helper(dataframe_encoded):
    """
    Test encoder helper function on dataset encoding
    """
    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Test encoder_helper method: missing rows and columns in the dataframe")
        raise err
    try:
        for column in cat_features:
            assert column in dataframe_encoded
    except AssertionError as err:
        logging.error(
            "Test encoder_helper method: missing encoded columns in the dataframe")
        raise err
    logging.info("Test encoder_helper method: PASSED")
    return dataframe_encoded


def test_feature_engineering(train_test_split):
    """
    Test feature engineering method on train test split
    """
    try:
        x_train = train_test_split[0]
        x_test = train_test_split[1]
        y_train = train_test_split[2]
        y_test = train_test_split[3]
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Test feature_engineering method: PASSED")
    except AssertionError as err:
        logging.error("Test feature_engineering method: mismatch in total observations for x and y")
        raise err
    return train_test_split


def test_train_models(train_test_split):
    """
    Test train_models method
    """
    churn_predictor.train_models(
        train_test_split[0],
        train_test_split[1],
        train_test_split[2],
        train_test_split[3])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("train_models method test: PASSED")
    except FileNotFoundError as err:
        logging.error("Test train_models method: missing files")
        raise err
    for image_name in [
        "Logistic_Regression",
        "Random_Forest",
        "Feature_Importance"]:
        try:
            with open("plot_figures/results/%s.jpg" % image_name, 'r'):
                logging.info("Test train_models method(report generation): PASSED")
        except FileNotFoundError as err:
            logging.error("Test train_models method(report generation): missing images")
            raise err


if __name__ == "__main__":
    for directory in ["logs", "plot_figures/eda", "plot_figures/results", "./models"]:
        files = glob.glob("%s/*" % directory)
        for file in files:
            os.remove(file)
    sys.exit(pytest.main(["-s"]))
