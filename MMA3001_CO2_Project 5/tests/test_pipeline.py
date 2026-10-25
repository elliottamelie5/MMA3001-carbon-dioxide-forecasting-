"""Unit tests for forecast design, metrics and leakage prevention."""
import numpy as np
import pandas as pd
import pytest
from src.pipeline import make_supervised,chronological_split,score,prepare_series

def test_forecast_alignment():
    s=pd.Series(np.arange(200,dtype=float),index=pd.date_range('2024-01-01',periods=200,freq='15min',tz='UTC'))
    frame=make_supervised(s)
    assert frame.iloc[0]['lag_0']==4
    assert frame.iloc[0]['lag_4']==0
    assert frame.iloc[0]['target']==8

def test_missing_values_not_interpolated():
    s=pd.Series(np.arange(200,dtype=float),index=pd.date_range('2024-01-01',periods=200,freq='15min',tz='UTC'))
    s.iloc[10]=np.nan
    frame=make_supervised(s)
    assert not frame.isna().any().any()
    assert s.index[10] not in frame.index

def test_chronological_partitions():
    frame=pd.DataFrame({'x':np.arange(250)})
    train,val,test=chronological_split(frame)
    assert train.index.max()<val.index.min()<test.index.min()
    assert len(train)+len(val)+len(test)<len(frame)

def test_score_known_values():
    result=score([0,2],[1,1])
    assert result['MAE_ppm']==pytest.approx(1)
    assert result['RMSE_ppm']==pytest.approx(1)

def test_reject_negative_horizon():
    with pytest.raises(ValueError):make_supervised(pd.Series(np.arange(200)),horizon=-1)

def test_sensor_selection():
    df=pd.DataFrame({'sensor':['a']*200,'timestamp':pd.date_range('2024-01-01',periods=200,freq='15min',tz='UTC'),'co2_ppm':[500.]*200})
    sensor,series=prepare_series(df)
    assert sensor=='a' and len(series)==200
