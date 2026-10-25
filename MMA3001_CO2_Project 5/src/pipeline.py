"""Reproducible CO2 forecasting with strict chronological evaluation.

The target is the CO2 concentration one hour ahead. Only historic CO2 is used
as a predictor to avoid sensor synchronization and occupancy-mapping assumptions.
"""
import argparse
import json
import time
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_environment(zip_path, max_rows=None):
    """Parse environmental sensor JSON records from a ZIP archive.

    Args:
        zip_path: Path to supplied environmental CSV ZIP.
        max_rows: Optional CSV row limit for a smoke test.
    Returns:
        DataFrame with sensor, timestamp (UTC), and CO2 in ppm.
    Raises:
        ValueError: When no valid CO2 readings exist.
    """
    records=[]
    with zipfile.ZipFile(zip_path) as archive:
        name=next(n for n in archive.namelist() if n.lower().endswith('.csv'))
        with archive.open(name) as file:
            for chunk in pd.read_csv(file, chunksize=10000, nrows=max_rows, dtype={'sensorid':str}):
                for sensor, raw in zip(chunk['sensorid'],chunk['jsondata']):
                    try:
                        variables=json.loads(raw)
                        for variable in variables:
                            if variable.get('variable',{}).get('name','').lower() == 'carbon dioxide':
                                for reading in variable.get('values',[]):
                                    records.append((str(sensor),reading.get('time'),reading.get('value')))
                    except (TypeError,ValueError,KeyError):
                        continue
    if not records:
        raise ValueError('No carbon dioxide readings found in archive')
    df=pd.DataFrame(records,columns=['sensor','unix_time','co2_ppm'])
    df['timestamp']=pd.to_datetime(df['unix_time'],unit='s',utc=True,errors='coerce')
    df['co2_ppm']=pd.to_numeric(df['co2_ppm'],errors='coerce')
    df=df.dropna(subset=['sensor','timestamp','co2_ppm'])
    df=df[df['co2_ppm'].between(250,10000)].copy()  # broad plausibility screen; not calibration
    return df[['sensor','timestamp','co2_ppm']]


def prepare_series(df, sensor=None, interval='15min', min_points=150):
    """Build a regular series without interpolating missing intervals.

    Args:
        df: Parsed CO2 readings.
        sensor: Optional sensor ID; otherwise select the sensor with most readings.
        interval: Pandas resampling interval.
        min_points: Minimum nonmissing observations required.
    Returns:
        Sensor ID and resampled pandas Series with missing slots preserved.
    """
    if sensor is None:
        sensor=str(df.groupby('sensor').size().idxmax())
    selected=df[df['sensor']==str(sensor)].sort_values('timestamp')
    series=selected.groupby('timestamp')['co2_ppm'].median().resample(interval).median()
    if series.notna().sum()<min_points:
        raise ValueError('Insufficient observations for selected sensor')
    return sensor,series


def make_supervised(series, horizon=4, lags=(0,1,2,3,4)):
    """Construct leakage-free lag features and future targets.

    At 15-minute sampling, horizon=4 corresponds to one hour ahead.
    Rows with any missing lag or target are excluded; no future values are imputed.
    """
    if horizon<1 or any(l<0 for l in lags):
        raise ValueError('Horizon and lags must be nonnegative with positive horizon')
    x=pd.DataFrame({f'lag_{lag}':series.shift(lag) for lag in lags})
    x['target']=series.shift(-horizon)
    x=x.dropna()
    if len(x)<100:
        raise ValueError('Too few complete forecast examples')
    return x


def chronological_split(frame):
    """Split sorted examples into 60% train, 20% validation, 20% held-out test.

    Purge four example rows between partitions to reduce overlapping target leakage.
    """
    n=len(frame);a=int(.6*n);b=int(.8*n);gap=8
    if min(a,b-a,n-b)<gap+10:
        raise ValueError('Insufficient samples for purged partitions')
    return frame.iloc[:a-gap],frame.iloc[a:b-gap],frame.iloc[b:]


def score(actual,predicted):
    """Return MAE and RMSE, both measured in ppm."""
    return {'MAE_ppm':float(mean_absolute_error(actual,predicted)),
            'RMSE_ppm':float(np.sqrt(mean_squared_error(actual,predicted)))}


def run(zip_path,output_dir='outputs',sensor=None,max_rows=None):
    """Run model selection on validation data and final evaluation on untouched test data."""
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    raw=load_environment(zip_path,max_rows=max_rows)
    chosen,series=prepare_series(raw,sensor)
    frame=make_supervised(series)
    train,val,test=chronological_split(frame)
    cols=[c for c in frame.columns if c.startswith('lag_')]
    xtrain,ytrain=train[cols],train['target']
    xval,yval=val[cols],val['target']
    xtest,ytest=test[cols],test['target']
    candidates={'persistence':None,'linear_regression':LinearRegression(),
                'tree_depth_3':DecisionTreeRegressor(max_depth=3,min_samples_leaf=20,random_state=42),
                'tree_depth_5':DecisionTreeRegressor(max_depth=5,min_samples_leaf=20,random_state=42),
                'tree_depth_8':DecisionTreeRegressor(max_depth=8,min_samples_leaf=20,random_state=42)}
    results=[]
    for name,model in candidates.items():
        start=time.perf_counter()
        if model is not None: model.fit(xtrain,ytrain)
        fit_seconds=time.perf_counter()-start
        predicted=xval['lag_0'].to_numpy() if model is None else model.predict(xval)
        results.append({'model':name,'fit_seconds':fit_seconds,**score(yval,predicted)})
    validation=pd.DataFrame(results).sort_values('MAE_ppm')
    validation.to_csv(output/'validation_metrics.csv',index=False)
    best=validation.iloc[0]['model']
    model=candidates[best]
    # Test data is used only once, after selection.
    pred=xtest['lag_0'].to_numpy() if model is None else model.predict(xtest)
    test_results={'sensor':chosen,'forecast_horizon_minutes':60,'train_rows':len(train),
                  'validation_rows':len(val),'test_rows':len(test),'chosen_model':best,
                  'heldout_test':score(ytest,pred),'pipeline_seconds':time.perf_counter()-t0,
                  'total_raw_co2_readings':len(raw),'missing_resampled_slots':int(series.isna().sum()),
                  'notes':'Single-sensor, 15-minute bins; complete-case lag examples; chronological purged split.'}
    (output/'results.json').write_text(json.dumps(test_results,indent=2))
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(ytest.index[:300],ytest.iloc[:300],label='Observed',linewidth=1)
    ax.plot(ytest.index[:300],pred[:300],label='Predicted',linewidth=1)
    ax.set(xlabel='UTC time',ylabel='CO2 (ppm)',title=f'One-hour CO2 forecast — sensor {chosen}')
    ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(output/'test_predictions.png',dpi=160);plt.close(fig)
    print(json.dumps(test_results,indent=2))
    return test_results

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Forecast CO2 from Monash environmental data')
    parser.add_argument('--zip',required=True,help='Path to environmental sensor ZIP')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--sensor',default=None)
    parser.add_argument('--max-rows',type=int,default=None)
    args=parser.parse_args()
    run(args.zip,args.output,args.sensor,args.max_rows)
