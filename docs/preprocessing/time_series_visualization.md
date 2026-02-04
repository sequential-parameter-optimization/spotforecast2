# Time Series Visualization

This module provides interactive time series visualization using Plotly, with support for multiple datasets and flexible customization options.

## Overview

The time series visualization module includes two main functions:

- **`visualize_ts_plotly()`** - Visualize multiple time series datasets with Plotly
- **`visualize_ts_comparison()`** - Compare datasets with optional statistical overlays

These functions provide a flexible, interactive way to explore time series data with support for train/validation/test splits or any custom dataset groupings.

## Installation

The time series visualization functions require `plotly`:

Using pip:
```bash
pip install plotly
```

Using uv:
```bash
uv pip install plotly
```

## Quick Start

### Basic Visualization

```python
import pandas as pd
import numpy as np
from spotforecast2.preprocessing.time_series_visualization import visualize_ts_plotly

# Create sample datasets
np.random.seed(42)
dates_train = pd.date_range('2024-01-01', periods=100, freq='h')
dates_val = pd.date_range('2024-05-11', periods=50, freq='h')
dates_test = pd.date_range('2024-07-01', periods=30, freq='h')

data_train = pd.DataFrame({
    'temperature': np.random.normal(20, 5, 100),
    'humidity': np.random.normal(60, 10, 100)
}, index=dates_train)

data_val = pd.DataFrame({
    'temperature': np.random.normal(22, 5, 50),
    'humidity': np.random.normal(55, 10, 50)
}, index=dates_val)

data_test = pd.DataFrame({
    'temperature': np.random.normal(25, 5, 30),
    'humidity': np.random.normal(50, 10, 30)
}, index=dates_test)

# Visualize all datasets
dataframes = {
    'Train': data_train,
    'Validation': data_val,
    'Test': data_test
}

visualize_ts_plotly(dataframes)
```

### Single Dataset Visualization

```python
# Visualize a single dataset
dataframes = {'Data': data_train}
visualize_ts_plotly(dataframes, columns=['temperature'])
```

### Custom Styling

```python
# Customize colors and template
visualize_ts_plotly(
    dataframes,
    template='plotly_dark',
    colors={
        'Train': 'blue',
        'Validation': 'green',
        'Test': 'red'
    },
    figsize=(1400, 600)
)
```

## API Reference

### visualize_ts_plotly()

Visualize multiple time series datasets interactively with Plotly.

**Signature:**
```python
def visualize_ts_plotly(
    dataframes: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    title_suffix: str = "",
    figsize: tuple[int, int] = (1000, 500),
    template: str = "plotly_white",
    colors: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataframes` | Dict[str, DataFrame] | Required | Dictionary mapping dataset names to DataFrames with datetime index |
| `columns` | list[str] | None | Columns to visualize. If None, all columns are used |
| `title_suffix` | str | "" | Suffix to append to column names in titles (e.g., "[°C]") |
| `figsize` | tuple[int, int] | (1000, 500) | Figure size as (width, height) in pixels |
| `template` | str | "plotly_white" | Plotly template name ("plotly_white", "plotly_dark", "ggplot2", etc.) |
| `colors` | Dict[str, str] | None | Dictionary mapping dataset names to colors. If None, uses default colors |
| `**kwargs` | Any | - | Additional arguments passed to go.Scatter() (e.g., fill='tozeroy') |

**Returns:**

None. Displays Plotly figures.

**Raises:**

- `ValueError` - If dataframes dict is empty, contains empty DataFrames, or if specified columns don't exist
- `ImportError` - If plotly is not installed
- `TypeError` - If dataframes parameter is not a dictionary

**Example:**

```python
import pandas as pd
import numpy as np
from spotforecast2.preprocessing.time_series_visualization import visualize_ts_plotly

# Create sample data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='h')
df = pd.DataFrame({
    'temperature': np.random.normal(20, 5, 100),
    'humidity': np.random.normal(60, 10, 100)
}, index=dates)

# Visualize single dataset
visualize_ts_plotly({'Data': df})
```

### visualize_ts_comparison()

Compare multiple datasets with optional statistical overlays.

**Signature:**
```python
def visualize_ts_comparison(
    dataframes: Dict[str, pd.DataFrame],
    columns: Optional[List[str]] = None,
    title_suffix: str = "",
    figsize: tuple[int, int] = (1000, 500),
    template: str = "plotly_white",
    colors: Optional[Dict[str, str]] = None,
    show_mean: bool = False,
    **kwargs: Any,
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataframes` | Dict[str, DataFrame] | Required | Dictionary mapping dataset names to DataFrames |
| `columns` | list[str] | None | Columns to visualize. If None, all columns are used |
| `title_suffix` | str | "" | Suffix to append to titles |
| `figsize` | tuple[int, int] | (1000, 500) | Figure size as (width, height) in pixels |
| `template` | str | "plotly_white" | Plotly template |
| `colors` | Dict[str, str] | None | Dictionary mapping dataset names to colors |
| `show_mean` | bool | False | If True, overlay the mean of all datasets |
| `**kwargs` | Any | - | Additional arguments for go.Scatter() |

**Returns:**

None. Displays Plotly figures.

**Raises:**

- `ValueError` - If dataframes dict is empty
- `ImportError` - If plotly is not installed

**Example:**

```python
import pandas as pd
import numpy as np
from spotforecast2.preprocessing.time_series_visualization import visualize_ts_comparison

# Create sample data
np.random.seed(42)
dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
dates2 = pd.date_range('2024-05-11', periods=100, freq='h')

df1 = pd.DataFrame({
    'value': np.random.normal(20, 5, 100)
}, index=dates1)

df2 = pd.DataFrame({
    'value': np.random.normal(22, 5, 100)
}, index=dates2)

# Compare with mean overlay
visualize_ts_comparison(
    {'Dataset1': df1, 'Dataset2': df2},
    show_mean=True
)
```

## Complete Workflow Examples

### Train/Validation/Test Split Visualization

```python
import pandas as pd
import numpy as np
from spotforecast2.preprocessing.time_series_visualization import visualize_ts_plotly

# Create time series data
np.random.seed(42)
full_data = pd.DataFrame({
    'temperature': np.sin(np.linspace(0, 10, 300)) + np.random.normal(0, 0.1, 300),
    'humidity': np.cos(np.linspace(0, 10, 300)) * 100 + np.random.normal(50, 5, 300)
}, index=pd.date_range('2024-01-01', periods=300, freq='h'))

# Split data
split1 = int(0.6 * len(full_data))
split2 = int(0.8 * len(full_data))

data_train = full_data.iloc[:split1]
data_val = full_data.iloc[split1:split2]
data_test = full_data.iloc[split2:]

# Visualize
dataframes = {
    'Train': data_train,
    'Validation': data_val,
    'Test': data_test
}

visualize_ts_plotly(
    dataframes,
    template='plotly_white',
    figsize=(1200, 600)
)
```

### Multiple Datasets Comparison

```python
from spotforecast2.preprocessing.time_series_visualization import visualize_ts_comparison

# Create datasets from different time periods
dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
dates2 = pd.date_range('2024-04-01', periods=100, freq='h')
dates3 = pd.date_range('2024-07-01', periods=100, freq='h')

df1 = pd.DataFrame({
    'temperature': np.random.normal(15, 3, 100)
}, index=dates1)

df2 = pd.DataFrame({
    'temperature': np.random.normal(22, 3, 100)
}, index=dates2)

df3 = pd.DataFrame({
    'temperature': np.random.normal(25, 3, 100)
}, index=dates3)

# Compare with mean
visualize_ts_comparison(
    {
        'Winter': df1,
        'Spring': df2,
        'Summer': df3
    },
    show_mean=True,
    colors={
        'Winter': 'blue',
        'Spring': 'green',
        'Summer': 'red'
    }
)
```

### Dynamic Dataset Handling

```python
# Function works with any number of datasets
dataframes = {}

for i in range(5):
    dates = pd.date_range(f'2024-{i+1:02d}-01', periods=50, freq='h')
    dataframes[f'Month_{i+1}'] = pd.DataFrame({
        'sales': np.random.gamma(2, 2, 50) * 1000
    }, index=dates)

visualize_ts_plotly(
    dataframes,
    title_suffix='[USD]',
    figsize=(1400, 600)
)
```

## Parameters and Configuration

### figsize Parameter

Figure size as (width, height) in pixels:

```python
# Small figure
visualize_ts_plotly(dataframes, figsize=(800, 400))

# Large figure for detailed inspection
visualize_ts_plotly(dataframes, figsize=(1600, 800))
```

### Template Options

Plotly provides several built-in templates:

```python
# Light theme (default)
visualize_ts_plotly(dataframes, template='plotly_white')

# Dark theme
visualize_ts_plotly(dataframes, template='plotly_dark')

# Minimal theme
visualize_ts_plotly(dataframes, template='plotly')

# Other themes
visualize_ts_plotly(dataframes, template='ggplot2')
visualize_ts_plotly(dataframes, template='seaborn')
```

### Color Customization

Define custom colors for each dataset:

```python
colors = {
    'Train': '#1f77b4',      # Blue
    'Validation': '#ff7f0e', # Orange
    'Test': '#2ca02c'        # Green
}

visualize_ts_plotly(dataframes, colors=colors)
```

### Advanced Scatter Customization

Pass additional options to Plotly Scatter:

```python
visualize_ts_plotly(
    dataframes,
    fill='tozeroy',           # Fill area under line
    line=dict(width=2),       # Line width
    opacity=0.8               # Transparency
)
```

## Best Practices

### 1. Use Datetime Index

Always use pandas datetime index for proper time axis handling:

```python
# Good
df = pd.DataFrame(data, index=pd.date_range('2024-01-01', periods=len(data), freq='h'))

# Avoid
df = pd.DataFrame(data)  # Will use default integer index
```

### 2. Consistent Data Shapes

Ensure all DataFrames have consistent columns for comparison:

```python
# Verify columns match
columns = set(df1.columns) & set(df2.columns) & set(df3.columns)
if not columns:
    raise ValueError("DataFrames have no common columns")
```

### 3. Handle Large Datasets

For large time series, consider subsampling:

```python
# Subsample every 10th point
df_sub = df[::10]
visualize_ts_plotly({'Data': df_sub})
```

### 4. Meaningful Dataset Names

Use descriptive names for datasets:

```python
# Good
dataframes = {
    'Training (2023)': data_train,
    'Validation (Jan 2024)': data_val,
    'Testing (Feb 2024)': data_test
}

# Avoid
dataframes = {
    'A': data_train,
    'B': data_val,
    'C': data_test
}
```

## Troubleshooting

### Issue: Overlapping Datasets

If datasets overlap in time, use separate figures:

```python
# Visualize one column at a time
for col in dataframes[list(dataframes.keys())[0]].columns:
    visualize_ts_plotly(dataframes, columns=[col])
```

### Issue: Memory Issues with Large Datasets

Downsample before visualization:

```python
# Downsample to hourly
df_downsampled = df.resample('1H').mean()
visualize_ts_plotly({'Data': df_downsampled})
```

### Issue: Missing Data in Visualization

Handle missing values before visualization:

```python
# Forward fill missing values
df_filled = df.fillna(method='ffill')
visualize_ts_plotly({'Data': df_filled})
```

## See Also

- [Outlier Detection and Visualization](outliers.md)
- [Plotly Documentation](https://plotly.com/python/)

## References

- Plotly Dash and Plotly.py documentation: https://plotly.com/python/
- Pandas datetime index: https://pandas.pydata.org/docs/user_guide/timeseries.html
