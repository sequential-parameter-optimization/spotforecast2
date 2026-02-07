# Outlier Detection and Visualization

This module provides comprehensive tools for detecting and visualizing outliers in time series data using the Isolation Forest algorithm.

## Overview

The outlier detection module includes three main functions:

- **`get_outliers()`** - Detect outliers using Isolation Forest
- **`visualize_outliers_hist()`** - Visualize outliers with static histograms
- **`visualize_outliers_plotly_scatter()`** - Visualize outliers with interactive Plotly scatter plots

These functions work together to provide a complete workflow for outlier analysis in time series data.

## Installation

The outlier visualization functions require `matplotlib` for histograms and `plotly` for interactive scatter plots.

Using pip:
```bash
pip install matplotlib plotly
```

Using uv:
```bash
uv pip install matplotlib plotly
```

## Quick Start

### Basic Outlier Detection

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.preprocessing.outlier import get_outliers

# Create sample data with outliers
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.concatenate([
        np.random.normal(20, 5, 100),
        [50, 60, 70]  # outliers
    ]),
    'humidity': np.concatenate([
        np.random.normal(60, 10, 100),
        [95, 98, 99]  # outliers
    ])
})

# Detect outliers
outliers = get_outliers(data, contamination=0.03)

for col, outlier_vals in outliers.items():
    print(f"{col}: {len(outlier_vals)} outliers detected")
```

### Histogram Visualization

```python
from spotforecast2.preprocessing.outlier_plots import visualize_outliers_hist

# Create sample data
np.random.seed(42)
data_original = pd.DataFrame({
    'temperature': np.concatenate([
        np.random.normal(20, 5, 100),
        [50, 60, 70]  # outliers
    ])
})

data_cleaned = data_original.copy()

# Visualize outliers
visualize_outliers_hist(
    data_cleaned,
    data_original,
    contamination=0.03,
    figsize=(12, 5),
    alpha=0.7
)
```

### Interactive Plotly Visualization

```python
from spotforecast2.preprocessing.outlier_plots import visualize_outliers_plotly_scatter

# Create time series data
dates = pd.date_range('2024-01-01', periods=103, freq='h')
data_original = pd.DataFrame({
    'temperature': np.concatenate([
        np.random.normal(20, 5, 100),
        [50, 60, 70]  # outliers
    ])
}, index=dates)

data_cleaned = data_original.copy()

# Visualize outliers with interactive plot
visualize_outliers_plotly_scatter(
    data_cleaned,
    data_original,
    contamination=0.03,
    template='plotly_white'
)
```

## API Reference

### get_outliers()

Detect outliers in each column using Isolation Forest.

**Signature:**
```python
def get_outliers(
    data: pd.DataFrame,
    data_original: Optional[pd.DataFrame] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
) -> Dict[str, pd.Series]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | The input DataFrame to check for outliers |
| `data_original` | DataFrame | None | Optional original DataFrame before outlier detection. If provided, helps identify which values became NaN due to outlier detection |
| `contamination` | float | 0.01 | The estimated proportion of outliers in the dataset (between 0 and 1) |
| `random_state` | int | 1234 | Random seed for reproducibility |

**Returns:**

A dictionary mapping column names to pandas Series of outlier values.

**Raises:**

- `ValueError` - If data is empty or contains no columns

**Example:**

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.preprocessing.outlier import get_outliers

# Create sample data
np.random.seed(42)
data = pd.DataFrame({
    'A': np.concatenate([np.random.normal(0, 1, 100), [10, 11, 12]]),
    'B': np.concatenate([np.random.normal(5, 2, 100), [100, 110, 120]])
})

# Detect outliers
outliers = get_outliers(data, contamination=0.03)
for col, outlier_vals in outliers.items():
    print(f"{col}: {len(outlier_vals)} outliers detected")
```

### visualize_outliers_hist()

Visualize outliers using stacked histograms.

**Signature:**
```python
def visualize_outliers_hist(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
    figsize: tuple[int, int] = (10, 5),
    bins: int = 50,
    **kwargs: Any,
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | The DataFrame with cleaned data (outliers may be NaN) |
| `data_original` | DataFrame | Required | The original DataFrame before outlier detection |
| `columns` | list[str] | None | List of column names to visualize. If None, all columns are used |
| `contamination` | float | 0.01 | The estimated proportion of outliers in the dataset |
| `random_state` | int | 1234 | Random seed for reproducibility |
| `figsize` | tuple[int, int] | (10, 5) | Figure size as (width, height) |
| `bins` | int | 50 | Number of histogram bins |
| `**kwargs` | Any | - | Additional keyword arguments passed to plt.hist() (e.g., color, alpha, edgecolor) |

**Returns:**

None. Displays matplotlib figures.

**Raises:**

- `ValueError` - If data or data_original is empty, or if specified columns don't exist

**Example:**

```python
import pandas as pd
import numpy as np
from spotforecast2.preprocessing.outlier_plots import visualize_outliers_hist

# Create sample data
np.random.seed(42)
data_original = pd.DataFrame({
    'temperature': np.concatenate([
        np.random.normal(20, 5, 100),
        [50, 60, 70]  # outliers
    ]),
    'humidity': np.concatenate([
        np.random.normal(60, 10, 100),
        [95, 98, 99]  # outliers
    ])
})

data_cleaned = data_original.copy()

# Visualize outliers
visualize_outliers_hist(
    data_cleaned,
    data_original,
    contamination=0.03,
    figsize=(12, 5),
    alpha=0.7,
    edgecolor='black'
)
```

### visualize_outliers_plotly_scatter()

Visualize outliers using interactive Plotly scatter plots.

**Signature:**
```python
def visualize_outliers_plotly_scatter(
    data: pd.DataFrame,
    data_original: pd.DataFrame,
    columns: Optional[list[str]] = None,
    contamination: float = 0.01,
    random_state: int = 1234,
    **kwargs: Any,
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | The DataFrame with cleaned data (outliers may be NaN) |
| `data_original` | DataFrame | Required | The original DataFrame before outlier detection |
| `columns` | list[str] | None | List of column names to visualize. If None, all columns are used |
| `contamination` | float | 0.01 | The estimated proportion of outliers in the dataset |
| `random_state` | int | 1234 | Random seed for reproducibility |
| `**kwargs` | Any | - | Additional keyword arguments passed to go.Figure.update_layout() (e.g., template, height) |

**Returns:**

None. Displays Plotly figures.

**Raises:**

- `ValueError` - If data or data_original is empty, or if specified columns don't exist
- `ImportError` - If plotly is not installed

**Example:**

```python
import pandas as pd
import numpy as np
from spotforecast2.preprocessing.outlier_plots import visualize_outliers_plotly_scatter

# Create time series data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=103, freq='h')
data_original = pd.DataFrame({
    'temperature': np.concatenate([
        np.random.normal(20, 5, 100),
        [50, 60, 70]  # outliers
    ]),
    'humidity': np.concatenate([
        np.random.normal(60, 10, 100),
        [95, 98, 99]  # outliers
    ])
}, index=dates)

data_cleaned = data_original.copy()

# Visualize outliers
visualize_outliers_plotly_scatter(
    data_cleaned,
    data_original,
    contamination=0.03,
    template='plotly_white',
    height=600
)
```

## Complete Workflow Example

Here's a complete example showing the typical workflow for outlier detection and visualization:

```python
import pandas as pd
import numpy as np
from spotforecast2_safe.preprocessing.outlier import get_outliers
from spotforecast2.preprocessing.outlier_plots import (
    visualize_outliers_hist,
    visualize_outliers_plotly_scatter)

# Create realistic time series data with outliers
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=200, freq='h')
data_original = pd.DataFrame({
    'temperature': np.concatenate([
        np.random.normal(20, 5, 197),
        [50, 60, 70]  # outliers
    ]),
    'humidity': np.concatenate([
        np.random.normal(60, 10, 197),
        [95, 98, 99]  # outliers
    ]),
    'pressure': np.concatenate([
        np.random.normal(1013, 10, 197),
        [800, 1200, 950]  # outliers
    ])
}, index=dates)

# Make a copy for cleaning
data_cleaned = data_original.copy()

# Step 1: Detect outliers
print("=== Outlier Detection ===")
outliers = get_outliers(
    data_original,
    contamination=0.015
)

for col, outlier_vals in outliers.items():
    pct = (len(outlier_vals) / len(data_original)) * 100
    print(f"{col}: {len(outlier_vals)} outliers ({pct:.2f}%)")

# Step 2: Visualize with histograms
print("\n=== Histogram Visualization ===")
visualize_outliers_hist(
    data_cleaned,
    data_original,
    contamination=0.015,
    figsize=(14, 4),
    alpha=0.7
)

# Step 3: Visualize with Plotly (interactive)
print("\n=== Interactive Plotly Visualization ===")
visualize_outliers_plotly_scatter(
    data_cleaned,
    data_original,
    contamination=0.015,
    template='plotly_white'
)

# Step 4: Selective column visualization
print("\n=== Selective Column Analysis ===")
visualize_outliers_hist(
    data_cleaned,
    data_original,
    columns=['temperature', 'humidity'],
    contamination=0.015
)
```

## Parameters and Configuration

### contamination parameter

The `contamination` parameter controls the expected proportion of outliers in the dataset:

- **0.01** (1%) - Conservative, detects severe outliers only
- **0.02** (2%) - Moderate, typical for most applications
- **0.05** (5%) - Liberal, detects more potential anomalies

Choose based on your domain knowledge and data characteristics.

### random_state parameter

The `random_state` parameter ensures reproducibility:

```python
# Same random_state produces consistent results
outliers1 = get_outliers(data, random_state=42)
outliers2 = get_outliers(data, random_state=42)
# outliers1 == outliers2
```

### Matplotlib histogram options

When using `visualize_outliers_hist()`, you can pass additional matplotlib histogram options:

```python
visualize_outliers_hist(
    data_cleaned,
    data_original,
    bins=100,           # More granular bins
    alpha=0.5,          # Transparency
    edgecolor='black',  # Border around bars
    linewidth=0.5       # Border thickness
)
```

### Plotly layout options

When using `visualize_outliers_plotly_scatter()`, you can customize the Plotly figure:

```python
visualize_outliers_plotly_scatter(
    data_cleaned,
    data_original,
    template='plotly_dark',    # Dark theme
    height=700,                 # Figure height
    width=1200                  # Figure width
)
```

## Algorithm Details

### Isolation Forest

The underlying algorithm uses scikit-learn's `IsolationForest`, which:

1. Randomly selects features and split values
2. Isolates anomalies by exploiting their rarity
3. Assigns anomaly scores based on path lengths
4. Marks points with scores exceeding the contamination threshold as outliers

**Key characteristics:**

- No distance computation needed (efficient for high dimensions)
- Scales well with number of features
- Robust to varying scales
- No hyperparameter tuning required beyond contamination

## Best Practices

### 1. Preprocessing

Clean your data before outlier detection:

```python
# Remove missing values
data_clean = data.dropna()

# Standardize if needed
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    index=data.index,
    columns=data.columns
)
```

### 2. Contamination Estimation

Estimate contamination based on domain knowledge:

```python
# For known outlier percentage
contamination = n_outliers / len(data)

# For exploratory analysis, try multiple values
for cont in [0.01, 0.02, 0.05]:
    outliers = get_outliers(data, contamination=cont)
    print(f"Contamination {cont}: {len(outliers)} outliers")
```

### 3. Visual Inspection

Always visualize results:

```python
# Histogram for distribution analysis
visualize_outliers_hist(data_cleaned, data_original)

# Time series plot for temporal patterns
visualize_outliers_plotly_scatter(data_cleaned, data_original)
```

### 4. Validation

Verify outliers make sense in context:

```python
outliers = get_outliers(data, contamination=0.02)

for col, vals in outliers.items():
    print(f"\n{col}:")
    print(f"  Regular range: {data[col].min():.2f} - {data[col].max():.2f}")
    print(f"  Outlier values: {sorted(vals.unique())}")
    print(f"  Outlier indices: {list(vals.index)}")
```

## Testing

All examples in this guide are validated by `tests/test_docs_outliers_examples.py` with 43 comprehensive pytest cases covering:

- Basic outlier detection functionality
- Contamination parameter variations (0.01, 0.02, 0.05)
- Random state reproducibility
- Data integrity and value validation
- Complete workflow integration
- Edge cases (small/large datasets, extreme values, NaN handling)
- Timeseries data with DatetimeIndex
- API examples and return types
- Safety-critical behavior validation

Run the tests:

```bash
# Run outliers documentation tests
uv run pytest tests/test_docs_outliers_examples.py -v

# Quick check
uv run pytest tests/test_docs_outliers_examples.py --tb=no -q
```

## Troubleshooting

### Issue: No outliers detected

**Solution:** Increase the `contamination` parameter:

```python
# Try higher contamination
outliers = get_outliers(data, contamination=0.05)
```

### Issue: Too many false positives

**Solution:** Decrease the `contamination` parameter:

```python
# Be more conservative
outliers = get_outliers(data, contamination=0.01)
```

### Issue: ImportError for plotly

**Solution:** Install plotly:

```bash
pip install plotly
```

Or use histogram visualization instead:

```python
visualize_outliers_hist(data_cleaned, data_original)
```

## See Also

- [Isolation Forest Algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

