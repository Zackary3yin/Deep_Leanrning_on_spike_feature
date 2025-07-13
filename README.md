## Predicting Coma Recovery

### 🧠 Project Overview (For Lauren)

You're joining a research project titled:

> **"Predicting Coma Recovery Using EEG features"**

The goal of this study is to predict neurological recovery in comatose ICU patients following cardiac arrest using EEG (electroencephalogram) signals. By leveraging over 1,000 patients and nearly 60,000 hours of EEG data, we aim to build predictive models that identify outcome-relevant EEG dynamics early and stably, aiding clinical decisions in real time.

---

### 🧪 What Are EEG Features?

Instead of using raw EEG waveforms directly, we extract 95 high-level EEG-derived features at an **hourly resolution**. These features quantify key physiological properties of brain activity, grouped into several categories:

| Feature Type                          | Description                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Spike Rate (spike Hz)**             | Frequency of epileptiform discharges, reflects seizure activity severity. A core trajectory variable. |
| **BCI (Background Continuity Index)** | Measures background continuity; associated with brain suppression or recovery.                        |
| **Amplitude**                         | Strength of EEG signal. A high proportion of <5 µV typically indicates suppressed brain activity.     |
| **Entropy**                           | Signal complexity measures (e.g., Shannon entropy); less predictive in this context.                  |

Each patient has a 3D data tensor: **\[patient × time × features]**, used as model input.

---

### ⚙️ Modeling Strategies

We built two modeling configurations to simulate different ICU monitoring strategies:

#### ✅ Configuration 1: Time Series Modeling (Sequential Mode)

* Each sample is a temporal sequence from 16h up to time `t` post-ROSC.
* Mimics a real ICU scenario where EEG accumulates over time.
* Input format: `X[:, :L, :]` (L = number of hours)
* Built using attention-based BiLSTM (deep learning only)
* More complex, but clinically realistic.

#### ✅ Configuration 2: Hourly Snapshot Modeling 🌟 *(You will start here)*

* Each model sees only a single time point's features (1-hour snapshot)
* Each hour is treated as an independent tabular classification task
* Simulates the question: *“Can we predict recovery based on this hour’s data alone?”*
* Simpler to implement and well-suited for comparing ML algorithms

---

### 🤖 Models Used in Configuration 2

You will run three types of models to compare predictive performance:

| Model Name              | Type              | Strengths                                                    |
| ----------------------- | ----------------- | ------------------------------------------------------------ |
| **Random Forest (RF)**  | Traditional ML    | Robust to small samples, easy to interpret, quick to train   |
| **XGBoost**             | Gradient Boosting | High accuracy, strong in capturing feature interactions      |
| **LSTM** | Deep Learning RNN | Captures non-linear feature trends and temporal dependencies |

All models are trained and evaluated on the same features and labels for fair comparison.

---

### 🎯 Tasks in Configuration 2

You’ll be working on three supervised classification tasks:

| Task ID | Task Description                                           | Label Type                | Classification |
| ------- | ---------------------------------------------------------- | ------------------------- | -------------- |
| 2.1     | CPC outcome prediction (with or without clinical features) | Good vs Poor outcome      | Binary         |
| 2.2     | EEG cluster prediction (3 clusters)                        | Cluster labels (from DTW) | Multi-class    |
| 2.3     | EEG cluster binary classification                          | Cluster 0 vs Clusters 1/2 | Binary         |

* All tasks use **5-fold stratified cross-validation**
* Metrics: **AUC** and **Accuracy**

---

### 📁 File to Run

Start by running:

```bash
Deep Learning pipline_hourly.ipynb
```

This notebook covers:

* Data loading and cleaning
* Feature selection
* Model training and evaluation (LSTM, RF, XGBoost)
* Visualizations (optional)

---

### 🔬 Next Steps

Once you're comfortable with this configuration and understand the LSTM and tree-based models, we can:

* Move to **Configuration 1** (sequential modeling):
  `Deep Learning pipline_time series.ipynb`
* Explore **model interpretation** with attention weights
* Test new types of EEG features (e.g., connectivity, information flow)
* Improve robustness with cross-site generalization or domain adaptation
