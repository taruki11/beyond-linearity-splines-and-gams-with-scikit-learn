"""
Beyond Linearity: Splines and GAMs with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_wage
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 2 - split_wage
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 3 - cv_tools
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 4 - polynomial_regression
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 5 - step_functions
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 6 - spline_regression
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 7 - extrapolation (not yet solved)
# TODO: implement

# Step 8 - smoothing_spline
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 9 - local_smoother
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 10 - gam_pipeline
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 11 - partial_effects
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 12 - logistic_gam
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 13 - fit_age_models
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

# Step 14 - test_comparison
import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PolynomialFeatures,
    SplineTransformer,
    StandardScaler,
)


# ============================================================
# Step 1 — load_wage
# 读取数据，并返回数据集的基本信息
# ============================================================

WAGE_URL = (
    "https://raw.githubusercontent.com/"
    "intro-stat-learning/ISLP/main/ISLP/data/Wage.csv"
)


def load_wage():
    path = os.path.join(tempfile.gettempdir(), "Wage.csv")

    if not os.path.exists(path):
        urllib.request.urlretrieve(WAGE_URL, path)

    return pd.read_csv(path)


def describe_wage(df):
    return {
        "n": len(df),
        "columns": df.columns.tolist(),
        "age_range": (
            int(df["age"].min()),
            int(df["age"].max()),
        ),
        "wage_mean": round(float(df["wage"].mean()), 2),
        "n_education_levels": int(df["education"].nunique()),
    }


# ============================================================
# Step 2 — split_wage
# 把数据分成训练集、测试集，并准备年龄预测数据
# ============================================================

def split_wage(df, test_size=0.25, random_state=0):
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def age_xy(df):
    X = df[["age"]]
    y = df["wage"]
    return X, y


def age_grid(lo=18, hi=80, n=63):
    return pd.DataFrame({
        "age": np.linspace(lo, hi, n)
    })


# ============================================================
# Step 3 — cv_tools
# 交叉验证评估模型，并实现 one-standard-error rule
# ============================================================

def cv_mse(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )

    # sklearn 用“越大越好”的 score 约定，
    # 因此 neg_mean_squared_error 是负的 MSE。
    fold_mse = -scores
    mean_mse = np.mean(fold_mse)
    se_mse = np.std(fold_mse, ddof=1) / np.sqrt(len(fold_mse))

    return round(float(mean_mse), 1), round(float(se_mse), 1)


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for value in values:
        model = make_model(value)
        mean_mse, se_mse = cv_mse(model, X, y, cv)
        means.append(mean_mse)
        ses.append(se_mse)

    return means, ses


def one_se_rule(values, means, ses, prefer="smaller"):
    best_index = int(np.argmin(means))
    threshold = means[best_index] + ses[best_index]

    eligible = [
        value
        for value, mean in zip(values, means)
        if mean <= threshold
    ]

    if prefer == "smaller":
        return min(eligible)
    if prefer == "larger":
        return max(eligible)

    raise ValueError("prefer must be 'smaller' or 'larger'")


# ============================================================
# Step 4 — polynomial_regression
# 多项式回归、嵌套 F 检验和曲线预测
# ============================================================

def poly_model(degree):
    # 先标准化年龄，避免高次幂数值过大。
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LinearRegression(),
    )


def poly_curve(X, y, degrees, cv):
    return cv_curve(poly_model, X, y, degrees, cv)


def anova_degrees(X, y, max_degree):
    n = len(y)
    results = []

    for degree in range(2, max_degree + 1):
        model_previous = poly_model(degree - 1).fit(X, y)
        model_current = poly_model(degree).fit(X, y)

        rss_previous = np.sum((y - model_previous.predict(X)) ** 2)
        rss_current = np.sum((y - model_current.predict(X)) ** 2)

        # 两个相邻阶数的多项式模型只相差一个参数。
        f_stat = (
            (rss_previous - rss_current)
            / (rss_current / (n - degree - 1))
        )
        p_value = f_dist.sf(f_stat, 1, n - degree - 1)

        results.append((
            degree,
            round(float(f_stat), 2),
            round(float(p_value), 4),
        ))

    return results


def choose_degree(X, y, degrees, cv, alpha=0.05):
    means, _ = poly_curve(X, y, degrees, cv)
    degree_min = degrees[int(np.argmin(means))]

    anova_results = anova_degrees(X, y, max(degrees))
    degree_anova = 1

    for degree, _, p_value in anova_results:
        if p_value < alpha:
            degree_anova = degree
        else:
            break

    return degree_min, degree_anova


def curve_on_grid(model, grid):
    predictions = model.predict(grid)
    return np.round(np.asarray(predictions).ravel(), 2)


# ============================================================
# Step 5 — step_functions
# 用 KBinsDiscretizer 将年龄变成区间指示变量
# ============================================================

def step_model(n_bins):
    return make_pipeline(
        KBinsDiscretizer(
            n_bins=n_bins,
            encode="onehot-dense",
            strategy="uniform",
        ),
        LinearRegression(),
    )


def step_curve(X, y, bins, cv):
    return cv_curve(step_model, X, y, bins, cv)


def choose_bins(X, y, bins, cv):
    means, ses = step_curve(X, y, bins, cv)
    bins_min = bins[int(np.argmin(means))]
    bins_1se = one_se_rule(bins, means, ses, prefer="smaller")
    return bins_min, bins_1se


def bin_edges(model):
    discretizer = model.named_steps["kbinsdiscretizer"]
    return [
        round(float(edge), 1)
        for edge in discretizer.bin_edges_[0]
    ]


def step_levels(model, grid):
    predictions = model.predict(grid)
    levels = np.unique(np.round(np.asarray(predictions).ravel(), 1))
    return [float(level) for level in levels]


# ============================================================
# Step 6 — spline_regression
# 三次回归样条模型，并选择节点数量
# ============================================================

def spline_model(n_knots, degree=3, extrapolation="constant"):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=degree,
            knots="quantile",
            extrapolation=extrapolation,
            include_bias=False,
        ),
        LinearRegression(),
    )


def spline_basis_size(model, X):
    transformer = model.named_steps["splinetransformer"]
    n_rows = min(5, len(X))
    transformed = transformer.transform(X.iloc[:n_rows])
    return transformed.shape[1]


def spline_curve(X, y, knot_counts, cv):
    return cv_curve(spline_model, X, y, knot_counts, cv)


def choose_knots(X, y, knot_counts, cv):
    means, ses = spline_curve(X, y, knot_counts, cv)
    knots_min = knot_counts[int(np.argmin(means))]
    knots_1se = one_se_rule(knot_counts, means, ses, prefer="smaller")
    return knots_min, knots_1se


# ============================================================
# Step 7 — extrapolation
# 比较多项式和样条在观测范围外的预测
# ============================================================

def beyond_data(models, ages):
    grid = pd.DataFrame({"age": ages})

    return {
        name: np.round(model.predict(grid), 1).tolist()
        for name, model in models.items()
    }


def extrapolation_report(X, y, ages):
    poly4 = poly_model(4).fit(X, y)
    spline_constant = spline_model(5).fit(X, y)
    spline_linear = spline_model(
        5,
        extrapolation="linear",
    ).fit(X, y)

    predictions = beyond_data(
        {
            "poly4": poly4,
            "spline_const": spline_constant,
            "spline_linear": spline_linear,
        },
        ages,
    )

    predictions["poly4_range"] = round(
        max(predictions["poly4"]) - min(predictions["poly4"]),
        1,
    )

    return predictions


# ============================================================
# Step 8 — smoothing_spline
# 用 Ridge 惩罚样条系数，控制曲线平滑程度
# ============================================================

def smooth_model(alpha, n_knots=20):
    return make_pipeline(
        SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            include_bias=False,
        ),
        Ridge(alpha=alpha),
    )


def effective_df(model, X):
    transformer = model.named_steps["splinetransformer"]
    ridge = model.named_steps["ridge"]

    basis = transformer.transform(X)
    basis = basis - basis.mean(axis=0)

    alpha = ridge.alpha
    n_basis = basis.shape[1]

    bt_b = basis.T @ basis
    penalized = bt_b + alpha * np.eye(n_basis)

    # 截距不受惩罚，所以额外加 1。
    edf = 1.0 + np.trace(np.linalg.inv(penalized) @ bt_b)

    return round(float(edf), 2)


def smooth_curve(X, y, alphas, cv):
    return cv_curve(smooth_model, X, y, alphas, cv)


def choose_alpha(X, y, alphas, cv):
    means, ses = smooth_curve(X, y, alphas, cv)
    alpha_min = alphas[int(np.argmin(means))]
    alpha_1se = one_se_rule(alphas, means, ses, prefer="larger")
    return alpha_min, alpha_1se


# ============================================================
# Step 9 — local_smoother
# 用最近邻回归近似局部平滑，并计算曲线粗糙度
# ============================================================

def local_model(span, n_train):
    n_neighbors = max(2, int(round(span * n_train)))
    return KNeighborsRegressor(n_neighbors=n_neighbors)


def local_curve(X, y, spans, cv):
    make_model = lambda span: local_model(span, len(X))
    return cv_curve(make_model, X, y, spans, cv)


def choose_span(X, y, spans, cv):
    means, ses = local_curve(X, y, spans, cv)
    span_min = spans[int(np.argmin(means))]
    span_1se = one_se_rule(spans, means, ses, prefer="larger")
    return span_min, span_1se


def roughness(curve):
    second_differences = np.diff(curve, n=2)
    return round(float(np.mean(np.abs(second_differences))), 3)


# ============================================================
# Step 10 — gam_pipeline
# 年龄和年份用样条，教育程度用 one-hot 编码
# ============================================================

def gam_preprocessor(age_knots=5, year_knots=4):
    return ColumnTransformer(
        transformers=[
            (
                "age",
                SplineTransformer(
                    n_knots=age_knots,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
                ["age"],
            ),
            (
                "year",
                SplineTransformer(
                    n_knots=year_knots,
                    degree=3,
                    knots="uniform",
                    include_bias=False,
                ),
                ["year"],
            ),
            (
                "education",
                OneHotEncoder(drop="first"),
                ["education"],
            ),
        ]
    )


def gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LinearRegression(),
    )


def gam_xy(df):
    X = df[["age", "year", "education"]]
    y = df["wage"]
    return X, y


def gam_feature_count(model, X):
    preprocessor = model.named_steps["columntransformer"]
    transformed = preprocessor.transform(X)
    return transformed.shape[1]


# ============================================================
# Step 11 — partial_effects
# 固定其他变量，只改变一个变量，观察它的相对影响
# ============================================================

def _baseline_row(X):
    baseline = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            baseline[column] = X[column].median()
        else:
            baseline[column] = X[column].mode().iloc[0]

    return baseline


def partial_effect(model, X, column, grid_values):
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(grid_values),
        columns=X.columns,
    )
    effect_data[column] = grid_values

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return np.round(np.asarray(centered).ravel(), 2)


def education_effect(model, X):
    levels = sorted(X["education"].unique())
    baseline = _baseline_row(X)

    effect_data = pd.DataFrame(
        [baseline] * len(levels),
        columns=X.columns,
    )
    effect_data["education"] = levels

    predictions = model.predict(effect_data)
    centered = predictions - np.mean(predictions)

    return {
        level: round(float(effect), 2)
        for level, effect in zip(levels, centered)
    }


def gam_summary(model, X):
    age_values = age_grid()["age"].to_numpy()
    age_effect = partial_effect(
        model, X, "age", age_values
    )

    year_values = np.arange(2003, 2010)
    year_effect = partial_effect(
        model, X, "year", year_values
    )

    edu_effects = education_effect(model, X)
    edu_values = np.array(list(edu_effects.values()))

    return {
        "age_range": round(
            float(np.max(age_effect) - np.min(age_effect)), 1
        ),
        "year_range": round(
            float(np.max(year_effect) - np.min(year_effect)), 1
        ),
        "education_range": round(
            float(np.max(edu_values) - np.min(edu_values)), 1
        ),
    }


# ============================================================
# Step 12 — logistic_gam
# 用 GAM 特征处理加逻辑回归，预测高收入概率
# ============================================================

class _HighEarnerSeries(pd.Series):
    @property
    def _constructor(self):
        return _HighEarnerSeries

    def unique(self):
        return [int(value) for value in super().unique()]


def high_earner(df):
    target = (df["wage"] > 250).astype(np.int32)
    return _HighEarnerSeries(target)


def logistic_gam_model(age_knots=5, year_knots=4):
    return make_pipeline(
        gam_preprocessor(age_knots, year_knots),
        LogisticRegression(max_iter=2000),
    )


def high_earner_probability(model, X, ages):
    baseline = _baseline_row(X)

    prediction_data = pd.DataFrame(
        [baseline] * len(ages),
        columns=X.columns,
    )
    prediction_data["age"] = ages

    probabilities = model.predict_proba(prediction_data)[:, 1]
    return np.round(probabilities, 4)


def logistic_gam_auc(X, target, cv):
    model = logistic_gam_model()
    scores = cross_val_score(
        model,
        X,
        target,
        cv=cv,
        scoring="roc_auc",
    )
    return round(float(np.mean(scores)), 3)


# ============================================================
# Step 13 — fit_age_models
# 用交叉验证/one-SE 规则选择参数，并拟合各个年龄模型
# ============================================================

def fit_age_models(X, y, cv):
    poly_degrees = [1, 2, 3, 4, 5, 6]
    step_bins = [2, 4, 8, 16]
    spline_knots = [3, 4, 5, 6, 8, 12]
    smooth_alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    local_spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]

    linear = poly_model(1).fit(X, y)

    _, poly_degree = choose_degree(
        X, y, poly_degrees, cv
    )
    poly = poly_model(poly_degree).fit(X, y)

    _, bins_1se = choose_bins(
        X, y, step_bins, cv
    )
    step = step_model(bins_1se).fit(X, y)

    _, knots_1se = choose_knots(
        X, y, spline_knots, cv
    )
    spline = spline_model(knots_1se).fit(X, y)

    _, alpha_1se = choose_alpha(
        X, y, smooth_alphas, cv
    )
    smooth = smooth_model(
        alpha_1se,
        n_knots=20,
    ).fit(X, y)

    _, span_1se = choose_span(
        X, y, local_spans, cv
    )
    local = local_model(
        span_1se,
        len(X),
    ).fit(X, y)

    return {
        "linear": (linear, 1),
        "poly": (poly, poly_degree),
        "step": (step, bins_1se),
        "spline": (spline, knots_1se),
        "smooth": (smooth, alpha_1se),
        "local": (local, span_1se),
    }


def curves_table(models, grid):
    table = {
        name: curve_on_grid(model, grid)
        for name, (model, _) in models.items()
    }

    return pd.DataFrame(
        table,
        index=grid["age"].to_numpy(),
    )


# ============================================================
# Step 14 — test_comparison
# 最后在测试集上比较模型 RMSE
# ============================================================

def test_rmse(models, X_test, y_test):
    rmse = {}

    for name, (model, _) in models.items():
        predictions = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, predictions))
        rmse[name] = round(float(score), 2)

    return rmse


def gam_test_rmse(train, test):
    X_train, y_train = gam_xy(train)
    X_test, y_test = gam_xy(test)

    model = gam_model().fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = np.sqrt(mean_squared_error(y_test, predictions))
    return round(float(score), 2)


def comparison_lines(rmse_by_model, gam_rmse):
    sorted_models = sorted(
        rmse_by_model.items(),
        key=lambda item: item[1][0],
    )

    lines = [
        f"{name:8s} rmse={rmse:6.2f} setting={setting}"
        for name, (rmse, setting) in sorted_models
    ]

    lines.append(
        f"{'gam':8s} rmse={gam_rmse:6.2f} "
        "setting=age+year+education"
    )

    return lines

