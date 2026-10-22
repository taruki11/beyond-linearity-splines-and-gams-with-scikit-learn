"""
Beyond Linearity: Splines and GAMs with Scikit-Learn scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Beyond Linearity: Splines and GAMs with scikit-learn (ISL, chapter 7).

Story: load the book's Wage data and hold out a test set; fit wage against age with
polynomials (degree chosen by nested F-tests, as in the book), step functions,
regression splines, a penalized smoothing spline and a local smoother, choosing
the rest by cross-validation with the one-standard-error rule; watch a quartic explode past the data while a spline with
linear tails does not; build an additive model of age, year and education with a
ColumnTransformer, read its partial effects, turn it into a logistic GAM for high
earners; then open the test set once and compare every curve.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def main() -> None:
    df = load_wage()
    info = describe_wage(df)
    print(f"Wage data: n={info['n']}, age {info['age_range'][0]}-{info['age_range'][1]}, mean wage {info['wage_mean']}k, "
          f"{info['n_education_levels']} education levels")
    train, test = split_wage(df)
    X, y = age_xy(train)
    cv = KFold(5, shuffle=True, random_state=0)
    grid = age_grid()
    print(f"train {len(train)} / test {len(test)}; the test set is opened once, at the end")

    # ---- 1. Polynomials and step functions ----
    degrees = [1, 2, 3, 4, 5, 6]
    pm, ps = poly_curve(X, y, degrees, cv)
    d_min, d_anova = choose_degree(X, y, degrees, cv)
    print("\npolynomial degree, CV MSE: " + "  ".join(f"d{d}={m:.0f}" for d, m in zip(degrees, pm)) + f"  (fold SE about {np.mean(ps):.0f}: the curve is flat, CV minimum at degree {d_min})")
    print("  nested F-tests: " + "  ".join(f"d{d}: F={F:.1f} p={p:.4f}" for d, F, p in anova_degrees(X, y, 5)) + f"  -> ANOVA keeps degree {d_anova}")
    bins = [2, 4, 8, 16]
    sm, _ = step_curve(X, y, bins, cv)
    b_min, b_1se = choose_bins(X, y, bins, cv)
    print("step functions, CV MSE by bins: " + "  ".join(f"{b}={m:.0f}" for b, m in zip(bins, sm)) + f"  -> one-SE {b_1se} bins, edges {bin_edges(step_model(b_1se).fit(X, y))}")

    # ---- 2. Regression splines and extrapolation ----
    knots = [3, 4, 5, 6, 8, 12]
    km, _ = spline_curve(X, y, knots, cv)
    k_min, k_1se = choose_knots(X, y, knots, cv)
    print("\ncubic regression spline, CV MSE by knots: " + "  ".join(f"{k}={m:.0f}" for k, m in zip(knots, km)) + f"  -> one-SE {k_1se} knots")
    ages_beyond = [70, 80, 90, 100]
    r = extrapolation_report(X, y, ages_beyond)
    print(f"beyond the data at ages {ages_beyond}:")
    print(f"  degree-4 polynomial : {r['poly4']}  (range {r['poly4_range']})")
    print(f"  spline, constant tail: {r['spline_const']}")
    print(f"  spline, linear tail  : {r['spline_linear']}")

    # ---- 3. Smoothing and local fits ----
    alphas = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    a_min, a_1se = choose_alpha(X, y, alphas, cv)
    print("\npenalized spline (20 knots): " + "  ".join(f"alpha={a:g}: df={effective_df(smooth_model(a).fit(X, y), X):.1f}" for a in alphas))
    print(f"  CV minimum at alpha={a_min:g}; one-SE rule picks alpha={a_1se:g}")
    spans = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7]
    s_min, s_1se = choose_span(X, y, spans, cv)
    rough = {s: roughness(curve_on_grid(local_model(s, len(X)).fit(X, y), grid)) for s in (0.02, s_1se)}
    print(f"local smoother: CV minimum at span {s_min}, one-SE picks {s_1se}; roughness of the curve at span 0.02 = {rough[0.02]}, at {s_1se} = {rough[s_1se]}")

    # ---- 4. The additive model ----
    Xg, yg = gam_xy(train)
    gam = gam_model().fit(Xg, yg)
    summary = gam_summary(gam, Xg)
    g_cv, _ = cv_mse(gam_model(), Xg, yg, cv)
    a_cv, _ = cv_mse(spline_model(k_1se), X, y, cv)
    print(f"\nGAM wage ~ s(age) + s(year) + education: {gam_feature_count(gam, Xg)} basis columns; CV MSE {g_cv:.0f} vs age-only spline {a_cv:.0f}")
    print(f"  partial-effect ranges: education {summary['education_range']}, age {summary['age_range']}, year {summary['year_range']} (thousand dollars)")
    edu = education_effect(gam, Xg)
    print("  education effects: " + ", ".join(f"{lvl.split('. ')[1]} {v:+.1f}" for lvl, v in edu.items()))
    t = high_earner(train)
    lgam = logistic_gam_model().fit(Xg, t)
    probs = high_earner_probability(lgam, Xg, [25, 35, 45, 55, 65])
    auc = logistic_gam_auc(Xg, t, StratifiedKFold(5, shuffle=True, random_state=0))
    print(f"  logistic GAM for wage > 250k ({int(t.sum())} of {len(t)} workers): P(high) at ages 25..65 = {probs.tolist()}, CV AUC {auc}")

    # ---- 5. Test set, opened once ----
    models = fit_age_models(X, y, cv)
    Xt, yt = age_xy(test)
    rmse = test_rmse(models, Xt, yt)
    g_rmse = gam_test_rmse(train, test)
    print("\ntest-set RMSE (thousand dollars):")
    for line in comparison_lines({n: (rmse[n], models[n][1]) for n in models}, g_rmse):
        print("  " + line)
    table = curves_table(models, grid)
    print("fitted wage at ages 20/40/60/80: " + ", ".join(f"{n} {table[n].iloc[[2, 22, 42, 62]].round(0).astype(int).tolist()}" for n in ("linear", "poly", "spline", "smooth")))


if __name__ == "__main__":
    main()

