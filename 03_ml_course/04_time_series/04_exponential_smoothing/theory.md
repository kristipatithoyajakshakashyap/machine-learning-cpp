# Theory

## Components

Additive Holt-Winters describes a series as level + trend + seasonal offset. After observing y_t the three states are updated with one-step errors:

- level:  l_t = alpha (y_t - s_{t-m}) + (1 - alpha)(l_{t-1} + b_{t-1})
- trend:  b_t = beta (l_t - l_{t-1}) + (1 - beta) b_{t-1}
- season: s_t = gamma (y_t - l_t) + (1 - gamma) s_{t-m}

m is the season length (12 for monthly data). The one-step forecast made after observing y_t is l_t + b_t + s_{t+1-m}: the current level, one more step of trend, and the seasonal offset last stored for the coming position.

## Weights

alpha, beta, gamma lie in [0, 1]. A weight near 1 makes the component follow the newest observation almost completely; a weight near 0 freezes it near its initial value. Because each update is a convex combination, the implied weights on past observations decay geometrically, which is where the name comes from. There is no closed-form best weight; the module scores a small grid on rolling origins.

## Initialisation

The recursions need a starting state. This module uses the first two seasons: the level is the mean of season one, the trend is (mean of season two - mean of season one) / m, and each seasonal offset averages the deviation of that position from its own season mean. Any initialisation works in the limit because old states are forgotten, but with only twelve years of data the start still matters, which is another reason to validate.

## Additive versus multiplicative

Additive: seasonal swings have constant size in the units of the series. Multiplicative: swings are proportional to the level (peaks grow with the series), which is the textbook description of AirPassengers. The multiplicative recursions replace subtractions by divisions (l_t = alpha y_t / s_{t-m} + ...). The additive model is kept here for clarity; applying it to log(y) is the usual way to get multiplicative behaviour without new code. The held-out residual figure shows what additive seasonality misses late in the series.

## When to prefer it

Prefer Holt-Winters over lag regression when the series has a smooth trend and a stable seasonal shape, when you need a model with a handful of interpretable numbers, or when there are no useful external regressors. Prefer lag regression when exogenous features matter or when the seasonal shape changes over time. Both are one-step models here; a multi-step Holt-Winters forecast would extend the trend h steps and reuse the stored seasonal offsets.
