# Math intuition

x_t=[y_(t-1),y_(t-2),y_(t-12),t,sin(2*pi*t/12),cos(2*pi*t/12)]. Fit squared loss plus an L2 penalty. Sine and cosine avoid a discontinuity between December and January; lag-12 captures yearly repetition.
