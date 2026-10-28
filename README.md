# ⚡ Dynamic Sector & Equity RRG Terminal

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?logo=plotly&logoColor=white)
![yfinance](https://img.shields.io/badge/Data-yfinance-00E676)

An interactive financial terminal built with **Python**, **Streamlit**, **Plotly**, and **yfinance** to visualize market leadership transitions using **Relative Rotation Graphs (RRG)**. 

Unlike static sector trackers, this terminal combines macro S&P 500 sector rotation with on-the-fly custom equity overlays and a **2D Vector Kinematics Scanner** that measures tail velocity, acceleration, and polar heading angles to isolate high-conviction breakouts.

---

## ✨ Key Features

* **📺 Live HTML/CSS Quadrant Ticker Tape:** A zero-latency, hover-pausable top marquee fed directly from cached price and RRG dataframes. Displays last price, bar percentage change, current RRG quadrant badge (`LEADING`, `WEAKENING`, `LAGGING`, `IMPROVING`), directional arrow, and instantaneous tail velocity.
* **🎯 Two-Tier Interactive Plotly RRG:** Simultaneously track all 11 SPDR Sector ETFs (`XLK`, `XLF`, `XLI`, `XLE`, etc.) alongside any custom ticker list (`RKLB, GEV, PLTR`). In `Sectors + Custom` mode, sector ETFs render at 42% opacity as a macro backdrop while custom targets pop at 100% opacity with diamond head markers and bold spline trails.
* **🔄 Dynamic Benchmark & Timeframe Switching:** Instantly re-center the RRG origin `(100, 100)` from `SPY` to `QQQ`, `IWM`, or a specific sector ETF (such as `XLK` or `XLI`) to evaluate how individual stocks are rotating relative to their direct sector peers across Weekly (`1wk`) or Daily (`1d`) bars.
* **🚀 2D Tail Kinematics Scanner:** Treats every RRG trail as a trajectory in `(RS-Ratio, RS-Momentum)` space, allowing you to filter and rank assets by instantaneous speed, positive tail acceleration, and Northeast (`0°–90°`) breakout headings.

---

## 📐 Mathematical Methodology

### 1. Relative Strength Ratio & Momentum
For each asset $i$ relative to benchmark $B$:
1. **Raw Relative Strength ($RS_t$):** Calculated as $100 \times (P_{i,t} / P_{B,t})$ and smoothed with an exponential moving average.
2. **RS-Ratio ($R_t$):** Rolling Z-score of the smoothed relative strength over lookback window $W$, normalized around `100.0`:
   $$R_t = 100 + 2.0 \times \left( \frac{RS_t - \mu_{RS, W}}{\sigma_{RS, W}} \right)$$
3. **RS-Momentum ($M_t$):** Rolling Z-score of the 1-bar rate-of-change of $R_t$, normalized around `100.0`.

### 2. Vector Kinematics in RRG Space
Given single-bar displacements $\Delta R_t = R_t - R_{t-1}$ and $\Delta M_t = M_t - M_{t-1}$:
* **1-Bar Tail Velocity ($\vert{}\vec{v}_t\vert{}$):**
  $$\vert{}\vec{v}_t\vert{} = \sqrt{(\Delta R_t)^2 + (\Delta M_t)^2}$$
* **Tail Acceleration ($a_t$):**
  $$a_t = \vert{}\vec{v}_t\vert{} - \vert{}\vec{v}_{t-1}\vert{}$$
* **Polar Heading Angle ($\theta_t$):**
  $$\theta_t = \operatorname{atan2}(\Delta M_t, \Delta R_t) \pmod{360^\circ}$$
  * **`0° – 90°` (↗ NE - Power Zone):** Gaining both relative trend and momentum simultaneously.
  * **`90° – 180°` (↖ NW - Early Recovery):** Momentum rising while relative trend bottoms.
  * **`180° – 270°` (↙ SW - Breakdown):** Losing both relative trend and momentum.
  * **`270° – 360°` (↘ SE - Rollover):** Relative trend still positive, but momentum is decelerating.

---

## 🛠️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/dynamic-rrg-terminal.git](https://github.com/YOUR_USERNAME/dynamic-rrg-terminal.git)
cd dynamic-rrg-terminal
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python -m streamlit run rrg_terminal.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 💡 Usage Tips

1. **Macro-to-Micro Workflow:** Start in `Sectors Only` mode with `SPY` as the benchmark on the `Weekly (1wk)` timeframe to identify which sectors are rotating into **Improving** or **Leading**.
2. **Peer Relative Drilling:** Switch to `Custom Only`, enter a basket of stocks within that leading sector, and change the **Benchmark** box from `SPY` to the sector ETF (e.g., `XLK` or `XLI`) to isolate the true alpha leaders inside that group.
3. **Kinematics Filtering:** In the scanner table below the chart, filter by **`↗ NE (Power)`** and check **`⚡ Positive Acceleration Only`** to surface assets whose RRG tails are actively lengthening toward the top-right quadrant.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
