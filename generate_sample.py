import numpy as np
import pandas as pd

np.random.seed(42)
N = 8760  # hours in a year

hours = np.arange(N)
day_of_year = hours // 24
hour_of_day = hours % 24

# ── Solar yield (kW) ─────────────────────────────────────────────────────────
# 500 kWp system, Germany-ish capacity factor ~0.11
# Seasonal envelope: peaks in June (day 172), zero Nov-Jan nights
seasonal = np.clip(np.sin(np.pi * (day_of_year - 80) / 185), 0, 1)
# Daily bell curve: peaks at 12:00
daily = np.exp(-0.5 * ((hour_of_day - 12) / 3.5) ** 2)
solar_base = 500 * seasonal * daily * 0.85
# Add cloud variability
cloud = np.random.beta(2, 1, N)  # skewed toward clear sky
solar = np.round(solar_base * cloud, 2)
solar = np.clip(solar, 0, 500)

# ── Load demand (kW) ─────────────────────────────────────────────────────────
# Industrial/commercial: 70-200 kW, weekday peaks, lower weekends
day_of_week = (day_of_year % 7)
is_weekday = (day_of_week < 5).astype(float)
# Daily load profile
morning_ramp  = np.clip((hour_of_day - 6) / 3, 0, 1)
evening_ramp  = np.clip((20 - hour_of_day) / 3, 0, 1)
daily_load    = morning_ramp * evening_ramp
base_load     = 50 + 120 * daily_load * (0.7 + 0.3 * is_weekday)
noise         = np.random.normal(0, 8, N)
load          = np.round(np.clip(base_load + noise, 20, 250), 2)

# ── Export price (€/kWh) — Germany espot style ───────────────────────────────
# Average ~€0.07/kWh, volatile, ~400 negative hours
seasonal_price = 0.06 + 0.04 * np.sin(2 * np.pi * day_of_year / 365)
daily_price    = 0.02 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
price_noise    = np.random.normal(0, 0.025, N)
export_price   = seasonal_price + daily_price + price_noise
# Inject ~380 negative price hours (high-solar midday periods, summer)
neg_mask = (hour_of_day >= 10) & (hour_of_day <= 15) & (day_of_year > 120) & (day_of_year < 250)
neg_prob = np.random.random(N)
neg_hours = neg_mask & (neg_prob < 0.35)
export_price[neg_hours] -= np.random.uniform(0.05, 0.25, neg_hours.sum())
export_price = np.round(np.clip(export_price, -0.50, 0.50), 4)

# ── Assemble and save ─────────────────────────────────────────────────────────
df = pd.DataFrame({
    'load_demand':  load,
    'solar_yield':  solar,
    'export_price': export_price
})

print(f"Rows: {len(df)}")
print(f"Load:   {load.sum():.0f} kWh/year,  avg {load.mean():.1f} kW,  peak {load.max():.1f} kW")
print(f"Solar:  {solar.sum():.0f} kWh/year, avg {solar.mean():.1f} kW,  CF={solar.sum()/(500*8760):.3f}")
print(f"Price:  avg={export_price.mean():.4f} €/kWh, neg hours={neg_hours.sum()}")
df.to_csv('sample_data/sample.csv', index=False)
print("Saved: sample_data/sample.csv")
