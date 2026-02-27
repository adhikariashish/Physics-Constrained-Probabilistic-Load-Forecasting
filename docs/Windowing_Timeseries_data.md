## Timeline Visualization (ERCOT 2019–2025 Example)

Assume:
- Training ends: `2022-12-31`
- Validation starts: `2023-01-01`
```
2019 --------------------------------------------------------- 2025
|------------------- TRAIN -------------------|---- VAL ----|-- TEST --|
2019-01-01                              2022-12-31  2023-01-01
```

---

## Example Window Inside TRAIN

Assume:
```
t = 2021-07-10 23:00
```

Window looks like:
```
<-------------------- 168 hours -------------------->  <--- 24 hours --->
|                                                     |
2021-07-03 00:00 ........................ 2021-07-10 23:00 | 2021-07-11 ...
                     (X context)                           (y target)
```

This creates one supervised example:
- **X:** July 3 → July 10
- **y:** July 11

---

## Why We Split Before Windowing

We must avoid **data leakage**.

If training ends on `2022-12-31` and validation starts on `2023-01-01`, we must **NOT** allow:
```
X = Dec 25 → Dec 31
y = Jan 1
```

That would leak validation data into training.

Instead:
1. Slice dataset into train/val/test
2. Build windows independently inside each slice

This ensures **strict temporal isolation**.

---

## Window Validity Constraints

To build a valid window:
```
t >= context_length - 1
t <= n - horizon - 1
```

Because:
- We need enough history for **X**
- We need enough future for **y**

Invalid windows are discarded.

---

## Shapes Produced

If:
- `context_length = 168`
- `horizon = 24`
- `feature_dim = 1` (system dataset)

Then:
```
X: [batch_size, 168, 1]
y: [batch_size, 24]
```

If using zonal features (8 zones):
```
X: [batch_size, 168, 8]
y: [batch_size, 24]
```

---

## Intuition: What the Model Learns

Each training example teaches the model:

> *When the past week looked like **THIS**, the next day looked like **THAT**.*

Across thousands of windows, the model learns:
- Daily cycles
- Weekly seasonality
- Seasonal patterns (summer vs winter)
- Long-term demand trends