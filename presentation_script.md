# 🎬 Presentation Script — Predictive Maintenance (Offshore)

**Estimated length:** ~4–6 minutes · **First person · Speak naturally**

`[SHOW ...]` = what to display on screen. `[read the number]` = fill in the real
value shown in the app's **Evaluation** tab before recording.

---

## 0. Opening — (~20s)

> "Hi, I'm Arthur. In this video I'll walk you through **Project 2** of
> *Managing Machine Learning Projects*: a predictive-maintenance prototype for
> **critical offshore equipment** — engines, pumps and thrusters. The goal is to
> **predict equipment failure within the next 72 hours** so a maintenance team
> can act before a breakdown at sea."

`[SHOW: the deployed Streamlit app, Overview tab]`

---

## 1. The problem & framing — (~45s)

> "This is framed as a **binary classification** task: for each sensor reading,
> will this unit fail within 72 hours — yes or no? The user is an **onshore
> maintenance and reliability manager** who needs an early warning."
>
> "A key challenge is that **failures are rare** — only about **3% of readings**.
> As you can see here, the dataset has [SHOW metrics] thousands of readings
> across several equipment units, with that small failure rate. The data here is
> **synthetic**, generated to validate the pipeline; in production you'd plug in
> the fleet's real sensor data."

`[SHOW: Overview tab — point at the three metrics: Readings, Equipment units, Failures %]`

---

## 2. Why accuracy is the wrong metric — (~30s)

> "Because the classes are so imbalanced, **accuracy is misleading**: a model
> that always predicts 'never fails' would score around 97% accuracy and be
> completely useless. So I evaluate with **PR-AUC, recall and precision**, and I
> use `class_weight = balanced` so the model pays attention to the rare failure
> cases."

`[SHOW: Overview tab — "Method decisions" section]`

---

## 3. Data understanding (EDA) — (~45s)

> "Let's look at the data. This chart shows the **class balance** — failures are
> the small bar on the right. The **correlation heatmap** shows how the sensors
> relate to the target: vibration and temperature rise as a unit degrades, while
> oil pressure drops."
>
> "And these distributions compare **healthy units versus units about to fail** —
> you can clearly see the curves separate, which is exactly the signal the model
> will learn from."

`[SHOW: EDA tab — scroll through the three charts]`

---

## 4. Modeling & validation — the no-leakage split — (~50s)

> "For modeling, the most important decision is **how I split the data**.
> Consecutive readings from the same machine are almost identical, so a random
> split would leak information and inflate the results. Instead, I split **by
> equipment** using GroupKFold — the model is **tested on machines it has never
> seen**, just like a new ship joining the fleet."
>
> "I compare three models — Logistic Regression, Random Forest and Gradient
> Boosting — with 5-fold cross-validation, scored by PR-AUC. Let me run it."

`[SHOW: Modeling tab — click "Run comparison", wait for the bar chart]`

> "Interestingly, the simple **Logistic Regression** baseline generalized best to
> new equipment — a good reminder that the most complex model doesn't always
> win."

`[SHOW: the comparison table and bar chart]`

---

## 5. Evaluation — choosing the threshold by recall — (~50s)

> "Now the final evaluation on the **held-out test equipment**. Here are the
> headline metrics: a **PR-AUC of [read the PR-AUC]** and **ROC-AUC of
> [read the ROC-AUC]**."
>
> "Since **missing a real failure is far more expensive than a false alarm**, I
> don't just use a 0.5 cutoff. I choose the **threshold that captures at least
> 80% of real failures**, and then report the resulting precision — here, a
> recall of **[read recall]** at a precision of **[read precision]**."
>
> "The **precision-recall curve** and the **confusion matrix** show this
> trade-off, and the **feature-importance** chart confirms the model relies on
> the degradation signals we saw in the EDA — vibration and temperature."

`[SHOW: Evaluation tab — metrics, PR curve, confusion matrix, feature importance]`

---

## 6. Live demo — interactive prediction — (~40s)

> "Finally, the part that makes it tangible for the user. On the **Prediction**
> tab, I can simulate a unit's sensors. With **normal readings**, the failure
> probability stays low — green, no action."
>
> "But if I push **vibration and temperature up** and **oil pressure down**,
> mimicking a degrading machine, the probability climbs above the decision
> threshold and the app raises a **red alert** recommending inspection. That's
> the early warning the maintenance manager would receive."

`[SHOW: Prediction tab — move sliders from normal → degraded, show OK then ALERT]`

---

## 7. Closing — (~25s)

> "To wrap up: this prototype maps directly onto the **CRISP-DM phases** — data
> understanding, preparation and modeling, and evaluation — and the design
> choices are driven by the **business cost of missing a failure**. The full code
> and this live app are on GitHub and Streamlit Cloud. Thanks for watching."

`[SHOW: brief return to Overview tab, or the GitHub repo page]`

---

## 📝 Recording tips

- **Fill in the real numbers** from the Evaluation tab (PR-AUC, ROC-AUC, recall,
  precision) before recording.
- Run the **Modeling** comparison **once before** you hit record so it's cached
  and loads instantly on camera.
- Total spoken time lands around **5 minutes**; trim sections 2 or 4 if you need
  it shorter.
