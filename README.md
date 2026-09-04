# Phishing Email Classifier

A binary classifier that flags an email as **Phishing** or **Legitimate**, built as
two independent models trained on the same data:

- **Aetheris-Lite** — SGD (linear), fast, tuned for high precision.
- **NexusVector-X** — XGBoost (gradient-boosted trees), stronger recall on phishing.

Try it live: **[foxlens-btjedlx5wjw3igdbzdxttw.streamlit.app]**

## How it works

Each model takes four inputs per email: the combined subject+body text, whether it
has an attachment, the number of links, and the links' text. Both were trained on
the [MeAJOR Corpus](https://zenodo.org/records/18471483) (~108k labeled emails)
with a 10% holdout carved out — shuffled — before any training or threshold
tuning, and never touched until final evaluation.

| | Test set | Holdout |
|---|---|---|
| **Aetheris-Lite** | P: 0.926, R: 0.835 | P: 0.921, R: 0.824 |
| **NexusVector-X** | P: 0.971, R: 0.952 | P: 0.961, R: 0.957 |

## A known limitation — worth reading before you trust either model

While testing the demo, I fed NexusVector-X this real-style phishing email:

> *"URGENT: Your PayPal Account Has Been Temporarily Restricted! ... verify your
> identity within 24 hours ... http://security-paypal-verification-session89.xyz/..."*

With the link included, NexusVector-X flagged it **Phishing at 97% confidence**.
I then removed only the link, keeping every threatening, urgent word in the body
unchanged. The verdict flipped to **Legitimate at 64% confidence** (36% phishing
probability).

I checked why: **81.9% of phishing emails in the training data contain at least
one link** (18.1% don't — 7,735 of 42,802). Because a link is such a strong,
easy-to-use signal on this dataset, the model leaned on it heavily and
under-weighted the actual threatening language in the message body. It works
very well for the phishing pattern it saw most, and is measurably weaker against
phishing that doesn't rely on a clickable link (image-based lures, "reply with
your details," QR codes, etc.).

This wasn't fixed retroactively — it's flagged here on purpose, from the actual
number, not a guess. The direction for a future retrain: rebalance or engineer
features so the text itself carries more weight relative to link presence
(e.g. oversampling link-free phishing examples, or inspecting feature
importance/SHAP values to confirm and correct the imbalance).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — the Streamlit interface
- `Aetheris-Lite.joblib`, `NexusVector-X.joblib` — the trained models
- `requirements.txt` — pinned to the exact library versions used at training time
  (mismatched versions will fail to unpickle the models)

## Disclaimer

The confidence shown in the app reflects the model's certainty in its own
prediction — not a guarantee of correctness. Treat it as decision support, not
statistical fact, especially given the limitation above.
