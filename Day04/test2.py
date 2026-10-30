import pandas as pd

fake_data = pd.DataFrame({
    "payment_method": ["card", "cash", "card", "paypal", "cash", "card", "paypal", "card", "cash", "paypal"]
})

print(fake_data["payment_method"].value_counts(normalize=True))