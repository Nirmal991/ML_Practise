from thefuzz import fuzz, process

choices = ["month-to-month", "m2m", "1-year"]
query = "month to month"

match = process.extractOne(query, choices, scorer=fuzz.ratio)
print(match)
print(f"Best match for '{query}' is '{match[0]}' with score {match[1]}")
print("_" * 60)
a = "Virat Kohli"
b = "Virat Kohly"
a = "Virat Kohli"
b = "Virat Kohly"
score = fuzz.ratio(a, b)
print(f"Fuzzy match score for '{a}' vs '{b}' is {score}")
print("_" * 60)
a = "AECOM"
b= "AECOM Technology Corporation"
score = fuzz.partial_ratio(a, b)
print(f"Fuzzy match score for '{a}' vs '{b}' is {score}")
print("_" * 60)
a = "Virat Kohli"
b= "Kohli Virat"
score = fuzz.token_sort_ratio(a, b)
print(f"Fuzzy match score for '{a}' vs '{b}' is {score}")
print("_" * 60)

a = "The Quick Brown Fox"
b = "Quick Brown Fox Fox Fox The"
score = fuzz.token_set_ratio(a, b)
print(f"Fuzzy match score for '{a}' vs '{b}' is {score}")