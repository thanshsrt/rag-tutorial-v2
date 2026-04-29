# Manual test
from evaluation import evaluate_review

diff = "def foo():\n    pass"
review = "The `foo` function in `auth.py` needs type hints. Consider adding them."
sources = ["src/auth.py"]

result = evaluate_review(diff, review, sources)
print(result)
# Expected: score ~8.5, passed=True