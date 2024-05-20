from typing import Callable, TypeVar, Generic

from string import digits

# Generic type
T = TypeVar("T")

# Lazy wrapper to do lazy initialize.
# from: [https://stackoverflow.com/questions/7151890/python-lazy-variables-or-delayed-expensive-computation]
class LazyWrapper(Generic[T]):
    def __init__(self, func: Callable[[], T]):
        self.func = func
    def __call__(self) -> T:
        try:
            return self.value
        except AttributeError:
            self.value = self.func()
            return self.value

# Removes numerics in the second item of given pair.
def remove_numerics(pair: tuple[str, str]):
    return (pair[0],
    ''.join(w.translate({ord(k): None for k in digits}) for w in pair[1]))

# Calculate the length of LCS.
def get_lcs_len(A: list[str], B: list[str]) -> int:
    la = len(A)
    lb = len(B)
    dp = [[0 for i in range(lb + 1)] for j in range(la + 1)]
    for i in range(la):
        for j in range(lb):
            dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
            if A[i].lower() == B[j].lower():
                dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j] + 1)
    return dp[la][lb]
