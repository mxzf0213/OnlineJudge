#include <stdio.h>

int solve(int n) {
    int dp[n+2];
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i < n; i++) {
        dp[i] = dp[i-2] + dp[i-1];
    }

    if (n >= 9) {
        return dp[n-1] - (2 * dp[n-9]);
    }
    return dp[n-1];
}

int main() {
    int n;
    scanf("%d", &n);
    printf("%d\n", solve(n));
    return 0;
}