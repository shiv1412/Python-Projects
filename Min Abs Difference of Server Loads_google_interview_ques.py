# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:10:14 2020

@author: sharm
"""

def minAbs(nums):
    s = sum(nums)//2
    n = len(nums)
    dp = [[0 for _ in range(s+1)] for x in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,s+1):
            if nums[i-1]<=j:
                dp[i][j]=max(dp[i-1][j], nums[i-1]+dp[i-1][j-nums[i-1]])
            else:
                dp[i][j]=dp[i-1][j]
                
    return (sum(nums)-dp[-1][-1]-dp[-1][-1])

print(minAbs([1,2,3,4,5]))