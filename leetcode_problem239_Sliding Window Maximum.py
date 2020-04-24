# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:26:26 2020

@author: sharm
"""

from collections import deque

class Solution:
    def maxSlidingWindow(self,nums: List[int],k:int) -> List[int]:
        q = deque([])
        maxs = []
        for i,num in enumerate(nums):
            while q and q[-1][0] < num:
                q.pop()
                q.append((num,i))
                
                if i >=k -1:
                    if q[0][1] < i-k+1:
                        q.popleft()
                    maxs.append(q[0][0])
        return maxs