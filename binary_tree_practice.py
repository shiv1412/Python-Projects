# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:19:06 2020

@author: sharm
"""
#pre order traveral on binary tree
#rescursive approach

def preorderTraversal1(self, root):
    res = []
    self.dfs(root,res)
    return res

def dfs(self,root,res):
    if root:
        res.append(root.val)
        self.dfs(root.left,res)
        self.dfs(root.right,res)
        print(dfs)
        
#iterative approach

def preorderTraversal2(self,root):
    def preorderTraversal2(self,root):
        stack , res = [] , []
        if root:
                stack.append(root)
                res.append(root.val)
                root = root.left
        else:
                node = stack.pop()
                root = node.right
        return res
            
#in order traveral on binary tree
#rescursive approach

## Recursively
def inorderTraversal(self, root):
	res = []
	self.helper(root, res)
	return res

def helper(self, root, res):
	if root:
		self.helper(root.left, res)
		res.append(root.val)
		self.helper(root.right, res)

## Iteratively
def inorderTraversal(self, root):
    res = []
    stack = []
    while stack or root:
        if root:
            stack.append(root)
            root = root.left
        else:
            node = stack.pop()
            res.append(node.val)
            root = node.right   
    return res



        
        