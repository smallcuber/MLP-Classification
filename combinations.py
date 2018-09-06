class Solution(object):
	def getCombine(self, n, k, index, prev, ans):
		if len(prev)==k:
			ans.append(prev)
			return
		for i in range(index,n-(k-len(prev))+2):
			self.getCombine(n, k, i+1, prev+[i], ans)

	def combine(self, n, k):
		ans = []
		if k <= n:
			self.getCombine(n, k, 1, [], ans)
		return ans