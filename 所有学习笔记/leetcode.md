##### [leetcode 2：两数相加](https://leetcode.cn/problems/add-two-numbers/)

```go
//该解法没有新建链表，而是修改较长的链表
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	s1, s2 := length(l1), length(l2)
	if s1 < s2 {		//将 l1 设置为较长链表
		l1, l2 = l2, l1
	}
	p, q := &ListNode{1, l1}, l2	//因为可能会在 l1 尾部新增节点，所以 p 不能指向当前节点，需要指向当前节点的前一个节点
	cur := 0						//进位
	for ; p.Next != nil; p = p.Next {
		t := p.Next.Val + cur
		if q != nil {
			t += q.Val
			q = q.Next
		}
		p.Next.Val = t%10
		cur = t/10
	}
	if cur == 1 {		//处理最后的进位
		p.Next = &ListNode{1, nil}
	}
	return l1
}
//求链表长度
func length(list *ListNode) int {
    ans := 0
	for p := list; p != nil; p = p.Next {
		ans++
	}
	return ans
}
```



##### [leetcode 3：无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

> 滑动窗口+hash

解法一：右指针前移，带动左指针前移

```go
func lengthOfLongestSubstring(s string) int {
    ans := 0
    left, right := 0, 0
    hash := map[byte]int{}		//这里的int代表字符出现的位置
    for right < len(s) {
        if _, ok := hash[s[right]]; ok {    //存在
            for left <= hash[s[right]] {	//将s[right]之前的全部清空
                delete(hash, s[left])
                left++
            }
        }
        hash[s[right]] = right
        right++
        ans = max(ans, right-left)
    }
    return ans
}
```

解法二：固定左指针位置，让右指针往前跑，然后左指针前移一位，再让右指针往前跑

```go
func lengthOfLongestSubstring(s string) int {
    m := map[byte]int{}			//这里的int代表字符出现的次数
    n := len(s)
    rk, ans := 0, 0  
    for i := 0; i < n; i++ {
        // 左指针向右移动一格，移除一个字符
        if i != 0 {
            delete(m, s[i-1])
        }
        // 不断地移动右指针
        for rk < n && m[s[rk]] == 0 {
            m[s[rk]]++
            rk++
        }
        ans = max(ans, rk - i)
    }
    return ans
}

//或者
func lengthOfLongestSubstring(s string) int {
    n := len(s)
    if n == 0  { return 0 }
    m := map[byte]int{}
    m[s[0]]++
    rk, ans := 1, 0
    for i := 0; i < n; i++ {
        for rk < n && m[s[rk]] == 0 {
            m[s[rk]]++
            rk++
        }
        ans = max(ans, rk - i)
        delete(m, s[i])
    }
    return ans
}
```

##### [leetcode 153：寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

注意：数组中无重复元素

```go
func findMin(nums []int) int {
    left, right, mid := 0, len(nums)-1, 0
    for left < right {
        mid = (right-left)>>1+left
        if nums[mid] > nums[right] {
            left = mid + 1
        }else{
            right = mid
        }
    }
    return nums[left]
}
```

##### [leetcode 154：寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

注意：数组中有重复元素

```go
//对数组最后一个元素x来讲，最小值左侧元素均>=x，最小值右侧元素均<=x，故以x作为参照做二分查找
//二分查找的本质在于舍去一部分数据，但需保证舍去后的数据仍满足上述条件，这应该就是二分查找时边界位置的确定准则
func minArray(numbers []int) int {
	low, high := 0, len(numbers)-1
	for low < high {
		mid := (high - low) >> 1 + low
		if numbers[mid] > numbers[high]{	//最小值左侧，故舍去mid及mid左侧的元素
			low = mid + 1
		}else if numbers[mid] < numbers[high] {		//最小值右侧，故舍去mid右侧的元素，但mid处需保留
			high = mid	//mid处元素留给high，让high仍然作为参照
		}else {
			high--	//不能确定在最小值左侧还是右侧，故不能盲目舍去数据，只能保守舍去右端点
		}
	}
	return numbers[low]
}
```

##### [leetcode 33：搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

注意：数组中不存在重复元素，返回元素索引，未找到返回-1

```go
//mid将数组一分为二，其中一边为有序，另一边为无序
func search(nums []int, target int) int {
    left, right, mid := 0, len(nums)-1, 0
    for left <= right {		//注意这里的等号，相对于前两题，已经确实存在值且在最终的left处，无需加等号，但这里不确定是否存在值，所以最终的left处也需要做判断
        mid = (right-left)>>1+left
        if nums[mid] == target {
            return mid
        }
        //mid左边是有序的，可以确定其上界为nums[mid]，下界为nums[0]
        if nums[left] <= nums[mid] {
            if target >= nums[left] && target < nums[mid] {
                right = mid-1
            }else{
                left = mid+1
            }
        }else{  //mid右边是有序的，其上界为nums[right]，下界为nums[mid]
            if target > nums[mid] && target <= nums[right] {
                left = mid+1
            }else{
                right = mid-1
            }
        }
    }
    return -1;
}
```

##### [leetcode 81：搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

注意：数组中有重复元素，若找到返回true，未找到返回false

```go
//在上一题的基础上增加一条判断，即nums[left] == nums[mid] == nums[right]，此时无法确定左右，只能left++，right--
func search(nums []int, target int) bool {
    left, right, mid := 0, len(nums)-1, 0
    for left <= right {		
        mid = (right-left)>>1+left
        if nums[mid] == target {
            return true
        }
        if nums[left] == nums[mid] && nums[mid] == nums[right] {
            left++
            right--
        }else if nums[left] <= nums[mid] {
            if target >= nums[left] && target < nums[mid] {
                right = mid-1
            }else{
                left = mid+1
            }
        }else{  
            if target > nums[mid] && target <= nums[right] {
                left = mid+1
            }else{
                right = mid-1
            }
        }
    }
    return false;
}
```

##### [leetcode 面试题 10.03. 搜索旋转数组](https://leetcode-cn.com/problems/search-rotate-array-lcci/)

注意：数组中存在重复元素，返回target在数组中的最小索引，不存在则返回-1

```go
func search(arr []int, target int) int {
    left, right, mid := 0, len(arr)-1, 0
    for left <= right {
        if arr[left] == target { return left }
        mid = (right-left) >> 1 + left
        if arr[mid] == target {			
            right = mid
        }else if arr[left] < arr[mid] {		//mid左边有序
            if target >= arr[left] && target < arr[mid] {
                right = mid-1
            }else{
                left = mid + 1
            }
        }else if arr[left] > arr[mid] {		//mid右边有序
            if target > arr[mid] && target <= arr[right] {
                left = mid + 1
            }else{
                right = mid - 1
            }
        }else{	
            left++
        }
    }
    return -1
}
```

##### leetcode 34：在排序数组中查找元素的第一个和最后一个位置

```go
//sort.Search()返回第一个满足条件的索引值，如果没有满足条件的，返回数组长度
func searchRange(nums []int, target int) []int {
    res := []int{-1, -1}
    one := sort.Search(len(nums), func(i int) bool { return nums[i] >= target})
	two := sort.Search(len(nums), func(i int) bool { return nums[i] >= target+1})
	if one < len(nums) && nums[one] == target {
        res[0], res[1] = one, two-1
    }
    return res
}
```

##### leetcode 128：最长连续序列

```go
//传统做法是选定一个值x，然后往前枚举：x+1，x+2...，但对每个数都枚举的话可能存在重复操作，比如遇到2，开始枚举3、4、5...，然后遇到1，又开始枚举：2、3、4、5...，一个大的优化就是，遇到一个数，先判断一下是否存在其先驱值，不存在的话再开始枚举
func longestConsecutive(nums []int) int {
	numsSet := make(map[int]bool)
	for _, num := range nums {	//填入哈希表并去重
		numsSet[num] = true
	}
	longest, current := 0, 0
	for num, _ := range numsSet {
		if !numsSet[num-1] {	//不存在前驱，执行枚举
			current = 1
			for numsSet[num+1] {	
				current++
				num++
			}
			if current > longest {
				longest = current
			}
			if longest >= len(numsSet) {	//小优化，这时长度已经最大了，直接退出
				return longest
			}
		}
	}
	return longest
}
```

##### [leetcode 11：盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```go
//首尾双指针，每次移动高度较小的那边，原因：移动高度较小的那边还有可能获取到更大的结果，而移动高度较大的那边只会得到更小的结果
func maxArea(height []int) int {
    res := 0
    for l, r := 0, len(height)-1; l < r; {
        res = max(res, min(height[l], height[r])*(r-l))
        if height[l] <= height[r] {
            l++
        }else{
            r--
        }
    }
    return res
}
```

##### leetcode 17：电话号码的字母组合

```go
//递归全排列
func letterCombinations(digits string) []string {
    if len(digit) == 0 { return nil }
    letters := []string{
		"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz",
	}
	res := []string{}
	str := make([]byte, len(digits))
	var recursion func(int) 
	recursion = func(p int) {
		if p == len(digits) {
			res = append(res, string(str))
			return
		}
		for _, v := range letters[int(digits[p]-'0'-2)] {
			str[p] = byte(v)
			recursion(p+1)
		}
	}
	recursion(0)
	return res
}
```

##### leetcode 31：下一个排列

```go
func nextPermutation(nums []int)  {
	size := len(nums)
	if size > 1 {
		i := size - 2 
		for i >= 0 && nums[i] >= nums[i+1] {	//找到一个较小值
			i--
		}
        if i >= 0 {
            j := size - 1
            for i < j && nums[i] >= nums[j] {	//找到一个比较小值略大的值
                j--
            }
            nums[i], nums[j] = nums[j], nums[i]		//两者交换
        }
		reverse(nums[i+1:])		//较小值后面的反转
	}
}
func reverse(nums []int) {
	for i, j := 0, len(nums)-1; i < j; i, j = i+1, j-1 {
		nums[i], nums[j] = nums[j], nums[i]
	}
}
```

##### leetcode 39：组合总和

```go
//给定数组中无重复元素，结果中一个元素可以多次使用
func combinationSum(candidates []int, target int) [][]int {
	res := [][]int{}
	size := len(candidates)
	nums := []int{}
	sort.Ints(candidates)	//排序，做一下剪枝
	var dfs func(int,int) 
	dfs = func(p, left int) {
		if left == 0 {
			res = append(res, append([]int{},nums...))
            return
		}
        if p == size { return }	//这个判断要放在 left==0 后面，否则可能会丢失结果
		if left >= candidates[p] {
            dfs(p+1, left) 	//不用当前值，但可能之前用过多次
		    nums = append(nums, candidates[p])
            dfs(p, left-candidates[p])	//使用当前值，注意这里不是p+1，p位置还要使用
            nums = nums[:len(nums)-1]	//回溯
		}
	}
	dfs(0, target)
	return res
}
```

##### [leetcode 40：组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

题目：给定数组中元素有重复，元素不能多次使用，且组合要唯一

题解一：自己写的，先排序，对于重复元素的处理方式为：假设有3个1，那这三个1分别使用0个、1个、2个、3个

```go
func combinationSum2(candidates []int, target int) [][]int {
	res := [][]int{}
	n := len(candidates)
	sort.Ints(candidates)
	var dfs func(int, int)
	nums := []int{}
	dfs = func(pos, left int) {
		if left == 0 {
			res = append(res, append([]int{},nums...))
			return
		}
		if left < 0 || pos == n || left < candidates[pos] { return }
		//找到重复元素个数
		i := 1
		for pos+i < n && candidates[pos+i] == candidates[pos] {
			i++
		}
		dfs(pos+i, left)	//直接跳过
		temp, tempLeft := nums, left
		for j := 1; j <= i; j++ {	//处理重复元素使用个数
			for k := 0; k < j; k++ {
				nums = append(nums, candidates[pos])
				left -= candidates[pos]
			}
			dfs(pos+i, left)
			nums, left = temp, tempLeft		//回溯
		}
	}
	dfs(0, target)
	return res
}
```

题解二：大佬写的，属于填坑法，将结果看成一个个的坑，然后每次填充一个，直至所有坑中元素的和为target，跟47题的填坑法类似

总结：填坑法特点：同一个循环来确定当前位置的不同情况，递归是为了确定下一个位置

```go
func combinationSum2(candidates []int, target int) [][]int {
	res := [][]int{}
	n := len(candidates)
	sort.Ints(candidates)
	var dfs func(int, int)
	nums := []int{}
	dfs = func(pos, left int) {
		if left == 0 {
			res = append(res, append([]int{},nums...))
			return
		}
		for i := pos; i < n && left >= candidates[i]; i++ {	//此循环是确定一个位置的不同填写方式，递归是确定下一个位置
			if i > pos && candidates[i] == candidates[i-1] {	//重复元素，不能重复发车
				continue
			}
			nums = append(nums, candidates[i])
			dfs(i+1, left-candidates[i])	//递归是确定下一个位置
			nums = nums[:len(nums)-1]
		}
	}
	dfs(0, target)
	return res
}
```

##### leetcode 62：不同路径

```go
func uniquePaths(m int, n int) int {
	dp := make([]int, n)	//一维dp数组就可以了
    for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i == 0 || j == 0 {	//边界
				dp[j] = 1
			}else{
				dp[j] += dp[j-1]
			}
		}
	}
	return dp[n-1]
}
```

leetcode 63：不同路径Ⅱ

```go
func uniquePathsWithObstacles(obstacleGrid [][]int) int {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	if obstacleGrid[0][0] == 1 || obstacleGrid[m-1][n-1] == 1 {	//首尾堵了，直接返回0
		return 0
	}
	dp := make([]int, n)	//一维dp数组即可
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if obstacleGrid[i][j] == 1 { 	//堵了，设为0
				dp[j] = 0
			}else{	//没堵
				if i == 0 && j == 0 {	
					dp[j] = 1
				}else if i == 0 {	//第一行等于前一个
					dp[j] = dp[j-1]
				}else if j > 0 {	//i>0，j>0
					dp[j] += dp[j-1]
				}
				//j==0时，第一列，dp[j] = dp[j]
			}
		}
	}
	return dp[n-1]
}
```

##### leetcode 55：跳跃游戏

题解一：自写版

```go
func canJump(nums []int) bool {
	size := len(nums)-1
	max := 0
	for i := 0; i < size; i++ {
        if max >= size {	//是否已经到达终点
            return true
        }
		if nums[i] != 0 {	//更新max
			if i + nums[i] > max {
				max = i + nums[i]
			}
		}else {		//判断是否可跳过0
			if max <= i { 
				return false 
			}
		}
	}
	return max >= size
}
```

题解二：官方版（哈哈，效率也一般啊~~~）

```go
func canJump(nums []int) bool {
	size := len(nums)
	max := 0
	for i := 0; i < size; i++ {
		if i <= max {
			if i + nums[i] > max {
				max = i + nums[i]
			}
			if max >= size-1 {
				return true
			}
		}
	}
	return false
}
```

##### [leetcode 95：不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

```go
func generateTrees(n int) []*TreeNode {
	if n == 0 { return nil }
	return dfs(1, n)
}
func dfs(start, end int) []*TreeNode {
	if start > end {
		return []*TreeNode{nil}
	}
	allTrees := []*TreeNode{}
	for i := start; i <= end; i++ {
		leftTrees := dfs(start, i-1)
		rightTrees := dfs(i+1, end)
		for _, left := range leftTrees {
			for _, right := range rightTrees {
				curTree := &TreeNode{i, nil, nil}
				curTree.Left = left
				curTree.Right = right
				allTrees = append(allTrees, curTree)
			}
		}
	}
	return allTrees
}
```

##### leetcode 96：不同的二叉搜索树

```go
func numTrees(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1] = 1, 1		//多申请一个空间，且dp[0] = 1，便于运算
	for i := 2; i <= n; i++ {
		for j := 1; j <= i; j++ {
			left := dp[j-1]		//左子树种类
			right := dp[i-j]	//右子树种类
			dp[i] += left * right	
		}
	}
	return dp[n]
}
```

##### leetcode 258：各位相加

```go
//100*x + 10 * y + z = 99*x + 9*y + x + y + z，之后还能拆，最后就是sum = num % 9，但要注意特例：num=0以及num为9的倍数
func addDigits(num int) int {
    if num < 10 {
        return num
    }
    if num % 9 == 0 {
        return 9
    }
    return num % 9
    //以上合为一条即为：return (num-1) % 9 + 1
}
```

##### leetcode 46：全排列

题解一：用一个hash表记录是否已经访问过

```go
func permute(nums []int) [][]int {
	res := [][]int{}
	size := len(nums)
	one := []int{}
    hash := map[int]bool{}
	for _, v := range nums {
		hash[v] = false
	}
	var dfs func(int)
	dfs = func(i int) {
		if i == size {
			res = append(res, append([]int{},one...))
			return
		}
		for _, v := range nums {
			if !hash[v] {	//剪枝
				hash[v] = true
				one = append(one, v)
				dfs(i+1)
				one = one[:len(one)-1]	//这两步为回溯
				hash[v] = false
			}
		}
	}
	dfs(0)
	return res
}
```

题解二：不使用hash标记，通过交换实现

```go
func permute(nums []int) [][]int {
	res := [][]int{}
	size := len(nums)
	var dfs func(int)
	dfs = func(pos int) {
		if pos == size {
			res = append(res, append([]int{}, nums...))
			return
		}
		for i := pos; i < size; i++ {
			nums[pos], nums[i] = nums[i], nums[pos]
			dfs(pos+1)
			nums[pos], nums[i] = nums[i], nums[pos]		//回溯
		}
	}	
	dfs(0)
	return res
}
```

##### [leetcode 47：全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

```go
//与上一题不同的是，本题中给定的nums中含有重复元素，要求输出不含有重复序列
//此解法思路为：将结果看成n个空格，每次往空格里面填充一个不重复的数字
func permuteUnique(nums []int) [][]int {
	res := [][]int{}
	size := len(nums)
	sort.Ints(nums)
	visited := make([]bool, size)	//记录是否已经访问
	perm := []int{}
	var dfs func(int)
	dfs = func(pos int) {
		if pos == size {
			res = append(res, append([]int{}, perm...))
			return
		}
		for i := 0; i < size; i++ {		
			//如果 已经访问过 或者 与其前置相同且其前置也被跳过了，则跳过此次循环
			if visited[i] ||  i > 0 && !visited[i-1] && nums[i] == nums[i-1] {
				continue
			}
			visited[i] = true
			perm = append(perm, nums[i])
			dfs(pos+1)
			visited[i] = false	//回溯
			perm = perm[:len(perm)-1]
		}
	}
	dfs(0)
	return res
}
```

##### leetcode 78：子集

```go
func subsets(nums []int) [][]int {
	res := [][]int{}
	size, one := len(nums), []int{}
	var dfs func(int)
	dfs = func(pos int) {
		if pos == size {
			res = append(res, append([]int{}, one...))
			return
		}
		dfs(pos+1)	//直接跳过
		one = append(one, nums[pos])
		dfs(pos+1)
		one = one[:len(one)-1]	//回溯
	}
	dfs(0)
	return res
}
```

##### leetcode 93：复原IP地址

```go
func restoreIpAddresses(s string) []string {
	res := []string{}
	size, nums := len(s), []byte{}
	var dfs func(int, int)
	dfs = func(pos, index int) {
        if pos > 4 { return }
		if index == size {
			if pos == 4 {
				res = append(res, string(nums[:len(nums)-1]))
			}
			return
		}
		nums = append(nums, s[index])	//用一个数字
		nums = append(nums, '.')
		dfs(pos+1, index+1)
		nums = nums[:len(nums)-2]	//回溯
		if s[index] != '0' {	//注意当为'0'时只能用'0'，不能再用两位或三位
			if index + 2 <= size {	//用两个数字
				nums = append(nums, s[index:index+2]...)
				nums = append(nums, '.')
				dfs(pos+1, index+2)
				nums = nums[:len(nums)-3]	//回溯
			}
			if index + 3 <= size && atoi(s[index:index+3]) <= 255 {	//用三个数字
				nums = append(nums, s[index:index+3]...)
				nums = append(nums, '.')
				dfs(pos+1, index+3)
				nums = nums[:len(nums)-4]	//回溯
			}
		}
	}
	dfs(0, 0)
	return res
}
//这里str只有三位
func atoi(str string) int {
	return int(str[0]-'0')*100 + int(str[1]-'0')*10 + int(str[2]-'0')
}
```

##### [leetcode 24： 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

题解一：两两交换节点值

```go
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	p, q := head, head.Next
	for {
		p.Val, q.Val = q.Val, p.Val
		p = q.Next
		if p == nil { break }
		q = p.Next
		if q == nil { break }
	}
	return head
}
```

题解二：调整节点指向

```go
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	newHead := &ListNode{0, head}	//搞个头节点
	tail := newHead	//尾指针
	for {
		one := tail.Next
		if one == nil { break }
		two := one.Next
		if two == nil { break }
		one.Next = two.Next
		two.Next = one
		tail.Next = two
		tail = one
	}
	return newHead.Next
}
```

##### [leetcode 38：外观数列](https://leetcode-cn.com/problems/count-and-say/)

```go
func countAndSay(n int) string {
	if n == 1 { return "1" }
	dp := []byte{'1'}
	for i := 2; i <= n; i++ {
		newDp := []byte{}
		count, char := 1, dp[0]
		for j := 1; j < len(dp); j++ {
			if dp[j] != char {	
				newDp = append(newDp, byte(count+'0'))
				newDp = append(newDp, char)
				count, char = 1, dp[j]
			}else{
				count++
			}
		}
		newDp = append(newDp, byte(count+'0'))
		newDp = append(newDp, char)
		dp = newDp
	}
	return string(dp)
}
```

##### [leetcode 15：三数之和](https://leetcode-cn.com/problems/3sum/)

题解：先排序，然后固定一位，双指针找另外两位，中间加上去重操作

```go
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
	res := [][]int{}
	size := len(nums)
	for i := 0; i < size; i++ {
        if nums[i] > 0 { break }	//大于0直接退出
        if i > 0 && nums[i] == nums[i-1] { continue }	//去重(第一位去重)
		for j, k := i+1, size-1; j < k;{
			if nums[j] + nums[k] + nums[i] == 0 {
				res = append(res, []int{nums[i], nums[j], nums[k]})
                 for j++; j < k && nums[j] == nums[j-1]; j++ {}	//去重(第二位去重，同时会带动第三位去重)
			}else if nums[j] + nums[k] + nums[i] < 0 {
				j++
			}else {
				k--
			}
		}
	}
	return res
}
```

##### [leetcode 16：最接近的三数之和](https://leetcode-cn.com/problems/3sum-closest/)

```go
func threeSumClosest(nums []int, target int) int {
	res := math.MinInt32
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {	//固定一位，双指针遍历两外两位
		for j, k := i+1, len(nums)-1; j < k;{
			sum := nums[i] + nums[j] + nums[k]
			if sum == target {
				return sum
			}else if sum < target {
				j++
			}else {
				k--
			}
			if abs(target - sum) < abs(res-target) {	//更新
				res = sum
			}
		}
	}
	return res
}
func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
```

##### [leetcode 56：合并区间](https://leetcode-cn.com/problems/merge-intervals/)

```go
func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {		//按左边界排序
		return intervals[i][0] < intervals[j][0]
	})
	res := [][]int{}
	res = append(res, intervals[0])
	p := 0
	for _, v := range intervals[1:] {
		if v[0] <= res[p][1] {	//判断左边界
			if v[1] > res[p][1] {	//判断右边界
				res[p][1] = v[1]
			}
		}else{
			res = append(res, v)
			p++
		}
	}
	return res
}
```

##### [lletcode 77：组合](https://leetcode-cn.com/problems/combinations/)

```go
func combine(n int, k int) [][]int {
	res := [][]int{}
	nums := []int{}
	var dfs func(int, int)
	dfs = func(cur, pos int) {
		if pos == k {
			res = append(res, append([]int{}, nums...))
			return
		}
		if n - cur + 1 >= k - pos {		//剪枝，剩余数目大于等于所需数目
			dfs(cur+1, pos)
			nums = append(nums, cur)
			dfs(cur+1, pos+1)
			nums = nums[:len(nums)-1]
		}
	}
	dfs(1, 0)
	return res
}
```

##### [leetcode 80：删除有序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)

```go
func removeDuplicates(nums []int) int {
	res := 0   //记录当前位置
	count := 1 //记录出现次数
	for i := 1; i < len(nums); i++ {
		if nums[i] != nums[res]{	//不相等
			res++
			nums[res] = nums[i]
			count = 1
		} else if count == 1 {	//相等且count = 1
			res++
            nums[res] = nums[i]
			count = 2
		}
	}
	return res+1
}
```

##### [leetcode 279：完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```go
//动态规划，状态转移方程：f(i) = min[f(i-j^2)] + 1，边界dp[0] = 0
func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 1; i <= n; i++ {
		minNum := n
		for j := 1; j*j <= i; j++ {
            if minNum > dp[i-j*j] {
                minNum = dp[i-j*j]
            }
		}
		dp[i] = minNum + 1
	}
	return dp[n]
}
```

##### [leetcode 82：删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

```go
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil || head.Next == nil { return head }
	newHead := &ListNode{}	//搞一个头结点，便于操作
	tail := newHead		//尾指针
	p, q := head, head.Next
	for p != nil && q != nil {
		if p.Val != q.Val {	//不相等
			tail.Next = p
			tail = tail.Next	//尾指针后移
			p, q = q, q.Next
		}else{	//相等
			//q往前找，直到不相同为止
			for q = q.Next; q != nil && q.Val == p.Val; q = q.Next {}
			if q == nil || q.Next == nil { 	//为空或者只剩一个节点，设置尾指针，退出循环
				tail.Next = q
				break
			}
			p, q = q, q.Next
		}
	}
	return newHead.Next
}
```

##### [leetcode 92：反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head.Next == nil || left == right { return head }
	newHead := &ListNode{	//头结点便于操作
		Next: head,
	}
	p := newHead
	for i := 1; i < left; i++ {
		p = p.Next	
	}
    //此时p为left前一个节点，用于之后的头插
	leftNode := p.Next	//left位置
	q := leftNode.Next
	//left+1到right头插法到left前面
	for i := left; i < right; i++{
		t := q.Next
		q.Next = p.Next
         p.Next = q
		q = t
	}
	leftNode.Next = q
	return newHead.Next
}
```

##### [leetcode 406：根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

```go
func reconstructQueue(people [][]int) [][]int {
	sort.Slice(people, func(i, j int) bool {	//第一位升序，第二位降序
		return people[i][0] < people[j][0] || (people[i][0] == people[j][0] && people[i][1] > people[j][1])
	})
	res := make([][]int, len(people))
	//res初始为空，将people中每一项放置到res中的第people[i][1]+1个空位上
	for _, p := range people {
		pos := p[1] + 1
		for i := range res {
			if res[i] == nil {
				pos--
				if pos == 0 {
					res[i] = p
					break
				}
			}
		}
	}
	return res
}
```

##### [leetcode 137：只出现一次的数字 II](https://leetcode-cn.com/problems/single-number-ii/)

```go
func singleNumber(nums []int) int {
    bitNum := [32]int{}
	for _, num := range nums {
		bitMask := 1
		for i := 31; i >= 0; i-- {
			if num & bitMask != 0 {
                bitNum[i] += 1
            }
			bitMask <<= 1
		}
	}
	var res, n int32 = 0, 1		//这里要设置为32位
	for i := 31; i >= 0; i-- {
		if bitNum[i] % 3 == 1 {
			res |= n	//使用 或 计算每一位，对于负数同样适用，加法只能用于正数
		}
		n <<= 1
	}
	return int(res)
}   
```

##### [leetcode 187：重复的DNA序列](https://leetcode-cn.com/problems/repeated-dna-sequences/)

```go
func findRepeatedDnaSequences(s string) []string {
    res := []string{}
    hash := make(map[string]int)
    for i := 0; i <= len(s)-10; i++ {
        sub := s[i:i+10]	//长度为10的滑动窗口
        hash[sub]++
        if hash[sub] == 2 {
            res = append(res, sub)
        }
    }
    return res
}
```

##### [leetcode 75：颜色分类](https://leetcode-cn.com/problems/sort-colors/)

题目：排序题，已经元素值为0、1、2三种

题解一：统计0、1、2出现次数，根据次数进行填充，算法略

题解二：单指针，记录当前填充位置，第一次填充0，第二次填充1
```go
func sortColors(nums []int)  {
	cur := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			nums[i], nums[cur] = nums[cur], nums[i]
			cur++
		}
	}
	for i := cur; i < len(nums); i++ {
		if nums[i] == 1 {
			nums[i], nums[cur] = nums[cur], nums[i]
			cur++
		}
	}
}
```
题解三：双指针，思路与题解二类似，采用双指针，分别记录0、1的位置，一次遍历
```go
func sortColors(nums []int)  {
    p0, p1 := 0, 0
	for i, v := range nums {
		if v == 0 {
			nums[i], nums[p0] = nums[p0], nums[i]
			if p0 < p1 {	//多执行一次交换
				nums[i], nums[p1] = nums[p1], nums[i]
			}
			p0++
			p1++
		}else if v == 1 {
			nums[i], nums[p1] = nums[p1], nums[i]
			p1++
		}
	}
}
```
题解四：双指针，分别记录0、2的位置，注意调整p2的位置时的操作
```go
func sortColors(nums []int)  {
    p0, p2 := 0, len(nums)-1
	for i := 0; i <= p2; i++ {
		for i <= p2 && nums[i] == 2 {	//循环交换
			nums[i], nums[p2] = nums[p2], nums[i]
			p2--
		}
		if nums[i] == 0 {
			nums[i], nums[p0] = nums[p0], nums[i]
			p0++
		}
	}
}
```

##### leetcode 198：打家劫舍，首尾可以一起偷

```go
func rob(nums []int) int {
    size := len(nums)
	switch size {
	case 1: return nums[0]
	case 2: return max(nums[0], nums[1])
	default:
		a, b := nums[0], max(nums[0], nums[1])	//a指向i-2，b指向i-1
		for i := 2; i < size; i++ {
			c := max(nums[i]+a, b)
			a, b = b, c
		}
		return b
	}
}
```

##### leetcode 213：打家劫舍，首尾不能一起偷

```go
//找出 偷首不偷尾、偷尾不偷首 这两种情况的最大值
func rob(nums []int) int {
    size := len(nums)
	switch size {
	case 1: return nums[0]
	case 2: return max(nums[0], nums[1])
	default:
		return max(rob_(nums[:size-1]), rob_(nums[1:]))
	}
}
func rob_(nums []int) int {
	a, b := nums[0], max(nums[0], nums[1])	//a指向i-2，b指向i-1
	for i := 2; i < len(nums); i++ {
		c := max(nums[i]+a, b)
		a, b = b, c
	}
	return b
}
```

##### [leetcode 36：有效的数独](https://leetcode-cn.com/problems/valid-sudoku/)

题解一：行、列、小九宫格分别进行遍历判断

```go
func isValidSudoku(board [][]byte) bool {	//行
	for i := 0; i < 9; i++ {
		hash := make(map[byte]bool)
		for j := 0; j < 9; j++ {
			if board[i][j] != '.' {
				if hash[board[i][j]] {
					return false
				}else{
					hash[board[i][j]] = true
				}
			}
		}
	}
	for i := 0; i < 9; i++ {	//列
		hash := make(map[byte]bool)
		for j := 0; j < 9; j++ {
			if board[j][i] != '.' {
				if hash[board[j][i]] {
					return false
				}else{
					hash[board[j][i]] = true
				}
			}
		}
	}
	for i := 0; i < 9; i = i+3 {	//小九宫格
		for j := 0; j < 9; j = j+3 {
            hash := make(map[byte]bool)
			for m := i; m < i+3; m++ {
				for n := j; n < j+3; n++ {
					if board[m][n] != '.' {
						if hash[board[m][n]] {
							return false
						}else{
							hash[board[m][n]] = true
						}
					}
				}
			}
		}
	}
	return true
}

```

题解二：只遍历一次

```go
func isValidSudoku(board [][]byte) bool {
	rows, cols := [9][9]int{}, [9][9]int{}	//分别记录行、列中数字出现次数
	subBoxs := [3][3][9]int{}	//记录小九宫格中数字出现次数
	for i, row := range board {
		for j, c := range row {
			if c == '.' { continue }
			index := c - '1'
			rows[i][index]++
			cols[j][index]++
			subBoxs[i/3][j/3][index]++
			if rows[i][index] > 1 || cols[j][index] > 1 || subBoxs[i/3][j/3][index] > 1 {	
				return false	//三者其一大于1则返回false
			}
		}
	}
	return true
}
```

##### [leetcode 57：插入区间](https://leetcode-cn.com/problems/insert-interval/)

```go
//采用一个一个将区间添加到结果中的方法
func insert(intervals [][]int, newInterval []int) [][]int {
	left, right := newInterval[0], newInterval[1]
	ans := [][]int{}
	merge := false	//标记是否已经将新区间加入
	for _, interval := range intervals {
		if interval[0] > right {	//区间在新曲间右侧
			if !merge {
				ans = append(ans, []int{left, right})	//新增的一定用left,right，不能用newInterval
				merge = true
			}
			ans = append(ans, interval)
		}else if interval[1] < left {	//区间在新曲间左侧
			ans = append(ans, interval)
		}else {		//该区间与新曲间有交叉，求并集
			left = min(left, interval[0])
			right = max(right, interval[1])
		}
	}
	if !merge {		//剩余一种特殊情况，新曲间在遍历结束后还未加入
		ans = append(ans, []int{left, right})
	}
	return ans
}
```

##### [leetcode 73：矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)

题解一：使用 m*n 的标记数组，标记数组是为了标记该元素 本来就为0 还是 被修改后的0

```go
func setZeroes(matrix [][]int)  {
    m, n := len(matrix), len(matrix[0])
	flag := make([][]bool, m)
	for i := 0; i < m; i++ {
		flag[i] = make([]bool, n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if matrix[i][j] == 0 && !flag[i][j] {	//本来就为0
				for r := 0; r < n; r++ {	//列置0
                    if matrix[i][r] != 0 {
                        matrix[i][r] = 0
					    flag[i][r] = true
                    }
				}
				for l := 0; l < m; l++ {	//行置0
                    if matrix[l][j] != 0 {
                        matrix[l][j] = 0
					    flag[l][j] = true
                    }
				}
			}
		}
	}
}
```

题解二：使用 m+n 的标记数组，用两个一维数组，分别标记该行、该列是否有0出现

```go
func setZeroes(matrix [][]int)  {
    m, n := len(matrix), len(matrix[0])
	rows, cols := make([]bool, m), make([]bool, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if matrix[i][j] == 0 {
				rows[i], cols[j] = true, true
			}
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if rows[i] || cols[j] {
				matrix[i][j] = 0
			}
		}
	}
}
```

题解三：不使用额外的空间，用原数组第一行和第一列做为标记数组，然后先提前用两个标记变量标记第一行和第一列是否有0，这样做的话会有个疑问，用原数组做标记会不会覆盖掉其中非0的元素，这里他们是相互联系的，如果不存在0，则标记数组非0不会动，如果存在0，到最后标记项处也得改为0，反正就是一条绳子上的蚂蚱

```go
func setZeroes(matrix [][]int)  {
    m, n := len(matrix), len(matrix[0])
	flagRow, flagCol := false, false
	for i := 0; i < n; i++ {	//第一行是否有0
		if matrix[0][i] == 0 {
			flagRow = true
		}
	}
	for i := 0; i < m; i++ {	//第一列是否有0
		if matrix[i][0] == 0 {
			flagCol = true
		}
	}
	for i := 1; i < m; i++ {	//做标记
		for j := 1; j < n; j++ {
			if matrix[i][j] == 0 {
				matrix[0][j] = 0
				matrix[i][0] = 0
			}
		}	
	}
	for i := 1; i < m; i++ {	//根据标记置0
		for j := 1; j < n; j++ {
			if  matrix[0][j] == 0 || matrix[i][0] == 0 {
				matrix[i][j] = 0
			}
		}	
	}
	if flagRow {
		for i := 0; i < n; i++ {	//第一行置为0
			matrix[0][i] = 0 
		}
	}
	if flagCol {
		for i := 0; i < m; i++ {	//第一列置为0
			matrix[i][0] = 0
		}
	}
}
```

##### [leetcode 90：子集 II](https://leetcode-cn.com/problems/subsets-ii/)

```go
func subsetsWithDup(nums []int) [][]int {
	res := [][]int{}
	size := len(nums)
	ans := []int{}
	sort.Ints(nums)
	var dfs func(int)
	dfs = func(pos int) {
		if pos == size {
			res = append(res, append([]int{}, ans...))
			return
		}
		i := pos + 1
		for i < size && nums[i] == nums[pos] {	//查找重复元素
			i++
		}
		for j := 0; j <= i-pos; j++ {	//重复元素分布使用0个、1个、2个...
			temp := ans
			for k := 0; k < j; k++ {
				ans = append(ans, nums[pos])
			}
			dfs(i)
			ans = temp
		}
	}
	dfs(0)
	return res
}
```

##### [leetcode 61：旋转链表](https://leetcode-cn.com/problems/rotate-list/)

```go
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 { return head }
	length := 1
	tail := head
	for ; tail.Next != nil; tail = tail.Next {	//计算链表长度，并得出尾节点
		length++
	}
	k = k % length
	if k > 0 {
		p := head
		for i := 0; i < length - k - 1; i++ {	//找到k-1个位置
			p = p.Next
		}
		q := p.Next		//取得k处
		p.Next = nil	//k-1置空
		tail.Next = head	//尾指针指向头节点
		head = q	//头节点变成k处
	}
	return head
}
```

##### [leetcode 71：简化路径](https://leetcode-cn.com/problems/simplify-path/)

```go
func simplifyPath(path string) string {
	splitStr := []string{}
	size := len(path)
	i := 1
    for i < size && path[i] == '/' {	//先预先去除多余的'/'，保证进入下面的for循环中path[i] != '/'
        i++
    }
	for i < size {	//只要能进入循环，说明path[i] != '/'
		r := i + 1
		for r < size && path[r] != '/' {
			r++
		}
		sub := path[i:r]
		if sub == ".." {
			if len(splitStr) > 0 {	//消去前面一个
				splitStr = splitStr[:len(splitStr)-1]
			}
		} else if sub != "." {	//正儿八经的路径，可以添加
			splitStr = append(splitStr, sub)
		}
		for r < size && path[r] == '/' {	
			r++
		}
		i = r	//将i置于下一个非'/'处
	}
    //合并路径
    if len(splitStr) == 0 { return "/" }	
	res := strings.Builder{}
	for _, v := range splitStr {
		res.WriteString("/")
		res.WriteString(v)
	}
	return res.String()
}
```

##### [leetcode 86：分隔链表](https://leetcode-cn.com/problems/partition-list/)

```go
//维护两个链表，一个存放小于x的节点。另一个存放大于等于x的节点，最后再合并一下即可
func partition(head *ListNode, x int) *ListNode {
	minNode := &ListNode{}	
	maxNode := &ListNode{}
	minTail, maxTail := minNode, maxNode
	for p := head; p != nil; p = p.Next {
		if p.Val < x {
			minTail.Next = p
			minTail = p
		}else{
			maxTail.Next = p
			maxTail = p
		}
	}
	maxTail.Next = nil
	minTail.Next = maxNode.Next
	return minNode.Next
}
```

##### [leetcode 143：重排链表](https://leetcode-cn.com/problems/reorder-list/)

```go
func reorderList(head *ListNode) {
    if head == nil || head.Next == nil { return }
	//寻找中间节点
	slow, fast := head, head
	for fast != nil { //循环结束后，对于1、2、3、4和1、2、3、4、5，slow均指向3
		fast = fast.Next
		if fast != nil {
			slow = slow.Next
			fast = fast.Next
		}
	}
	//将slow为首的链表反转
	newNode := &ListNode{0, slow}
	for p := slow.Next; p != nil; {
		slow.Next = p.Next
		p.Next = newNode.Next
		newNode.Next = p
		p = slow.Next
	}
	slow.Next = nil
	//两个链表交叉合并
	one, two := head, newNode.Next
	for one.Next != slow {
		p, q := one.Next, two.Next
		one.Next = two
		two.Next = p
		one, two = p, q
	}
	one.Next = two	//one链表最后一个指针
}
```

##### [leetcode 160：相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    A, B := headA, headB
	for A != B {
		if A == nil {
			A = headB
			B = B.Next
		}else if B == nil {
			B = headA
			A = A.Next
		}else {
			A = A.Next
			B = B.Next
		}
	}
	return A
}
```

##### [leetcode 461：汉明距离](https://leetcode-cn.com/problems/hamming-distance/)

```go
func hammingDistance(x int, y int) int {
    num := x ^ y
    count := 0
    for num != 0 {
        count++
        num = num & (num-1)
    }
    return count
}
```

##### [leetcode 543：二叉树的直径](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

```go
func diameterOfBinaryTree(root *TreeNode) int {
    res := 0
    var dfs func(*TreeNode) int	
    dfs = func(r *TreeNode) int {	//返回以r为根的树可以 对外 提供的直径，即左右子树中的较大值
        if r == nil { return 0 }
        left, right := dfs(r.Left), dfs(r.Right)
        if r.Left != nil { left += 1 }
        if r.Right != nil { right += 1 }
        res = max(res, left + right)
        return max(left, right)
    }
    dfs(root)
    return res
}
```

##### [leetcode 617：合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

```go
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
    if root1 == nil {
        return root2
    }
    if root2 == nil {
        return root1
    }
    root1.Left = mergeTrees(root1.Left, root2.Left)
    root1.Right = mergeTrees(root1.Right, root2.Right)
    root1.Val += root2.Val
	return root1
}
```

##### [leetcode 152：乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

```go
//动态规划特例：当前位置的最优解未必是由前一个位置的最优解转移得到的，乘积有正负
func maxProduct(nums []int) int {
	minF, maxF, ans := nums[0], nums[0], nums[0]
	for i := 1; i < len(nums); i++ {
		minTemp, maxTemp := minF, maxF
		maxF = max(minTemp*nums[i], max(maxTemp*nums[i], nums[i]))
		minF = min(maxTemp*nums[i], min(minTemp*nums[i], nums[i]))
		ans = max(ans, maxF)
	}
	return ans
}
```

##### [leetcode 101：对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```go
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var dfs func(*TreeNode, *TreeNode) bool
	dfs = func(left, right *TreeNode) bool {
		if left == nil && right == nil {
			return true
		}
		if left == nil || right == nil {
			return false
		}
		return left.Val == right.Val && dfs(left.Left, right.Right) && dfs(left.Right, right.Left)
	}
	return dfs(root.Left, root.Right)
}
```

##### [leetcode 739：每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

```go
func dailyTemperatures(temperatures []int) []int {
	size := len(temperatures)
	ans := make([]int, size)
	stack := []int{}	//单调栈，栈底到栈顶依次递减
	for i := 0; i < size; i++ {
         //将栈中较小的全部出栈，同时计算下标差
		for len(stack) != 0 && temperatures[i] > temperatures[stack[len(stack)-1]] {
             t := len(stack)-1
			ans[stack[t]] = i - stack[t]
			stack = stack[:t]
		}
		stack = append(stack, i)
	}
	return ans
}
```

##### [leetcode 142：环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

```go
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil {
		slow = slow.Next
		if fast.Next == nil {
			return nil
		}
		fast = fast.Next.Next
        if fast == slow {	//快慢指针相遇，取指针p为head，p和slow同步前进直至相遇，数学推导a=(n-1)(b+c)+c
			p := head
			for p != slow {
				p = p.Next
				slow = slow.Next
			}
			return p
		}
	}
	return nil
}
```

##### [leetcode 287：寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

题解一：原地哈希表

```go
func findDuplicate(nums []int) int {
    ans := 0
	for i := 0; i < len(nums); i++ {
        if nums[abs(nums[i])-1] < 0 {
            ans = abs(nums[i])
        }
		nums[abs(nums[i])-1] = -abs(nums[abs(nums[i])-1])
	}
	return ans
}
```

题解二：将该数组看成一个带环链表，问题转化为求环的入口

```go
func findDuplicate(nums []int) int {
    slow, fast := nums[0], nums[nums[0]]
	for slow != fast {
		slow = nums[slow]
		fast = nums[nums[fast]]
	}
	slow = 0
	for slow != fast {
		slow = nums[slow]
		fast = nums[fast]
	}
	return slow
}
```

##### [leetcode 49：字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

```go
//这道题主要是hash值的设置，将字符串排序作为hash值最稳妥，不存在意外冲突，但耗时
func groupAnagrams(strs []string) [][]string {
	hash := map[string][]string{}
	for _, str := range strs {
        temp := []byte(str)
        sort.Slice(temp, func(i, j int) bool {
            return temp[i] < temp[j]
        })
        a := string(temp)
        hash[a] = append(hash[a], str)
	}
	ans := make([][]string, 0, len(hash))
	for _, v := range hash {
		ans = append(ans, v)
	}
	return ans
}
//采用计数统计作为hash值
func groupAnagrams(strs []string) [][]string {
    mp := map[[26]int][]string{}
    for _, str := range strs {
        cnt := [26]int{}
        for _, b := range str {
            cnt[b-'a']++
        }
        mp[cnt] = append(mp[cnt], str)
    }
    ans := make([][]string, 0, len(mp))
    for _, v := range mp {
        ans = append(ans, v)
    }
    return ans
}
```

##### [leetcode 114：二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

题解一：先序遍历，结果存入数组，修改数组中各节点的指针

```go
func flatten(root *TreeNode)  {
	prePrder := []*TreeNode{}
	var dfs func(*TreeNode)
	dfs = func(tn *TreeNode) {
		if tn != nil {
			prePrder = append(prePrder, tn)
			dfs(tn.Left)
			dfs(tn.Right)
		}
	}
	dfs(root)
	for i := 0; i < len(prePrder)-1; i++ {
		prePrder[i].Left = nil
		prePrder[i].Right = prePrder[i+1]
	}
}
```

题解二：迭代进行先序遍历，先序遍历和修改指针同步进行，这里用到的迭代不是之前那种一直往左走并入栈

```go
func flatten(root *TreeNode)  {
	if root == nil { return }
	stack := []*TreeNode{root}
	var pre *TreeNode
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if pre != nil {
			pre.Left, pre.Right = nil, cur
		}
		left, right := cur.Left, cur.Right
		if right != nil {
			stack = append(stack, right)
		}
		if left != nil {
			stack = append(stack, left)
		}
		pre = cur
	}
}
```

题解三：找前驱节点

```go
func flatten(root *TreeNode) {
	cur := root
	for cur != nil {
		if cur.Left != nil {
			next := cur.Left	//保存左子树
             //找到左子树的最右节点
			pre := next
			for pre.Right != nil {
				pre = pre.Right
			}
			pre.Right = cur.Right	//最右节点右子树置为cur.Right
			cur.Left, cur.Right = nil, next		//修改cur左右指针
		}
		cur = cur.Right
	}
}
```

##### [leetcode 21：合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

题解一：非递归

```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	newNode := &ListNode{}
	tail := newNode
	for l1 != nil && l2 != nil {
		if l1.Val <= l2.Val {
			tail.Next = l1
			l1 = l1.Next
		}else {
			tail.Next = l2
			l2 = l2.Next
		}
		tail = tail.Next
	}
	if l1 != nil {
		tail.Next = l1
	}
	if l2 != nil {
		tail.Next = l2
	}
	return newNode.Next
}
```

题解二：递归

```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil { return l2 }
	if l2 == nil { return l1 }
	if l1.Val <= l2.Val {
		l1.Next = mergeTwoLists(l1.Next, l2)
		return l1
	}else{
		l2.Next = mergeTwoLists(l2.Next, l1)
		return l2
	}
}
```

##### [leetcode 148：排序链表](https://leetcode-cn.com/problems/sort-list/)

```go
func sortList(head *ListNode) *ListNode {
	if head == nil { return nil }
	length := 0
	for p := head; p != nil; p = p.Next {	//求链表长度
		length++
	}
	newNode := &ListNode{0, head}
	for subLen := 1; subLen < length; subLen <<= 1 {
		pre, cur := newNode, newNode.Next
		for cur != nil {	//链表未合并完
			head1 := cur	//第一个链表头节点
			for i := 1; i < subLen && cur != nil && cur.Next != nil; i++ {
				cur = cur.Next
			}
			head2 := cur.Next	//第二个链表头节点
			cur.Next = nil	//断开两个链表的连接
			cur = head2
			for i := 1; i < subLen && cur != nil && cur.Next != nil; i++ {
				cur = cur.Next
			}
			var next *ListNode
			if cur != nil {
				next = cur.Next
				cur.Next = nil
			}
			pre.Next = mergeTwoLists(head1, head2)
			for pre.Next != nil {	//pre指向合并后链表的尾部
				pre = pre.Next
			}
			cur = next
		}
	}
	return newNode.Next
}
```

##### [leetcode 98：验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

题解一：递归一

```go
func isValidBST(root *TreeNode) bool {
	if root == nil { return true }
    //找前驱，即左子树最右
	pre := root.Left
	for pre != nil && pre.Right != nil {
		pre = pre.Right
	}
	if pre != nil && pre.Val >= root.Val {
		return false
	}
    //找后驱，即右子树最左
	pre = root.Right
	for pre != nil && pre.Left != nil {
		pre = pre.Left
	}
	if pre != nil && pre.Val <= root.Val {
		return false
	}
	return isValidBST(root.Left) && isValidBST(root.Right)
}
```

题解二：递归二

```go
func isValidBST(root *TreeNode) bool {
	return helper(root, math.MinInt64, math.MaxInt64)
}
//判断自身以及左右子树中的节点值是否处于一定范围
func helper(root *TreeNode, lower, upper int) bool {
	if root == nil {
		return true
	}
	if root.Val <= lower || root.Val >= upper {
		return false
	}
	return helper(root.Left, lower, root.Val) && helper(root.Right, root.Val, upper)
}
```

题解三：非递归

```go
func isValidBST(root *TreeNode) bool {
	stack := []*TreeNode{}
	var pre *TreeNode
	for root != nil || len(stack) > 0 {
		//左走并入栈
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		//出栈并访问
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if pre != nil {
			if root.Val <= pre.Val {
				return false
			}
		}
		//右转
        pre = root
	    root = root.Right
	}
	return true
}

```

##### [leetcode 102：二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```go
func levelOrder(root *TreeNode) [][]int {
	if root == nil { return nil }
	queue := []*TreeNode{root}	
	ans, order := [][]int{}, []int{}
	curNum, nextNum := 1, 0	//当前层节点，下一层节点数
	for len(queue) > 0 {
        //出队并访问
		root = queue[0]
         queue = queue[1:]
		curNum--
         order = append(order, root.Val)
        //左右子树入队，同时nextNum++
		if root.Left != nil {
			nextNum++
			queue = append(queue, root.Left)
		}
		if root.Right != nil {
			nextNum++
			queue = append(queue, root.Right)
		}
        //更新curNum，nextNum，注意位置，一定放在左右子树都入队且nextNum++之后
		if curNum == 0 {
			ans = append(ans, order)
			order = []int{}
			curNum, nextNum = nextNum, 0
		}
	}
	return ans
}
```

##### [leetcode 79：单词搜索](https://leetcode-cn.com/problems/word-search/)

```go
func exist(board [][]byte, word string) bool {
	m, n, length := len(board), len(board[0]), len(word)
	visit := make([][]bool, m)
	for i := 0; i < m; i++ {
		visit[i] = make([]bool, n)
	}
	var dfs func(int, int, int) bool
	dfs = func(i, j, pos int) bool {
		if pos == length {
			return true
		}
		res := false
		if i >= 0 && i < m && j >= 0 && j < n && !visit[i][j] && board[i][j] == word[pos] {
			visit[i][j] = true
			res = dfs(i-1, j, pos+1) ||
				  dfs(i+1, j, pos+1) ||
				  dfs(i, j-1, pos+1) ||
				  dfs(i, j+1, pos+1)
			if !res {	//好像不加判断也可以
				visit[i][j] = false
			}
		}
		return res
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if dfs(i, j, 0) {
				return true
			}
		}
	}
	return false
}
```

##### [leetcode 238：除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

```go
func productExceptSelf(nums []int) []int {
    size := len(nums)
	ans := make([]int, size)
	//乘以左边
	left := 1
	for i := 0; i < size; i++ {
		ans[i] = left
		left *= nums[i]
	}
    //乘以右边
	right := 1
	for i := size-1; i >= 0; i-- {
		ans[i] *= right
		right *= nums[i]
	}
	return ans
}
```

##### [leetcode 538：把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

题解一：反向中序遍历，递归

```go
func convertBST(root *TreeNode) *TreeNode {
    sum := 0
    var dfs func(*TreeNode)
    dfs = func(root *TreeNode) {
        if root != nil {
            dfs(root.Right)
            sum += root.Val
            root.Val = sum
            dfs(root.Left)
        }
    }
    dfs(root)
    return root
}
```

题解二：非递归法，其实就是找右子树的最左边

```go
//暂时略过
```

##### [leetcode 146：LRU 缓存机制](https://leetcode-cn.com/problems/lru-cache/)

```go
//双向链表
type dLinkNode struct {
	Key, Value int
	Pre, Next  *dLinkNode
}
type LRUCache struct {
	cap  int
	hash map[int]*dLinkNode
	head *dLinkNode
}

func Constructor(capacity int) LRUCache {
	res := LRUCache{
		cap:  capacity,
		hash: make(map[int]*dLinkNode, capacity),
		head: &dLinkNode{0, 0, nil, nil},
	}
	res.head.Pre, res.head.Next = res.head, res.head
	return res
}
//将v插入到链表尾部
func (this *LRUCache) Insert(v *dLinkNode) {
	v.Pre = this.head.Pre  //修改v的Pre指向
	v.Next = this.head     //修改v的Next指向
	this.head.Pre.Next = v //修改尾节点的Next指向
	this.head.Pre = v      //修改head的Pre指向
}

func (this *LRUCache) Get(key int) int {
	if v, ok := this.hash[key]; ok {
        //将v节点拎出来放到链表尾部
		v.Pre.Next = v.Next //修改v的前置节点
		v.Next.Pre = v.Pre  //修改v的后置节点
		this.Insert(v)
		return v.Value
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	v := this.hash[key]
	if v == nil { //不存在
        //不存在就新建节点插入到链表尾部
		new := &dLinkNode{key, value, nil, nil}
		this.Insert(new)
		this.hash[key] = new
		if this.head.Value < this.cap { //未满
			this.head.Value++
		} else { //已满
            //删除头节点（经常访问的都放到了链表尾部，链表头部即为不经常访问的）
			p := this.head.Next
			p.Next.Pre = this.head
			this.head.Next = p.Next
			delete(this.hash, p.Key)
		}
	} else { //存在
        //修改节点值并将节点放到链表尾部
		v.Pre.Next = v.Next //修改v的前置节点
		v.Next.Pre = v.Pre  //修改v的后置节点
		v.Value = value
		this.Insert(v)
	}
}
```

##### [leetcode 347：前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

题解一：先用map统计次数，然后根据map生成数组，对该数组排序，取前k项

```go
func topKFrequent(nums []int, k int) []int {
    hash := make(map[int]int)
	for _, v := range nums {
		hash[v]++
	}
	temp := [][]int{}
	for k, v := range hash {
		temp = append(temp, []int{k,v})
	}
	sort.Slice(temp, func(i, j int) bool {
		return temp[i][1] > temp[j][1]
	})
	res := make([]int, k)
	for i := 0; i < k; i++ {
		res[i] = temp[i][0]
	}
	return res
}
```

题解二：同样根据map生成数组，然后采用最小堆（或最大堆），弹出根节点，堆调整，重复k次，此时堆中前k项即为结果

```go
//嗯~已经实现，但性能也一般
```

##### [leetcode 207：课程表](https://leetcode-cn.com/problems/course-schedule/)（拓扑排序）

题解一：深度优先遍历

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
	edges := make([][]int, numCourses) //存储节点的邻接节点
	visited := make([]int, numCourses) //有三种状态，0表示未访问，1表示访问中，2表示访问结束
    result := []int{}   //模拟栈，存储拓扑排序结果（逆序），注意：这个算法记录的不一定对，它会把未出现的课程也加进去
	valid := true
	var dfs func(int)
	dfs = func(u int) {
		visited[u] = 1               //修改状态为：访问中
		for _, v := range edges[u] { //遍历该节点的所有邻接节点
			if visited[v] == 0 { //未访问过则执行dfs，之后判断valid
				dfs(v)
				if !valid {
					return
				}
			} else if visited[v] == 1 {	//中间发现还有一个访问中，那就说明有环
				valid = false
				return
			}
		}
		visited[u] = 2 //修改状态为：访问结束
		result = append(result, u)	//记录拓扑序列
	}
	//获取每个节点的邻接节点
	for _, info := range prerequisites {
		edges[info[1]] = append(edges[info[1]], info[0])
	}
	//对于每个未经访问的节点，执行dfs
	for i := 0; i < numCourses && valid; i++ {
		if visited[i] == 0 {
			dfs(i)
		}
	}
	return valid
}
```

题解二：广度优先遍历

```go
func canFinish(numCourses int, prerequisites [][]int) bool {
    var (
        edges = make([][]int, numCourses)	//记录节点的邻接节点
        indeg = make([]int, numCourses)		//记录节点的入度
        result []int	//记录拓扑排序（正序）
    )
	//获取每个节点的邻接节点以及每个节点的入度数
    for _, info := range prerequisites {
        edges[info[1]] = append(edges[info[1]], info[0])
        indeg[info[0]]++
    }
    q := []int{}	//模拟队列，将入度为0的节点加入队列
    for i := 0; i < numCourses; i++ {
        if indeg[i] == 0 {
            q = append(q, i)
        }
    }
    for len(q) > 0 {
        u := q[0]	//访问
        q = q[1:]	//出队
        result = append(result, u)
        for _, v := range edges[u] {	//访问每一个邻接节点
            indeg[v]--	//每个邻接节点的入度数减一
            if indeg[v] == 0 {	//入度为0则加入队列
                q = append(q, v)
            }
        }
    }
    return len(result) == numCourses	//如果存在环，得到的拓扑序列数目会小于numCourses
}
```

##### [leetcode 560：和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

```go
//小->大三个点：0、i、j，题目要求满足sum(i,j)==k的个数，这里记前缀和sum(0,i)为pre1，记前缀和sum(0,j)为pre2，则sum(i,j) = pre2 - pre1
func subarraySum(nums []int, k int) int {
	count, pre := 0, 0
	hash := map[int]int{}	//存储前缀和
	hash[0] = 1				//提前加一个前缀和为0时的出现次数为1
	for i := 0; i < len(nums); i++ {
		pre += nums[i]		//当前前缀和
		count += hash[pre-k]	//当前前缀和pre - k = 历史前缀和，注意hash[x]，如果x不存在则返回0，存在则返回保存的值
		hash[pre] += 1
	}
	return count
}
```

##### [leetcode 162：寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

```go
//二分查找，假设一个有极大值的曲线，起点A，极大值B，终点C，记[A,B)为线段一，记[B,C]为线段二
func findPeakElement(nums []int) int {
    left, right, mid := 0, len(nums)-1, 0
    for left < right {
        mid = (right-left) >> 1 + left
        if nums[mid] > nums[mid+1] {	//落在了线段二
            right = mid
        }else{	//落在了线段一
            left = mid + 1
        }
    }
    return left 
}
```

##### [leetcode 1005：K 次取反后最大化的数组和](https://leetcode-cn.com/problems/maximize-sum-of-array-after-k-negations/)

```go
//贪心算法，每次选择最小值进行反转
func largestSumAfterKNegations(nums []int, k int) int {
    size := len(nums)
	for ; k > 0; k--{
		index, min := 0, nums[0]
		for i := 1; i < size; i++ {
			if nums[i] < min {
				index, min = i, nums[i]
			}
		}
        nums[index] = -min
       if min >= 0 {	//最小值大于等于0时，补充些操作，之后可以直接退出
            if (k-1)%2 != 0 {
                nums[index] = min
            }
            break
        }
	}
	sum := 0
	for i := 0; i < size; i++ {
		sum += nums[i]
	}
	return sum
}
```

##### [leetcode 200：岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

题解一：深度优先遍历，采用了额外空间记录是否已经访问

```go
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	var dfs func(int,int)
	dfs = func(row, col int) {
		if row >= 0 && row < m && col >= 0 && col < n && !visited[row][col] && grid[row][col] == '1' {
			visited[row][col] = true
			dfs(row-1,col)
			dfs(row+1,col)
			dfs(row,col-1)
			dfs(row,col+1)
		}
	}
	ans := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if !visited[i][j] && grid[i][j] == '1' {
				dfs(i, j)
				ans++
			}
		}
	}
	return ans
}
```

题解二：两个优化：1、不需要使用额外空间，将原数组中已访问过元素置为'0'；2、优化条件判断

```go
func numIslands(grid [][]byte) int {
    ans := 0
    m, n := len(grid), len(grid[0])
    var dfs func(int, int)
    dfs = func(i, j int) {
        grid[i][j] = '0'
        if i-1 >= 0 && grid[i-1][j] == '1' { dfs(i-1, j) }
        if i+1 < m && grid[i+1][j] == '1' { dfs(i+1, j) }
        if j-1 >= 0 && grid[i][j-1] == '1' { dfs(i, j-1) }
        if j+1 < n && grid[i][j+1] == '1' { dfs(i, j+1) }
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if grid[i][j] == '1'{
                dfs(i, j)
                ans++
            }
        }
    }
    return ans
}
```

##### [leetcode 394：字符串解码](https://leetcode-cn.com/problems/decode-string/)

```go
func decodeString(s string) string {
	stack := []byte{}
	for i := 0; i < len(s); i++ {
		if s[i] != ']' { //入栈
			stack = append(stack, s[i])
		} else { //出栈
			t := len(stack) - 1
            //找出[]中的字母
			temp := []byte{}
			for stack[t] != '[' {
				temp = append(temp, stack[t])
				stack = stack[:t]
				t--
			}
            //再出栈过滤掉'['
			stack = stack[:t]
			t--
            //找出数字（倍数）n
			n, c := 0, 1
			for t >= 0 && stack[t] >= '0' && stack[t] <= '9' {
				n += int(stack[t]-'0')*c
				c *= 10
				stack = stack[:t]
				t--
			}
            //将[]中的字母复制n份再入栈（记得转为逆序）
			for i := 0; i < n; i++ {
				for j := len(temp)-1; j >= 0; j-- {
					stack = append(stack, temp[j])
				}
			}
		}
	}
	return string(stack)
}

```

##### [leetcode 208：实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

```go
type Trie struct {
	children map[byte]*Trie		//存储子节点，也可用长度为26的数组代替map
	isEnd    bool				//存储此节点是否是某个单词的结尾处
}

func Constructor() Trie {
	return Trie{
		children: map[byte]*Trie{},
		isEnd:    true,
	}
}

func (this *Trie) Insert(word string) {
	a := this
	for _, c := range word {
		t := byte(c)
		if v, ok := a.children[t]; ok { //存在
			a = v
		} else { //不存在
			a.children[t] = &Trie{
				children: map[byte]*Trie{},
				isEnd: false,
			}
			a = a.children[t]
		}
	}
    a.isEnd = true	//最后一个节点置为true
}

func (this *Trie) Search(word string) bool {
	a := this
	for _, c := range word {
		if v, ok := a.children[byte(c)]; ok {
			a = v
		} else {
			return false
		}
	}
	return a.isEnd
}

func (this *Trie) StartsWith(prefix string) bool {
	a := this.children
	for _, c := range prefix {
		if v, ok := a[byte(c)]; ok {
			a = v.children
		} else {
			return false
		}
	}
	return true
}
```

##### [leetcode 139：单词拆分](https://leetcode-cn.com/problems/word-break/)

```go
//动态规划
func wordBreak(s string, wordDict []string) bool {
	size := len(s)
	wordMap := map[string]bool{}
	for _, v := range wordDict {
		wordMap[v] = true
	}
    dp := make([]bool,size+1)		//dp[n]标记s[0:n]是否可拆分
	dp[0] = true					//设置为true是为了使得边界条件也满足状态转移方程
    for i := 1; i <= size; i++ {	//表示s[0:i]
        for j := i-1; j >= 0; j-- {	//j将s[0:i]拆成两部分，s[0:j],s[j:i],前一部分由dp[j]决定，后一部分由s[j:i]是否在wordDict中决定，两者结合得出dp[i]
			suffix := s[j:i]
			if wordMap[suffix] && dp[j] {
				dp[i] = true
				break
			}
		}
	}
	return dp[size]
}
```

##### [leetcode 337：打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

```go
func rob(root *TreeNode) int {
	var dfs func(*TreeNode) (int,int)
	dfs = func(tn *TreeNode) (int, int) {
		if tn == nil { 
			return 0, 0
		}
		ly, ln := dfs(tn.Left)	//ly、ln分别表示偷了tn.Left的收益和不偷tn.Left的收益，同时也可以理解为tn.Left的收益和tn.Left孩子节点的收益
		ry, rn := dfs(tn.Right)	//同上
		return tn.Val + ln + rn, max(ly,ln)+max(ry,rn)
	}
	return max(dfs(root))
}
```

##### [leetcode 714：买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

题解一：动态规划

```go
func maxProfit(prices []int, fee int) int {
	n := len(prices)
	dp := make([][2]int, n)	//dp[i][0]、dp[i][1]分别表示第i天结束时手里没有股票的收益和手里还有股票时的收益
	dp[0][1] = -prices[0]
	for i := 1; i < n; i++ {
        dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i]-fee) //有两种因素会导致手里没有股票：1、上一天手里就没有股票，且本次不交易；2、上一天有股票，本次将其卖掉
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i]) //有两种因素会导致手里还有股票：1、上一天手里就有股票，且本次不交易；2、上一天没有股票，本次买入
	}
	return dp[n-1][0]
}

//改进，将dp数组改成两个变量存储就行
func maxProfit(prices []int, fee int) int {
	n := len(prices)
	sell, buy := 0, -prices[0]
	for i := 1; i < n; i++ {
        t := sell
		sell = max(sell,buy+prices[i]-fee)
		buy = max(buy, t-prices[i])	
	}
	return sell
}
```

题解二：贪心

```go
func maxProfit(prices []int, fee int) int {
	buy := prices[0] + fee //提前支付手续费
	profit := 0
	for i := 1; i < len(prices); i++ {
		if prices[i] + fee < buy {	//找到了一个价格更低的
			buy = prices[i] + fee
		}else if prices[i] > buy {	//执行交易，但无需支付手续费
			profit += prices[i]-buy
			buy = prices[i]		
		}
	}
	return profit
}
```

##### [leetcode 309：最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

```go
func maxProfit(prices []int) int {
    n := len(prices)
	if n == 0 { return 0 }
	//dp[i][0]: 手上持有股票时的最大收益
	//dp[i][1]: 手上不持有股票，且处于冷冻期的最大收益
	//dp[i][2]: 手上不持有股票，且未处于冷冻期的最大收益
    dp := make([][3]int, n)	
    dp[0][0] = -prices[0]
    for i := 1; i < n; i++ {
        dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i]) //状态一由状态一或状态三转移得来
		dp[i][1] = dp[i-1][0] + prices[i]	//状态二由状态一转移得来
		dp[i][2] = max(dp[i-1][1], dp[i-1][2])	//状态三由状态二、状态三转移得来
    }
    return max(dp[n-1][1],dp[n-1][2])
}
```

##### [leetcode 647：回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

```go
//选取一个中心，往两边扩展，该中心由一个字符或者两个字符组成
func countSubstrings(s string) int {
    ans := 0
    n := len(s)
    for i := 0; i < n; i++ {
        for j := 0; j <= 1; j++ {	//j = 0表示以一个字符为中心进行扩展，j = 1表示以两个字符为中心进行扩展
            l, r := i, i+j
            for l >= 0 && r < n && s[l] == s[r] {
                ans++
                l--
                r++
            }
        }
    }
    return ans
}
```

##### [leetcode 5：最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```go
//仍然选取上一题的中心扩展法
func longestPalindrome(s string) string {
    maxLen, n := 0, len(s)
    var res string
    for i := 0; i < n; i++ {
        for j := 0; j <= 1; j++ {
            l, r := i, i+j
            for l >= 0 && r < n && s[l] == s[r] {
                if r-l+1 > maxLen {
                    maxLen = r-l+1
                    res = s[l:r+1]
                }
                l--
                r++
            }
        }
    }
    return res
}
```

##### [leetcode 438：找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

题解一：滑动窗口

```go
//滑动窗口，注意，使用滑动窗口时一般需要单独设置初始窗口值，之后再开始循环
func findAnagrams(s string, p string) []int {
	slen, plen := len(s), len(p)
	if slen < plen {
		return nil
	}
	sCount, pCount := [26]int{}, [26]int{}
	res := []int{}
	for i := 0; i < plen; i++ { //初始窗口值
		sCount[s[i]-'a']++
		pCount[p[i]-'a']++
	}
	if sCount == pCount {
		res = append(res, 0)
	}
	for i := plen; i < slen; i++ {
		sCount[s[i-plen]-'a']-- //窗口左侧前移
		sCount[s[i]-'a']++      //窗口右侧前移
		if sCount == pCount {
			res = append(res, i-plen+1)
		}
	}
	return res
}
```

题解二：滑动窗口+双指针

```go
//说实话，没看太明白
func findAnagrams(s string, p string) []int {
	slen, plen := len(s), len(p)
	if slen < plen {
		return nil
	}
	sCount, pCount := [26]int{}, [26]int{}
	res := []int{}
	for i := 0; i < plen; i++ { //初始窗口值
		pCount[p[i]-'a']++
	}
	left, right := 0, 0
	for ;right < slen; right++ {
		curRight := s[right]-'a'
		sCount[curRight]++
		for sCount[curRight] > pCount[curRight] {
			curLeft := s[left]-'a'
			sCount[curLeft]--
			left++
		}
		if right-left+1 == plen {
			res = append(res, left)
		}
	}
	return res
}
```

##### 01背包问题初始版本

###### 测试样例

```go
	weights := []int{6, 3, 4, 5, 1, 2, 3, 5, 4, 2}	//重量/体积
	profits := []int{540, 200, 180, 350, 60, 150, 280, 450, 320, 120}	//收益
	vol := 30	//背包容量
	fmt.Println(myPackage2(weights, profits, vol))	//最大值为2410
```

###### 暴力解法

```go
//每种商品可以选或者不选，罗列所有情况(2^n)取其最大值
func myPackage(weights, profits []int, vol int) int {
	ans, size := 0, len(weights)
	var dfs func(int, int, int)
	dfs = func(i, money, left int) {
		if left == 0 || i == size {		//背包剩余空间为0或者已经没有可选择的商品
			if money > ans {
				ans = money
			}
		} else {
			if left-weights[i] >= 0 {
				dfs(i+1, money+profits[i], left-weights[i])	//选择第i件物品
			}
			dfs(i+1, money, left)	//不选择第i件物品
		}
	}
	dfs(0, 0, vol)
	return ans
}
```

###### 动态规划，二维数组

```go
func myPackage2(weights, profits []int, vol int) int {
	m, n := len(weights), vol	//行m代表商品的数量，列n代表背包的容量
    //初始化dp数组
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	//求解第一行（第一行表示此时只有第1件商品可以选择，只要背包容量大于等于该件商品的体积即可获得该件物品的收益）
	for i := 0; i < n; i++ {
		if i+1 >= weights[0] {	//背包容量大于第1件商品的体积
			dp[0][i] = profits[0]
		}
	}
	//求解剩余行
	for i := 1; i < m; i++ {
		for j := 0; j < n; j++ {
			if weights[i] > j+1 {	//第i件商品的体积大于背包容量，只能放弃该商品
				dp[i][j] = dp[i-1][j]
			}else if weights[i] == j+1 {	//背包刚好可以放下第i件商品，可以选择不放，或者放，取两者间的最大值
				dp[i][j] = max(dp[i-1][j], profits[i])
			}else {		//背包可以放下第i件商品，且有剩余空间，可以选择不放，或者放，取两者间的最大值，若选择放，则将剩余空间看成新的背包去放前i-1件商品，这里有个逆向思维，其实就是求出前i-1件放入背包的最大收益，再将第i件物品放入背包
				dp[i][j] = max(dp[i-1][j], profits[i]+dp[i-1][j-weights[i]])	
			}
		}
	}
	return dp[m-1][n-1]
}
```

###### 动态规划，一维数组

```go
func myPackage3(weights, profits []int, vol int) int {
	m, n := len(weights),vol
	dp := make([]int, n+1)
    //处理第一行
	for i := 1; i <= n; i++ {
		if i >= weights[0] {
			dp[i] = profits[0]
		}
	}
    //处理剩余行
	for i := 1; i < m; i++ {
		for j := n; j >= 0; j-- {	//逆序访问
            //共三种情况，背包放不下，背包刚好放下，背包放下后还有剩余
            //第一种情况，dp[j]不做改变，不用写
			if weights[i] <= j {	//包含了第二、三种情况，第二种情况是因为默认dp[0] = 0
				dp[j] = max(dp[j], profits[i]+dp[j-weights[i]])
			}
		}
	}
	return dp[n]
}
```

###### 01背包问题总结

- 何时使用：当元素有不选择(0)和选择(1)两者情况时，可以考虑01背包
- 处理步骤：初始化dp数组，特殊处理第一行，处理剩余行，有时第一行也无需特殊处理
- 01背包问题的空间优化问题：降低dp数组的维度，另外多申请一个空间，并填充合适的初始值，便于后续比较，此外降维后一定一定一定要逆序访问

##### [leetcode 416：分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

题解一：01背包，使用二维数组

```go
func canPartition(nums []int) bool {
	sum := 0
	for _, v := range nums {
		sum += v
	}
	if sum&1 == 1 {	//和为奇数直接返回false
		return false
	}
	m, n := len(nums), sum>>1
	dp := make([][]bool, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]bool, n)
	}
    //处理第一行，只需要将背包容量为nums[0]处置为true
	if nums[0] <= n {
		dp[0][nums[0]-1] = true
	}
    //处理剩余行
	for i := 1; i < m; i++ {
		for j := 0; j < n; j++ {
			if nums[i] > j+1 { //背包放不下
				dp[i][j] = dp[i-1][j]
			} else if nums[i] == j+1 { //背包刚好可以放下
				dp[i][j] = true
			} else { //背包可以放下，且有剩余空间
				dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
			}
		}
        if dp[i][n-1] == true {	//做一下剪枝
            return true
        }
	}
	return dp[m-1][n-1]
}
```

题解二：01背包，使用一维数组

```go
func canPartition(nums []int) bool {
    sum := 0
	for _, v := range nums {
		sum += v
	}
	if sum&1 == 1 {
		return false
	}
	m, n := len(nums), sum>>1
	dp := make([]bool, n+1)	//多申请了一个空间，且dp[0]置为true，方便操作，后续无需再加if判断
    dp[0] = true
    //模拟处理第一行
	if nums[0] <= n {
		dp[nums[0]] = true
	}
    //模拟处理剩余行
	for i := 1; i < m; i++ {
		for j := n; nums[i] <= j; j-- {	//用一维数组时要逆序访问，当前的dp[j]由之前的dp[j]和dp[j-nums[i]]决定，逆序访问可避免dp[j-nums[i]]被提前更新
			if dp[n] {	//剪枝
				return true
			}
			dp[j] = dp[j] || dp[j-nums[i]]
		}
	}
	return dp[n]
}

```

##### [leetcode 474：一和零](https://leetcode-cn.com/problems/ones-and-zeroes/)

```go
//该题背包有两个限制条件，不优化则需要采用三维数组，优化后使用二维数组即可，但注意优化后需要逆序访问
func findMaxForm(strs []string, m int, n int) int {
    //初始化dp数组
	dp := make([][]int, m+1)
	for i := 0; i <= m; i++ {
		dp[i] = make([]int, n+1)
	}
    //该题第一行不需要单独处理
	for i := 0; i < len(strs); i++ {
		zero, one = getZeroAndOne(strs[i])	
		for j := m; j >= 0; j-- {	//dp数组逆序访问，从右下角向左上角
			for k := n; k >= 0; k-- {
				if zero <= j && one <= k {
					dp[j][k] = max(dp[j][k], 1+dp[j-zero][k-one])
				}
			}
		}
	}
	return dp[m][n]
}

func getZeroAndOne(str string) (int,int) {
	zero, one := 0, 0
	for _, c := range str {
		switch c {
		case '0':
			zero++
		case '1':
			one++
		}
	}
	return zero, one
}

```

##### [leetcode 133：克隆图](https://leetcode-cn.com/problems/clone-graph/)

题解一：递归bfs

```go
func cloneGraph(node *Node) *Node {
	hash := map[int]*Node{}
	var bfs func(*Node) *Node
	bfs = func(n *Node) *Node {
        if n == nil { return nil }
		if hash[n.Val] == nil {
			new := &Node{n.Val, make([]*Node, len(n.Neighbors))}
			hash[n.Val] = new	//注意这句话放在下面循环后会导致递归死循环，相当于后面的递归没有了终止条件
			for i, v := range n.Neighbors {
				new.Neighbors[i] = bfs(v)
			}
		}
		return hash[n.Val]
	}
	return bfs(node)
}
```

题解二：队列bfs

```go
func cloneGraph(node *Node) *Node {
	if node == nil { return nil }
	hash := map[*Node]*Node{}
	queue := []*Node{}
	queue = append(queue, node)
	hash[node] = &Node{node.Val,[]*Node{}}	//存储第一个节点
	for len(queue) != 0 {
		n := queue[0]
		queue = queue[1:]
		for _, v := range n.Neighbors {
			if hash[v] == nil {
				hash[v] = &Node{v.Val,[]*Node{}}
				queue = append(queue, v)
			}
            hash[n].Neighbors = append(hash[n].Neighbors, hash[v])	//更新当前节点的Neighbors
		}
	}
	return hash[node]
}
```

##### [leetcode LCP 18. 早餐组合](https://leetcode-cn.com/problems/2vYnGI/)

```go
//双指针
func breakfastNumber(staple []int, drinks []int, x int) int {
	sort.Ints(staple)
	sort.Ints(drinks)
	res := 0
	for i, j := 0, len(drinks)-1; i < len(staple) && j >= 0; {
		if staple[i] + drinks[j] <= x {
			res += j+1
			res %= 1000000007
			i++
		}else{
			j--
		}
	}
	return res%(1000000007)
}
```
##### [leetcode 437：路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

题解一：递归

```go
//求出以root节点为根的路径和
func rootSum(root *TreeNode, targetSum int) int {
    if root == nil { return 0}
    res := 0
    if root.Val == targetSum { res++ }
    res += rootSum(root.Left, targetSum-root.Val)
    res += rootSum(root.Right, targetSum- root.Val)
    return res
}
//递归对树中每一个节点执行rootSum()求出所有路径和
func pathSum(root *TreeNode, targetSum int) int {
    if root == nil {
        return 0
    }
    res := 0
    res += rootSum(root, targetSum)
    res += pathSum(root.Left, targetSum)
    res += pathSum(root.Right, targetSum)
    return res
}
```
题解二：前缀和
```go
//有一条路径 root->A->B，结点A的前缀和为root->A所有结点值的和（不包括A），记为sumA，结点B的前缀和同理，记为sumB，如果 sumB+B.Val-sumA == targetSum，则路径A->B为一条可用路径，移项得 sumB+B.Val-targetSum == sumA，因此遍历到结点B时，只需要找到哈希表preSum中项为sumA的值
func pathSum(root *TreeNode, targetSum int) int {
    preSum := map[int]int{}
    preSum[0] = 1
    ans := 0
    var dfs func(*TreeNode, int)
    dfs = func(node *TreeNode, curSum int) {
        if node == nil { return }
        curSum += node.Val
        ans += preSum[curSum-targetSum]
        preSum[curSum]++
        dfs(node.Left, curSum)
        dfs(node.Right, curSum)
        preSum[curSum]--
    }
    dfs(root, 0)
    return ans
}
```

##### [leetcode 111：二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

```go
func minDepth(root *TreeNode) int {
    if root == nil { return 0 }	//这里要单独判断一次，因为不能dfs初始值不适合带0，即dfs(root, 0)
    ans := 100000
    var dfs func(*TreeNode, int) 
    dfs = func(node *TreeNode, depth int) {
        if node == nil { return }
        if depth >= ans { return }	//剪枝
        if node.Left == nil && node.Right == nil && depth < ans {
            ans = depth
        }
        dfs(node.Left, depth+1)
        dfs(node.Right, depth+1)
    }
    dfs(root, 1)
    return ans
}
```

##### [leetcode 290：单词规律](https://leetcode-cn.com/problems/word-pattern/)

```go
//双hash表，一个hash用于字符到字符串的映射，另一个用于字符串到字符的映射，两hash表的key和value分别交叉相等
func wordPattern(pattern string, s string) bool {
    word2ch := map[string]byte{}	//字符串到字符的映射
    ch2word := map[byte]string{}	//字符到字符串的映射
    words := strings.Split(s, " ")
    if len(pattern) != len(words) {
        return false
    }
    for i, word := range words {
        ch := pattern[i]
        //分别表示word在word2ch中，但值不为ch，以及ch在ch2word中，但值不为word两者情况
        if word2ch[word] > 0 && word2ch[word] != ch || ch2word[ch] != "" && ch2word[ch] != word {
            return false
        }
        word2ch[word] = ch
        ch2word[ch] = word
    }
    return true
}
```

##### [leetcode 199：二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

就是求层序遍历中每一层的最后一个节点

题解一：剑指offer提供，用cur、next分别记录当前层未打印节点数，下一层节点数

```go
func rightSideView(root *TreeNode) []int {
    if root == nil { return nil }
	res := []int{}
	queue := []*TreeNode{root}
	cur, next := 1, 0
	for len(queue) > 0 {
		root = queue[0]
		if root.Left != nil {
			queue = append(queue, root.Left)
			next++
		}
		if root.Right != nil {
			queue = append(queue, root.Right)
			next++
		}
		queue = queue[1:]
		cur--
		if cur == 0 {
			res = append(res, root.Val)
			cur, next = next, 0
		}
	}
	return res
}
```

题解二：可以省略cur、next两个变量，用队列长度代替

```go
func rightSideView(root *TreeNode) []int {
	if root == nil { return nil }
	res := []int{}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		size := len(queue)	//相当于当前层节点数
		for i := 0; i < size; i++ {
			root = queue[0]		//取队列第一个元素
			if root.Left != nil {
				queue = append(queue, root.Left)
			}
			if root.Right != nil {
				queue = append(queue, root.Right)
			}
			queue = queue[1:]	//出队
			if i == size-1 {	//当前层最后一个节点
				res = append(res, root.Val)
			}
		}
	}
	return res
}
```

##### [leetcode 223：矩形面积](https://leetcode-cn.com/problems/rectangle-area/)

```go
func computeArea(ax1 int, ay1 int, ax2 int, ay2 int, bx1 int, by1 int, bx2 int, by2 int) int {
    overlapX := min(ax2,bx2)-max(ax1,bx1)
    overlapY := min(ay2,by2)-max(ay1,by1)
    return (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-max(0,overlapX)*max(0,overlapY)
}
```

##### [leetcode 221：最大正方形](https://leetcode-cn.com/problems/maximal-square/)

```go
//动态规划，dp[i][j]表示以(i,j)为右下角的矩形，只包含1的正方形的边长最大值，当matrix(i,j)=0时，dp[i][j]=0，当matrix(i,j) =1时，dp[i][j]等于左边、上边、左上边三者最小值加1 
func maximalSquare(matrix [][]byte) int {
	m, n := len(matrix), len(matrix[0])
	//初始化dp数组
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	maxSide := 0
	//处理第一行
	for i := 0; i < n; i++ {
		if matrix[0][i] == '1' {
			dp[0][i] = 1
			maxSide = 1
		}
	}
	//处理第一列
	for i := 1; i < m; i++ {
		if matrix[i][0] == '1' {
			dp[i][0] = 1
			maxSide = 1
		}
	}
	//处理中间
	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			if matrix[i][j] == '1' {
				dp[i][j] = min(min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1])+1	//三者最小值加1
			}
			if dp[i][j] > maxSide {
				maxSide = dp[i][j]
			}
		}
	}
	return maxSide*maxSide
}
```

##### [leetcode 581：最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

题解一：暴力解法，排序并比较左右边界

```go
func findUnsortedSubarray(nums []int) int {
	temp := append([]int{}, nums...)
	sort.Ints(nums)
	left, right := 0, len(nums)-1
	for ;left <= right && temp[left] == nums[left]; left++ {}
	for ;right >= left && temp[right] == nums[right]; right-- {} 
	return right-left+1
}

```

题解二：一次遍历

```go
//从左往右遍历确定右边界，就是找到一个最靠后且小于其之后所有值的索引位置
//从右往左遍历确定左边界，就是找到一个最靠前且大于其之前所有值的索引位置
func findUnsortedSubarray(nums []int) int {
    n := len(nums)
    minn, maxn := math.MaxInt64, math.MinInt64
    left, right := -1, -1
    for i, num := range nums {
        if maxn > num {
            right = i
        } else {
            maxn = num
        }
        if minn < nums[n-i-1] {
            left = n - i - 1
        } else {
            minn = nums[n-i-1]
        }
    }
    if right == -1 {	//当数组原本就是升序时，right的值不会发生改变
        return 0
    }
    return right - left + 1
}
```

##### [leetcode 147：对链表进行插入排序](https://leetcode-cn.com/problems/insertion-sort-list/)

```go
func insertionSortList(head *ListNode) *ListNode {
    if head == nil { return nil }
    newHead := &ListNode{0,head}
    lastSortNode := head	//记录已排序部分的最后一个节点
    for p := head.Next; p != nil; {
        if lastSortNode.Val <= p.Val {	//无需再寻找插入点
            lastSortNode = p
            p = p.Next
        }else{
            q := newHead
            for ; p.Val > q.Next.Val; q = q.Next {}	//寻找插入点
            k := p.Next
            lastSortNode.Next = k
            p.Next = q.Next
            q.Next = p
            p = k
        }
    }
    return newHead.Next
}
```

##### [leetcode 剑指 Offer II 091. 粉刷房子](https://leetcode-cn.com/problems/JEj789/)

```go
func minCost(costs [][]int) int {
    //dp数组中三个值分别表示第i间房屋用1号颜色、2号颜色、3号颜色时的前i间房屋的最少成本
	dp := append([]int{}, costs[0]...)
	for i := 1; i < len(costs); i++ {
		temp := make([]int, 3)
		temp[0] = min(dp[1], dp[2]) + costs[i][0]
		temp[1] = min(dp[0], dp[2]) + costs[i][1]
		temp[2] = min(dp[0], dp[1]) + costs[i][2]
		dp = temp
	}
	return min(min(dp[0], dp[1]),dp[2])
}
```

##### [leetcode 91：解码方法](https://leetcode-cn.com/problems/decode-ways/)

题解一：dp数组

```go
func numDecodings(s string) int {
    size := len(s)
    dp := make([]int, size+1)
    dp[0] = 1
    for i := 1; i <= size; i++ {
       if s[i-1] != '0' {
           dp[i] += dp[i-1]
       }
       if i > 1 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
            dp[i] += dp[i-2]
       }
       if dp[i] == 0 { return 0 }	//可以提前终止
    }
    return dp[size]
}
```

题解二：只用几个变量

```go
func numDecodings(s string) int {
    a, b, c := 0, 1, 0	// a = f[i-2], b = f[i-1], c = f[i]
    for i := 1; i <= len(s); i++ {
        c = 0
        if s[i-1] != '0' {
            c += b
        }
        if i > 1 && s[i-2] != '0' && ((s[i-2]-'0')*10+(s[i-1]-'0') <= 26) {
            c += a
        }
        if c == 0 { return 0 }	//可以提前终止（出现了无法解码的0）
        a, b = b, c
    }
    return c
}

```

##### [leetcode 97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)

题目描述：判断 s3 == s1 + s2，这里的加号指交叉相加

题解：动态规划判断 s1的前 i 项 + s2的前 j 项 == s3的前 i+j 项

解法一：二维dp数组，且分别求第一行、第一列、其余行列，需要三个for循环

```go
func isInterleave(s1 string, s2 string, s3 string) bool {
    m, n, t := len(s1), len(s2), len(s3)
    if m + n != t { return false }
    dp := make([][]bool, m+1)
    for i := 0; i <= m; i++ {
        dp[i] = make([]bool, n+1)
    }
    dp[0][0] = true
    //第一行
    for j := 1; j <= n; j++ {
        dp[0][j] = dp[0][j-1] && s2[j-1] == s3[j-1]
    }
    //第一列
    for i := 1; i <= m; i++ {
        dp[i][0] = dp[i-1][0] && s1[i-1] == s3[i-1]
    }
    //其余行列
    for i := 1; i <= m; i++ {
        for j := 1; j <= n; j++ {
            dp[i][j] = (dp[i][j-1] && s2[j-1] == s3[i+j-1]) || (dp[i-1][j] && s1[i-1] == s3[i+j-1])
        }
    }
    return dp[m][n]
}
```

解法二：二维dp数组，一个for循环，内套2个if判断

```go
func isInterleave(s1 string, s2 string, s3 string) bool {
    m, n, t := len(s1), len(s2), len(s3)
    if m + n != t { return false }
    dp := make([][]bool, m+1)
    for i := 0; i <= m; i++ {
        dp[i] = make([]bool, n+1)
    }
    dp[0][0] = true
    for i := 0; i <= m; i++ {
        for j := 0; j <= n; j++ {
            if i > 0 {
                dp[i][j] = dp[i][j] || (dp[i-1][j] && s1[i-1] == s3[i+j-1])
            }
            if j > 0 {
                dp[i][j] = dp[i][j] || (dp[i][j-1] && s2[j-1] == s3[i+j-1])
            }
        }
    }
    return dp[m][n]
}
```

解法三：一维dp数组，滚动数组

```go
func isInterleave(s1 string, s2 string, s3 string) bool {
    m, n, t := len(s1), len(s2), len(s3)
    if m + n != t { return false }
    dp := make([]bool, n+1)
    dp[0] = true
    for i := 0; i <= m; i++ {
        for j := 0; j <= n; j++ {
            if i > 0 {		//行，用
                dp[j] = dp[j] && (s1[i-1] == s3[i+j-1])
            }
            if j > 0 {		//列，不用 || 用
                dp[j] = dp[j] || (dp[j-1] && s2[j-1] == s3[i+j-1])
            }
        }
    }
    return dp[n]
}
```

##### [leetcode 131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

题解：先用动态规划求出 s[i:j] 是否是回文子串，再用回溯法求出划分结果

```go
func partition(s string) [][]string {
    //先求出各种划分是否是回文子串
    n := len(s)
    dp := make([][]bool, n)
    for i := 0; i < n; i++ {
        dp[i] = make([]bool, n)
        for j := range dp[i] {
            dp[i][j] = true
        }
    }
    for i := n-1; i >= 0; i-- {
        for j := i+1; j < n; j++ {
            dp[i][j] = s[i] == s[j] && dp[i+1][j-1]
        }
    }
    //回溯法求出划分结果
    res, split := [][]string{}, []string{}
    var dfs func(int)
    dfs = func(i int) {
        if i == n {
            res = append(res, append([]string{}, split...))
            return
        }
        for j := i; j < n; j++ {
            if dp[i][j] {
                split = append(split, s[i:j+1])
                dfs(j+1)
                split = split[:len(split)-1]
            }
        }
    }
    dfs(0)
    return res
}
```

##### [leetcode 1464：数组中两元素的最大乘积](https://leetcode.cn/problems/maximum-product-of-two-elements-in-an-array/)

```go
//提供一种高效求出数组中的最大值和第二最大值的方法
func maxProduct(nums []int) int {
    max, max2 := 0, 0
    for _, v := range nums {
        if v > max {
            max2, max = max, v
        }else if v > max2 {
            max2 = v
        }
    }
    return (max-1)*(max2-1)
}
```

##### [leetcode 215：数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

解法一：二分思想 + 快排

```go
func findKthLargest(nums []int, k int) int {
    left, right, n := 0, len(nums)-1, len(nums)
    partition := func(left, right int) int {
        key := nums[left]
        for left < right {
            for left < right && nums[right] >= key { right-- }
            nums[left] = nums[right]
            for left < right && nums[left] <= key { left++ }
            nums[right] = nums[left]
        }
        nums[left] = key
        return left
    }
    index := 0
    for {
        index = partition(left, right)
        if index == n-k {
            break
        }else if index < n-k {
            left = index+1
        }else {
            right = index-1
        }
    }
    return nums[index]
}
```

解法二：大根堆

```go
func findKthLargest(nums []int, k int) int {
	size := len(nums)
    //建立大根堆（从最后一个非叶节点开始调整）
	for i := size>>1-1; i >= 0; i-- {
		_adjust(nums,i)
	}
	for i := 0; i < k-1; i++ {
        //将根顶元素换下去并删除
		nums[0] = nums[len(nums)-1]
		nums = nums[:len(nums)-1]
        //再重新调整
		_adjust(nums, 0)
	}
	return nums[0]
}
//堆调整
func _adjust(nums []int, pos int) {
	parent, child, root, length := 0, 0, nums[pos], len(nums)
	for parent = pos; (parent+1)*2 <= length; parent = child {
		child = parent*2 + 1	//左孩子
		if child + 1 < length && nums[child] < nums[child+1] {	//存在右孩子且右孩子大于左孩子
			child++		//切换到右孩子的位置
		}
		if root >= nums[child] {	//已经找到合适的位置
			break
		} else {
			nums[parent] = nums[child]	//将比根节点大的孩子节点提上来
		}
	}
	nums[parent] = root
}
```

##### [leetcode 241：为运算表达式设计优先级](https://leetcode.cn/problems/different-ways-to-add-parentheses/)

```go
func diffWaysToCompute(expression string) []int {
    res := []int{}
	is, num := isDigit(expression)
	if is {		//是数字直接返回
		res = append(res, num)
	}else {		//非数字，则看成 x op y，找到 op，左右划分，在将左右两边的值合并
		for i, c := range expression {
			if c == '+' || c == '-' || c == '*' {
				left := diffWaysToCompute(expression[:i])
				right := diffWaysToCompute(expression[i+1:])
				for _, l := range left {
					for _, r := range right {
						switch c {
						case '+':
							res = append(res, l+r)
						case '-':
							res = append(res, l-r)
						case '*':
							res = append(res, l*r)
						}
					}
				}
			}
		}
	}
	return res
}
//判断是否是数字
func isDigit(s string) (bool, int) {
	res, err := strconv.Atoi(s)
	if err != nil {
		return false, 0
	}
	return true, res
}
```

