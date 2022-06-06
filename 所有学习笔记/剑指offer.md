##### 剑指offer 03：数组中重复的数字

###### 题目一：找出数组中重复的数字

题目描述：数组长度为n，元素大小为0 ~ n - 1

解法一：额外定义一个数组做哈希表，遍历填充比较，时间：O(n)， 空间：O(n)， 代码略

解法二：数组自身作为标记数组

```go
func findRepeatNumber(nums []int) int {
    for _, v := range nums {
        t := abs(v)
        if nums[t] < 0 {
            return t
        }else{
            nums[t] = -nums[t]
        }
    }
    return 0
}
```

解法三：无需额外定义数组，通过数组内部比较交换得出结果，时间：O(n)， 空间：O(1)

```go
func findRepeatNumber(nums []int) int {
	for i := 0; i < len(nums); i++ {
        for i != nums[i] {	//此处虽然又是一层循环，但循环次数有限，O(1)级别
			if nums[i] == nums[nums[i]] {	//元素与其正确的位置上元素相等，说明重复
				return nums[i]
			}else {	//通过交换，将元素放在正确的位置上
				nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
			}
		}
	}
	return 0
}
```

###### 题目二：不修改数组找出重复的数字

题目描述：长度为n + 1的数组，元素大小为1 ~ n

解法：以1 ~ n 做二分查找，例n = 8，二分为1 ~ 4 和 5 ~ 8，对于1 ~ 4，在原数组中查找大小在1 ~ 4 区间内的元素个数count，若count > 4，说明此区间有重复元素，再继续二分判断。。。时间：O(logn)，空间：O(1)

```go
func getDuplication(nums []int) int {
	start, end := 1, len(nums) - 1
	for end >= start {
		middle := (end - start) >> 1 + start //这样求mid可以避免用加法带来的溢出问题
		count := countRange(nums, start, middle)	//求左半边
		if end == start {
			if count > 1 {
				return start
			}else {
				break
			}
		}
		if count > (middle - start + 1) {	
			end = middle
		}else {
			start = middle + 1
		}
	}
	return -1
}

func countRange(nums []int, start int, end int) int {
	count := 0
	for _, n := range nums {
		if n >= start && n <= end {
			count++
		}
	}
	return count
}
```

##### 剑指offer 04：二维数组的查找

解法：从右上角（或者左下角）开始比较，可以快速剔掉一行或者一列

```go
//以右上角为参照，右上角左边都比其小，右上角下边都比其大
func findNumberIn2DArray(matrix [][]int, target int) bool {
    if len(matrix) < 1 {
        return false
    }
    i, j := 0, len(matrix[0]) - 1
	for i < len(matrix) && j >= 0 {
		if matrix[i][j] == target {
			return true
		}else if matrix[i][j] > target {	//剔除一列
			j--
		}else {		//剔除一行
			i++
		}
	}
	return false
}
```

##### 剑指offer 05：替换空格

前提：假设在原字符串上进行替换，且保证输入字符串后面有足够空间

题解：若从前往后进行遍历替换，则有多次无效移动，时间为O(n2)，正确的解法是从后往前，时间O(n)

```go
func replaceSpace(s string) string {
	if len(s) == 0 {
		return s
	}
    //求出空格数量，计算字符串新长度
	blank := 0
	for _, v := range s {
		if v == ' ' {
			blank++
		}
	}
	newLength := len(s) + blank*2
	res := make([]byte, newLength)
	p1, p2 := len(s)-1, newLength-1	//双指针
	for p1 >= 0 {
		if s[p1] != ' ' {
			res[p2] = s[p1]
			p2--
			p1--
		} else {
			res[p2] = '0'
			p2--
			res[p2] = '2'
			p2--
			res[p2] = '%'
			p2--
			p1--
		}
	}
	return string(res)
}
```

##### 剑指offer 06：从尾到头打印链表

```go
func reversePrint(head *ListNode) []int {
    res := []int{}
	if head != nil {
		for head != nil {	//将链表值读取到数组中
			res = append(res, head.Val)
			head = head.Next
		}
		for i, j := 0, len(res) - 1; i < j; i, j = i+1, j-1 {	//反转数组
			res[i], res[j] = res[j], res[i]
		}
	}
	return res
}
```

##### 剑指offer 07：重建二叉树

题目描述：根据先序遍历和中序遍历构建二叉树

题解：①确定根节点root，为先序遍历第一个元素；②确定root在中序遍历中的位置，进行左右子树划分；③分别递归处理左右子树

```go
//自己写的，挺挫
func buildTree(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 {		//空结点
		return nil
	}
	root := new(TreeNode)
	root.Val = preorder[0]
	i := 0
	for i = 0; i < len(inorder); i++ {
		if inorder[i] == preorder[0] {	//划分左右子树
			break
		}
	}
	root.Left = buildTree(preorder[1:1+i], inorder[:i])		//求左子树
	root.Right = buildTree(preorder[1+i:], inorder[i+1:])	//求右子树
	return root
}
//大佬写的，很精简
func buildTree(preorder []int, inorder []int) *TreeNode {
	for k := range inorder {
		if inorder[k] == preorder[0] {
			return &TreeNode{
				Val: preorder[0],
				Left:  buildTree(preorder[1:k+1], inorder[0:k]),
				Right: buildTree(preorder[k+1:], inorder[k+1:]),
			}
		}
	}
	return nil
}
```

扩展：从这道题得出构建树的一般规律：

- 新建一个节点，节点赋值，确定左子树，确定右子树（不用考虑具体细节，这就够了）

- 然后递归，整棵树就构建完成

##### 剑指offer 08：二叉树的下一个节点

题目描述：二叉树节点中新增一个指向其父节点的指针，给定一棵二叉树和其中的一个节点，找到中序遍历中该节点的下一个节点

题解：分三种情形如下：

1、该节点右子树不为空，则其下一个节点为其右子树的最左边

2、该节点右子树为空，且为其父节点的左子节点，则其下一个节点即为其父节点

3、该节点右子树为空，且为其父节点的右子节点，沿着其父节点向上遍历，找到一个符合情形2的节点即为结果，若未找到，则为空。此处说法不太严谨，大致意思是：记该节点为T，其父节点为P，T右子树为空且T是P的右节点，说明以P为根的树已经访问完毕，接下来该访问位于P右侧的一棵树，且该棵树以P为左子节点

```go
func GetNext(target *TreeNode) *TreeNode {
	if target == nil {
		return nil
	}
	if target.Right != nil {	//情形1
		target = target.Right
		for target.Left != nil {
			target = target.Left
		}
		return target
	}
	for target.Parent != nil && target.Parent.Left != target {	//全部转化为情形2
		target = target.Parent
	}
	return target.Parent
}
```

##### 剑指offer 09：用两个栈实现队列

题解：栈用链表实现，使用链表头插法模拟入栈，链表指针前移模拟出栈，在golang中，由于指针简化以及gc机制，链表操作很简单

```go
//链表节点
type ListNode struct {
	Val  int
	Next *ListNode
}

type CQueue struct {
	left, right *ListNode
}

func Constructor() CQueue {
	return CQueue{
		left: nil,	//left用于入队
		right: nil,	//right用于出队
	}
}

//入队：left入栈
func (this *CQueue) AppendTail(value int) {
	this.left = &ListNode{	//链表头插法，模拟入栈
		value,
		this.left,
	}
}

//出队分三种情况：
//1、left和right均为空，返回-1
//2、right不为空，right出栈返回
//3、right为空但left不为空，left出栈然后入栈到right中，转化为情况2
func (this *CQueue) DeleteHead() int {
	if this.right == nil {	
		for this.left != nil{	//情况3-->情况2
			this.right = &ListNode{	//right入栈
				this.left.Val,
				this.right,
			}
			this.left = this.left.Next	//left出栈
		}
	}
	if this.right != nil {	//情况2
		v := this.right.Val
		this.right = this.right.Next  //right出栈
		return v
	}
	return -1	//情况1
}
```

扩展：使用两个队列实现栈：

​			入栈：队列1入队

​			出栈：队列1中除去最后一个元素，其余全部入队列2，之后队列1剩余的那个元素出队

##### 剑指offer 10：斐波那契数列

###### 题目一：斐波那契数列第n项

```go
func fib(n int) int {
	if n < 2 {
		return n
	}
	var i, j, k int = 0, 1, 0
	for m := 2; m <= n; m++ {
		k = (i + j) % (1e9 + 7)		//题目要求，取个模
		i, j = j, k
	}
	return k
}
```

###### 题目二：青蛙跳台阶问题

```go
func numWays(n int) int {	//f(0) = 1, f(1) = 1, f(n) = f(n-1) + f(n-2)
    if n < 2 {	
		return 1
	}
	var i, j, k int = 1, 1, 0
	for m := 2; m <= n; m++ {
		k = (i + j) % (1e9 + 7)		//题目要求，取个模
		i, j = j, k
	}
	return k
}
```

扩展：用8个2 *1小矩形填充2 * 8大矩形，可横着放可竖着放

​		  当填充大矩形最左边时，若小矩形竖着放，则剩余右边2 * 7的区域，此情况记为f(7)；

​		  若小矩形横着放，则下边必须再横着放一个小矩形，则剩余右边2 * 6的区域，此情况记为f(6)

​		  则f(8) = f(7) + f(6)

##### 剑指offer 11：旋转数组的最小数字

二分查找处理，对二分查找时的参照、要舍去的数据、边界位置要细细品味

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

扩展：对年龄进行排序，题目特征：元素大小有固定范围，可申请常数级辅助空间，时间为O(n)

```go
func sortAges(ages []int) {
	minAge, maxAge := 200, 0
	for _, v := range ages { //获取年龄最大值、最小值
		if v > maxAge { maxAge = v }
		if v < minAge { minAge = v }
	}
	ageTimes := make([]int, maxAge-minAge+1)
	for _, v := range ages {	//计算各年龄的次数
		v -= minAge
		ageTimes[v]++
	}
	for i, j := 0, 0; i < len(ageTimes); i++ {
		age := i + minAge
		for k := 0; k < ageTimes[i]; k++ {	//填充排序好的年龄数据
			ages[j] = age
			j++
		}
	}
}
```

##### 剑指offer 12：矩阵中的路径

```go
func exist(board [][]byte, word string) bool {
    m, n, sw := len(board), len(board[0]), len(word)
    //访问标记数组
    visited := make([][]bool, m) 
    for i := 0; i < m; i++ {
        visited[i] = make([]bool, n)
    }
    var dfs func(int, int, int) bool 
    dfs = func(i, j, p int) bool {
        if p == sw { return true }
        res := false
        if i >= 0 && i < m && j >= 0 && j < n && !visited[i][j] && board[i][j] == word[p] {
            visited[i][j] = true
            res = dfs(i+1, j, p+1) || dfs(i-1, j, p+1) || dfs(i, j+1, p+1) || dfs(i, j-1, p+1)
            visited[i][j] = false   //回溯
        }
        return res
    }
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if(dfs(i, j, 0)) {
                return true
            }
        }
    }
    return false
}

//另解，不用额外定义标记数组，对原矩阵做修改当成标记
var dfs func(int, int, int) bool 
dfs = func(i, j, p int) bool {
    if p == sw { return true }
    res := false
    if i >= 0 && i < m && j >= 0 && j < n {
        if board[i][j] == word[p] {
            board[i][j] = '1'	//随便找一个范围之外的字符
            res = dfs(i+1, j, p+1) || dfs(i-1, j, p+1) || dfs(i, j+1, p+1) || dfs(i, j-1, p+1)
            board[i][j] = word[p]	//回溯
        }
    }
    return res
}
```

##### 剑指offer 13：机器人的运动范围

题解一：上下左右都走一遍

```go
func movingCount(m int, n int, k int) int {
    visit := make([][]bool, m)
	for i := range visit {
		visit[i] = make([]bool, n)
	}
    count := 0
    var dfs func(int, int)
    dfs = func(i, j int) {
        if i >= 0 && i < m && j >= 0 && j < n && !visit[i][j] && sum(i)+sum(j) <= k {
            visit[i][j] = true
            count++
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)
        }
    }
    dfs(0, 0)
    return count
}

func sum(i int) int {
    sum := 0
    for i != 0 {
        sum += i % 10
        i /= 10
    }
    return sum
}
```

题解二：只需要走右和下两个方向

```go
//visited[i][j] = (visited[i-1][j] || visited[i][j-1]) && (sum(i) + sum(j) > k) 	从左边或者上边走来
func movingCount(m int, n int, k int) int {
    visited := make([][]bool, m)
    for i := 0; i < m; i++ {
        visited[i] = make([]bool, n)
    }
    count := 1
    visited[0][0] = true
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            if (i == 0 && j == 0) || (sum(i) + sum(j) > k) {
                continue
            }
            if i >= 1 {
                visited[i][j] = visited[i][j] || visited[i-1][j]
            }
            if j >= 1 {
                visited[i][j] = visited[i][j] || visited[i][j-1]
            }
            if visited[i][j] {
                count++
            }
        }
    }
    return count
}
//自己写的
func movingCount(m int, n int, k int) int {
    visited := make([][]bool, m)
    for i := 0; i < m; i++ {
        visited[i] = make([]bool, n)
    }
    count := 1
    visited[0][0] = true
    //处理第一行
    for j := 1; j < n && visited[0][j-1] && sum(j) <= k; j++ {
        count++
        visited[0][j] = true
    }
    //处理第一列
    for i := 1; i < m && visited[i-1][0] && sum(i) <= k; i++ {
        count++
        visited[i][0] = true
    }
    //处理其余
    for i := 1; i < m; i++ {
        for j := 1; j < n; j++ {
            visited[i][j] = (visited[i-1][j] || visited[i][j-1]) && (sum(i)+sum(j) <= k)
            if visited[i][j] {
                count++
            }
        }
    }
    return count
}
```

##### 剑指offer 14：剪绳子

动态规划与贪心算法的关系：如果要求某个问题的最优解， 且该问题可以分解为多个子问题，可以考虑动态规划，如果在分解子问题时存在某个特殊选择可得到最优解，可以考虑贪心算法。

可以应用动态规划解决问题的四个特征：

1、求问题的一个最优解

2、整体问题的最优解依赖于各个子问题的最优解

3、子问题之间还有相互重叠的更小的子问题

4、从上往下分析问题，从下往上求解问题（分析问题时写出递归式，解决问题时从小问题向大问题逐渐转化，即先求出f(1)，然后f(2)...f(n)，求解小问题有两种方法，一种是从左往右遍历，另一种是从右往左遍历，选择不会存在重复子问题的那种遍历）

```go
//题目要求：长度为n的绳子分成m段，m > 1, n > 1
//分析如下：
//1、划分子问题：n可以切分为[1,n-1]、[2,n-2]、[3,n-3]...
//2、求出每个子问题的最优解：max( k*f(n-k), k*(n-k) )，即n-k可以继续切，也可以不切，取两者间的最值
//3、挑选出所有子问题的解的最值：f(n) = max(所有子问题)= max( max(k*f(n-k), k*(n-k)) )
func cuttingRope(n int) int {
    dp := make([]int, n+1)
    dp[1] = 1	//f(1)=1
    for i := 1; i <= n; i++ {
        for j := 1; j <= i/2; j++ {		//计算一半即可
            dp[i] = max(dp[i], max(j*dp[i-j], j*(i-j)))
        }
    }
    return dp[n]
}
```

##### 剑指offer 15：二进制中1的个数

```go
//解法一：n右移
func numberOf1(n int) int {
	count := 0
	for n != 0 {	//当n为负数时，该解法会引起死循环，因为到最后n经过多次右移会变成全1
		if n & 1 == 1 {
			count++
		}
		n >>= 1		//go中是带符号右移，即右移时高位补符号位
	}
	return count
}

//解法二：标志位flag左移
func numberOf1(n int) int {
	count, flag := 0, 1
	for flag != 0 {		//需要循环的次数为：int的长度
		if n & flag != 0 {
			count++
		}
		flag <<= 1	//flag左移可分别判断n中的每一位是否为1
	}
	return count
}

//解法三：n&(n-1)
func numberOf1(n int) {
    count := 0
    for n != 0 {	//n中有几个1就循环几次
        count++
        n = (n - 1) & n		//每次会消去n中最右边的1
    }
    return count
}
```

扩展：1、判断整数n是否是2的整数次幂，return n & (n - 1) == 0。若n为2的整数次幂，则其中只有一个1

2、求需改变m中的几位二进制数可以将其转为n：m & n ,统计其中1的个数

##### 剑指offer 16：数值的整数次方

```go
func myPow(x float64, n int) float64 {	//这道题考察全面性以及二分法的运用
	if n == 0 { return 1 }
    if n < 0 { return 1/myPow(x, -n) }
	base := myPow(x, n>>1)	//细节一：右移代替除法
	res := base * base
	if n&1 == 1 {	//细节二：与1代替取余
		res *= x
	}
	return res
}
```

##### 剑指offer 17：打印从1到最大的n位数

注意：这道题没有表面上简单，最大的数可能超越了uint64的范围，故这道题为大数问题

解法一：可用字符数组表达大数，需要的解决的问题有两个：字符数组模拟加1，打印字符数组

```go
func print1ToMaxOfNDigits(n int) {
	if n < 1 {
		return
	}
	number := make([]byte, n)
	for i := 0; i < len(number); i++ {	//初始化为'0'
		number[i] = '0'
	}
	for !(Increment(number)) {
		printNumber(number)
	}
}
//模拟+1
func Increment(number []byte) bool { 
	carry := 0	//进位
	for i := len(number) - 1; i >= 0; i-- {
		if i == len(number) - 1 {	//最后一位数字加1
			number[i]++
		}
        //处理进位问题
		number[i] = byte(int(number[i]) + carry)
		if number[i] > '9' {
			number[i] = '0'
			carry = 1
		}else {
			carry = 0
		}
		if carry == 0 { 
			return false
		}
	}
	return carry == 1
}
//打印字符数组代表的数字
func printNumber(number []byte) { 
	index := 0
	for k, v := range number {
		if v != '0' {
			index = k
			break
		}
	}
    if index == 0 && number[index] == '0' {	 //去掉那些全为0的情况
		return
	}
	fmt.Println(string(number[index:]))
}
```

解法二：本题本质上是数字排列问题，即n个从0到9数字的全排列，递归实现

```go
func print1ToMaxOfNDigits(length int) {
	res := make([]byte, length)
	var recursion func(int)
	recursion = func(pos int) {
		if pos == length {
			printNumber(res)	//打印，见解法一
			return
		}
		for i := 0; i < 10; i++ {	//填充pos位置
			res[pos] = byte('0' + i)
			recursion(pos + 1)	//填充下一个位置
		}
	}
	recursion(0)	//从位置0开始
}
```

##### 剑指offer 18：删除链表的节点

题目一：给定单链表头指针和一个节点指针，O(1)时间内删除该节点

解法：将该节点的下一代节点的值复制过来，转为删除该节点的下一个节点，O(1)

```go
func deleteNode(head **ListNode,  dNode *ListNode) {	//因为可能改变头指针的值，所以要写成头指针的指针
	if *head == nil || dNode == nil { return }
	if *head == dNode {		//链表只有一个节点
		*head = nil
		return
	}	
	if dNode.Next != nil {	//删除的不是尾元素
		dNode.Val = dNode.Next.Val
		dNode.Next = dNode.Next.Next
	}else {		//删除的是尾元素
		pNode := *head
		for pNode.Next != dNode {
			pNode = pNode.Next
		}
		pNode.Next = nil
	}
}
```

题目二：删除链表中重复的节点

```go
func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil { return nil }
	value := head.Val
	p := head
	for p != nil && p.Next != nil {
		if p.Next.Val == value {
			p.Next = p.Next.Next	//可能引起p == nil
		}else {
			p = p.Next	//可能引起p.Next == nil
		}
		value = p.Val	//更新value
	}
	return head
}
```

##### 剑指offer 19：正则表达式匹配

```go
func isMatch(s string, p string) bool {	
	return matchCore(s, p, 0, 0)
}

func matchCore(s, p string, sPointer, pPointer int) bool {
	if sPointer == len(s) && pPointer == len(p) { //两者均空
		return true
	}
	if sPointer != len(s) && pPointer == len(p) { //s不空p空
		return false
	}
    //以下代码包含s和p均不空、s空p不空两种情况（比较时一定先判断s是否为空）
	//有第二个字符且为*
	if pPointer + 1 < len(p) && p[pPointer+1] == '*' {	
        if sPointer < len(s) && (p[pPointer] == s[sPointer] || p[pPointer] == '.') {	//匹配
			return matchCore(s, p, sPointer+1, pPointer+2) ||	//移动两位
				   matchCore(s, p, sPointer+1, pPointer) ||		//移动一位
				   matchCore(s, p, sPointer, pPointer+2)	//s不动，p移动两位，针对s空p不空的情况
		}else {	//不匹配，移动两位
			return matchCore(s, p, sPointer, pPointer+2)
		}
	}
	//没有第二个字符或者第二个字符不为*
	if sPointer < len(s) && (p[pPointer] == s[sPointer] || p[pPointer] == '.') {	//一定先判断s是否为空
		return matchCore(s, p, sPointer+1, pPointer+1)
	}
	return false
}
```

扩展：涉及到两个对象（字符串、数组、二叉树等）的比较判断等问题，一般规律：

​		   要考虑到四种情况：一空二不空、一不空二空、两者均空、两者均不空

##### 剑指offer 20：表示数值的字符串

```go
//字符串被.和e|E分成A、B、C三部分，其中A、C为有符号整数，B为无符号整数
func isNumber(s string) bool {
	if len(s) == 0 {
		return false
	}
    s = strings.Trim(s, " ")	//去除前后包含的空格
	a := 0
	var pointer *int = &a
	isNumber := scanInt(s, pointer)     //扫描A
	if *pointer < len(s) && s[*pointer] == '.' {
		*pointer++
		isNumber = scanUint(s, pointer) || isNumber  //扫描B，用||是因为小数点前后有没有整数均可 
	}
	if *pointer < len(s) && (s[*pointer] == 'e' || s[*pointer] == 'E') {
		*pointer++
		isNumber = isNumber && scanInt(s, pointer)  //扫描C，用&&是因为e|E前面要有数字，后面要有整数
	}
	return isNumber && len(s) == *pointer
}

//扫描无符号整数
func scanUint(s string, pointer *int) bool {
	if *pointer >= len(s) {
		return false
	}
	index := *pointer	//起始位置
	for *pointer < len(s) && s[*pointer] >= '0' && s[*pointer] <= '9' {
		*pointer++
	}
	return *pointer > index	//扫描后大于起始位置说明有数字
}
//扫描有符号整数
func scanInt(s string, pointer *int) bool {
	if *pointer >= len(s) {
		return false
	}
	if s[*pointer] == '+' || s[*pointer] == '-' {
		*pointer++
	}
	return scanUint(s, pointer)
}
```

##### 剑指offer 21：调整数组顺序使奇数位于偶数前面

注意：双指针遍历数组时，一定要判断移动后的指针是否越界！！！

```go
//双指针，分别指向奇数和偶数，然后交换
func exchange(nums []int) []int {
	odd := len(nums) - 1 //奇数指针
	even := 0            //偶数指针
	for even < odd {
        for odd >= 0 && nums[odd]&1 == 0 {  //寻找奇数（一定判断指针是否越界）
			odd--
		}
		for even < len(nums) && nums[even]&1 == 1 { //寻找偶数
			even++
		}
		if even < odd {
			nums[even], nums[odd] = nums[odd], nums[even]
		}
	}
	return nums
}
```

扩展：改变上述函数，使其不单单适用于奇数偶数交换，而适用于其他条件的交换

​		   方法：将判断标准抽取成一个函数，若交换条件改变，只需改变该函数即可

​		   这就是解耦的思想，提高代码重用性，便于扩展

```go
func exchange(nums []int) []int {
	begin, end := 0, len(nums) - 1
	for begin < end {
		if begin < len(nums) && judge(nums[begin]) {
			begin++
		}
		if end >= 0 && !judge(nums[end]) {
			end--
		}
		if begin < end {
			nums[begin], nums[end] = nums[end], nums[begin]
		}
	}
	return nums
}
func judge(num int) bool {	
	return num&1 == 1
}
```

##### 剑指offer 22：链表中倒数第k个节点

```go
func getKthFromEnd(head *ListNode, k int) *ListNode {
    if head == nil || k < 1{
		return nil
	}
	first, second := head, head
	for i := 0; i < k - 1; i++ {	//第一个指针前进k-1步
		if first.Next != nil {
			first = first.Next
		} else {	//对应k超出了链表长度的情况
			return nil
		}
	}
	for first.Next != nil {
		first = first.Next
		second = second.Next
	}
	return second
}   
```

##### 剑指offer 23：链表中环的入口节点

题解：

1、快慢指针确定是否有环

2、确定环中的节点个数k

3、快慢指针确定环的入口，快指针先走k步，然后双指针一起移动指针相遇

```go
//返回环中的一个节点，若没有环则返回nil
func meettingNode(head *ListNode) *ListNode {
	fast, slow := head, head
	for fast != nil {
		fast = fast.Next
		if fast != nil {
			fast = fast.Next
			slow = slow.Next
		}
		if fast == slow {
			break
		}
	}
	return fast
}
func EntryNodeOfLoop(head *ListNode) *ListNode {
	fast := meettingNode(head)
	if fast == nil {
		return nil
	}
	//获取环长度
	loopLength, slow := 1, fast.Next
	for fast != slow {
		slow = slow.Next
		loopLength++
	}
	//计算环入口
	fast, slow = head, head
	for loopLength > 0 {	//快指针先行loopLength步
		fast = fast.Next
		loopLength--
	}
	for fast != slow {		//快慢同行直至相遇
		fast = fast.Next
		slow = slow.Next
	}
	return fast
}
```

##### 剑指offer 24：反转链表

```go
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	p := head.Next	//取下头节点
	head.Next = nil	//头节点next置空
	for p != nil {
		q := p.Next
		p.Next = head
		head = p
		p = q
	}
	return head
}
```

##### 剑指offer 25：合并两个排序的链表

解法一：迭代

```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	head := &ListNode{}	//建立一个头节点，便于操作，后面返回head.Next
	tail := head	//头插法尾指针
	for l1 != nil && l2 != nil {
		if l1.Val <= l2.Val {
			tail.Next = l1
			tail = l1
			l1 = l1.Next
		} else {
			tail.Next = l2
			tail = l2
			l2 = l2.Next
		}
	}
	if l1 != nil {
		tail.Next = l1
	}
	if l2 != nil {
		tail.Next = l2
	}
	return head.Next
}
```

解法二：递归

```go
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2 
	}
	if l2 == nil {
		return l1
	}
	var head *ListNode
	if l1.Val < l2.Val {
		head = l1
		head.Next = mergeTwoLists(l1.Next, l2)
	}else {
		head = l2
		head.Next = mergeTwoLists(l1, l2.Next)
	}
	return head
}
```

##### 剑指offer 26：树的子结构

```go
//该题分三种情况：
//1、当两棵树根节点相同时，就需要判断A树是否包含B树，
//   这个包含的意思是B中与A中结构相同的位置需要一一对应相等
//2、若1中寻找失败，不能停止寻找，交给A的左子树和B进行比较
//3、若1、2均失败，还是不能停止寻找，交给A的右子树和B进行比较
func isSubStructure(A *TreeNode, B *TreeNode) bool {
	res := false
	if A != nil && B != nil {
		if A.Val == B.Val {
			res = hasTree(A, B)
		}
		if !res {
			res = isSubStructure(A.Left, B)
		}
		if !res {
			res = isSubStructure(A.Right, B)
		}
	}
	return res
}
func hasTree(A *TreeNode, B *TreeNode) bool {
	if B == nil {	//表示B中没有A对应位置的结构，仍然为true
		return true
	}
	if A == nil {	//表示B中多出来了A没有的结构，为false
		return false
	}
	if A.Val != B.Val {
		return false
	}
	return hasTree(A.Left, B.Left) && hasTree(A.Right, B.Right)
}
```

精简版代码

```go
func isSubStructure(A *TreeNode, B *TreeNode) bool {
	return (A != nil && B != nil) && (hasTree(A, B) || isSubStructure(A.Left, B) || isSubStructure(A.Right, B))
}
func hasTree(A *TreeNode, B *TreeNode) bool {
	if B == nil { return true } 
	if A == nil || A.Val != B.Val{ return false }
	return hasTree(A.Left, B.Left) && hasTree(A.Right, B.Right)
}
```

扩展：二叉树相关题目的递归解法思路及步骤

```go
/**
1、空指针判断
2、根节点行为（该行为可能一条语句就能完成，也可能需要另写一个函数完成，一般该函数可能也是递归且符合这5点）
3、根节点的左子树行为	（此处涉及到二叉树的三种遍历，根据实际情况调整2-4的顺序）
4、根节点的右子树行为
5、整合结果，可能是2-4中三个结果取或，也可能是相与
/
```

##### 剑指offer 27：二叉树的镜像

```go
func mirrorTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	root.Left, root.Right = mirrorTree(root.Right), mirrorTree(root.Left)	//后序遍历
	return root
}
```

##### 剑指offer 28：对称的二叉树

题解：对二叉树进行 中左右遍历 、中右左遍历，判断两次遍历序列是否一致

```go
func isSymmetric(root *TreeNode) bool {	
	return recursion(root, root)
}
//方式一
func recursion(root1, root2 *TreeNode) bool {
	if root1 == nil && root2 == nil {	//两者均不空
		return true
	}
	if root1 == nil || root2 == nil {	//两者中有一个为空
		return false
	}
    //两者均不空
	return root1.Val == root2.Val && recursion(root1.Left, root2.Right) && recursion(root1.Right, root2.Left)
}
//方式二
func recursion(root1, root2 *TreeNode) bool {
	if root1 != nil && root2 != nil {	//两者均不空的情况
		return root1.Val == root2.Val && recursion(root1.Left, root2.Right) && recursion(root1.Right,root2.Left)
	}
	return root1 == root2	//其余三种情况
}
```

##### 面试题29：顺时针打印矩阵

题解：一组变量控制剩余行、列数，即边界条件，另一组变量记录待访问位置

​		   将此种思想延伸扩展到其他类似题目

```go
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}
	m, n := len(matrix), len(matrix[0])	//记录剩余行、列数
	row, col := 0, -1	//记录待访问位置，初始时在入口左边一个位置
	res := make([]int, m*n)
	k := len(res)
	for i := 0; i < k; {
		//右走n步
		for j := 0; j < n && i < k; j++ {	//转圈可能随时停止，所以每次循环都要判断是否已经访问完 
			col++
			res[i] = matrix[row][col]
			i++
		}
		m--		//行数减一
		//下走m步
		for j := 0; j < m && i < k; j++ {
			row++
			res[i] = matrix[row][col]
			i++
		}
		n--		//列数减一
		//左走n步
		for j := 0; j < n && i < k; j++ {
			col--
			res[i] = matrix[row][col]
			i++
		}
		m--		//行数减一
		//上走m步
		for j := 0; j < m && i < k; j++ {
			row--
			res[i] = matrix[row][col]
			i++
		}
		n--		//列数减一
	}
	return res
}
```

##### 剑指offer 30：包含min函数的栈

题解：用链表模拟栈

```go
type Node struct {
	val, min int
	next *Node
}

type MinStack struct {
	head *Node
}

/** initialize your data structure here. */
func Constructor() MinStack {
	return MinStack{}
}

//头插法建立链表模拟栈
func (this *MinStack) Push(x int)  {
	if this.head == nil {	//栈为空
		this.head = &Node{
			val: x,
			min: x,
			next: nil,
		}
	}else {		//栈不空
		this.head = &Node{
			val: x,
			next: this.head,
            min: this.head.min,
		}
		if x < this.head.min {
			this.head.min = x
		}
	}
}

func (this *MinStack) Pop()  {
	this.head = this.head.next
}
func (this *MinStack) Top() int {
	return this.head.val
}
func (this *MinStack) Min() int {
	return this.head.min
}
```

##### 剑指offer 31：栈的压入、弹出序列

```go
func validateStackSequences(pushed []int, popped []int) bool {
	length := len(popped)
	if length == 0 {
		return true
	}
	stack, top, i := make([]int, length), -1, 0
	for _, num := range pushed {
		top++	//入栈
		stack[top] = num
		for top >= 0 && stack[top] == popped[i] {
			i++
			top--	//出栈
		}
	}
	return top == -1
}
```

##### 剑指offer 32：从上到下打印二叉树

###### 题目一：不分行从上到下打印二叉树（树的层序遍历）

```go
func levelOrder(root *TreeNode) []int {
    res := []int{}
    if root != nil {
        queue := []*TreeNode{root}	//根节点入队
        for len(queue) != 0 {
            root = queue[0]	
            res = append(res, root.Val)	
            queue = queue[1:]	//出队
            if root.Left != nil {	//左子树入队
                queue = append(queue, root.Left)
            }
            if root.Right != nil {	//右子树入队
                queue = append(queue, root.Right)
            }
        }
    }
    return res
}
```

###### 题目二：分行从上到下打印二叉树

题解：用一个变量记录当前层未打印的节点数，用另一个变量记录下层节点数

```go
func levelOrder(root *TreeNode) [][]int {
    res := [][]int{}
    if root != nil {
        queue := []*TreeNode{root}
        cur, next := 1, 0	//当前层未打印，下一层节点数
        level := []int{}
        for len(queue) != 0 {
            root = queue[0]
            level = append(level, root.Val)	
            queue = queue[1:]	//出队
            cur--				//未打印数减一
            if root.Left != nil {	//左子树入队
                queue = append(queue, root.Left)
                next++				//下一层节点数加一
            }
            if root.Right != nil {	//右子树入队
                queue = append(queue, root.Right)
                next++				//下一层节点数加一
            }
            if cur == 0 {			//当前层已经打印结束
                res = append(res, level)
                level = []int{}
                cur, next = next, 0
            }
        }
    }
    return res
}
```

###### 题目三：之字形打印二叉树

解法一：将题目二中的结果按照层数奇偶翻转一下即可

解法二：用栈，但是要用到两个栈，当前层一个，下一层一个，否则会数据覆盖

​			   用current, next实现两个栈的交替出栈入栈

​			   入栈顺序也要交替，一个先左后右，一个先右后左

```go
type myStack struct {
	Val  *TreeNode
	Next *myStack
}

func levelOrder(root *TreeNode) [][]int {
	if root == nil {
		return [][]int{}
	}
	res, num := [][]int{}, []int{}
	levelStack := [2]*myStack{}
	current, next := 0, 1	//实现两个栈的交替出栈入栈
	push(&levelStack[current], root)
	for levelStack[0] != nil || levelStack[1] != nil {
		root = levelStack[current].Val
		levelStack[current] = levelStack[current].Next
		num = append(num, root.Val)
		if current == 0 {	//先入左，再入右
			if root.Left != nil {
				push(&levelStack[next], root.Left)
			}
			if root.Right != nil {
				push(&levelStack[next], root.Right)
			}
		}else {		//先入右，再入左
			if root.Right != nil {
				push(&levelStack[next], root.Right)
			}
			if root.Left != nil {
				push(&levelStack[next], root.Left)
			}
		}
		if levelStack[current] == nil {
			res = append(res, num)
			num = []int{}
			current = 1 - current
			next = 1 - next
		}
	}
	return res
}

func  push(head **myStack, node *TreeNode) {
	*head = &myStack{
		Val:  node,
		Next: *head,
	}
}
```

##### 剑指offer 33：二叉搜索树的后序遍历序列

题解一：切片最后一个元素为根节点root，从后往前遍历找到第一个小于root的元素位置index，则index左边的元素应该都小于root

```go
func verifyPostorder(postorder []int) bool {
	length := len(postorder)
	if length < 2 {
		return true
	}
	index, root := length - 2, postorder[length - 1]
	for ; index >= 0 && postorder[index] > root; index-- {}	//找到第一个小于root的元素位置index
	for i := 0; i < index; i++ {	//判断index之前的元素是否都小于root
		if postorder[i] > root {
			return false
		}
	}
	return verifyPostorder(postorder[:index+1]) && verifyPostorder(postorder[index+1:length-1])
}
```

题解二：从前往后遍历

```go
func verifyPostorder(postorder []int) bool {
    size := len(postorder)-1
    if size < 1 {
        return true
    }
    i := 0
    for ; i < size && postorder[i] < postorder[size]; i++ {}	//找到第一个大于尾元素的元素位置
    j := i+1
    for ; j < size && postorder[j] > postorder[size]; j++ {}	//判断剩余元素是否都大于尾元素
    return  j >= size && verifyPostorder(postorder[:i]) && verifyPostorder(postorder[i:size])
}
```

##### 剑指offer 34：二叉树中和为某一值的路径

```go
func pathSum1(root *TreeNode, target int) [][]int {
	path := [][]int{}
	nums := []int{}
	var dfs func(*TreeNode, int)	
	dfs = func(root *TreeNode, sum int){
		if root == nil {
			return
		}
		sum -= root.Val
		nums = append(nums, root.Val)	//入栈
		if root.Left == nil && root.Right == nil && sum == 0 {
			path = append(path, append([]int{}, nums...))
		}
		dfs(root.Left, sum)
		dfs(root.Right, sum)
		nums = nums[:len(nums)-1]	//回退一步，相对于出栈
	}
	dfs(root, target)
	return path
}
```

##### 剑指offer 35：复杂链表的复制

解法一（时间O(n2），空间O(1)）：

1、复制节点，设置Next：复制原始链表的每一个节点，并用Next指针链接起来

2、设置Random：计算从原始链表头部开始沿着Next指针到达某一结点的Random所需的步数，在复制链表中走相同的步数设置Random

```go
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	//复制节点，设置Next
	copy := copyNode(head)
	//设置Random
	cur, copyCur := head, copy
	for cur != nil {
		p, ran, step := head, cur.Random, 0
		if ran != nil {		//Random不为空
			for p != ran {	//计算步数
				step++
				p = p.Next
			}
			for p = copy; step > 0; step-- {
				p = p.Next
			}
			copyCur.Random = p
		}else {		//Random为空
			copyCur.Random = nil
		}
		cur = cur.Next
		copyCur = copyCur.Next
	}
	return copy
}

func copyNode(head *Node) *Node {
	var copy, tail *Node
	//设置头节点
	copy = &Node{
		Val: head.Val,
		Next: nil,
	}
	tail = copy
	head = head.Next
	for head != nil {
		tail.Next = &Node{
			Val: head.Val,
			Next: nil,
		}
		tail = tail.Next
		head = head.Next
	}
	return copy
}
```

解法二（时间O(n)，空间O(n)）:

1、复制节点，设置Next，同时将原始节点和复制节点作为k-v存入map中

2、设置Random：根据原始链表中的Random，结合map中的配对信息，设置复制链表的Random（若A->D，则相对应A'->D'）

```go
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	//复制节点，设置Next
	copy, m := copyNode(head)
	//设置Random
	cur, copyCur := head, copy
	for cur != nil {
		if cur.Random == nil {
			copyCur.Random = nil
		}else{
			m[cur].Random = m[cur.Random]
		}
		cur = cur.Next
		copyCur = copyCur.Next
	}
	return copy
}

func copyNode(head *Node) (*Node, map[*Node]*Node) {
	var copy, tail *Node
	m := make(map[*Node]*Node)
	//设置头节点
	copy = &Node{
		Val: head.Val,
		Next: nil,
	}
	tail = copy
	m[head] = tail
	head = head.Next
	for head != nil {
		tail.Next = &Node{
			Val: head.Val,
			Next: nil,
		}
		tail = tail.Next
		m[head] = tail
		head = head.Next
	}
	return copy, m
}
```

解法三（时间：O(n)，空间：O(1)）:

1、复制节点，然后将每一个复制的节点依次插入到原始节点的后面，即A->A'->B-B'

2、设置Random：若A->D，则相对应'A'->D'

3、设置Next：将上述链表按照位置的奇偶一分为二，偶数位置的节点串起来即为复制链表

```go
func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}
	//复制节点
	copy := copyNode(head)
	//设置Random
	p := copy
	for p != nil {
		if p.Random != nil {
			p.Next.Random = p.Random.Next
		}
		p = p.Next.Next
	}
	//设置Next
	head = copy
	res := copy.Next
	tail1, tail2 := head, res
	cur := 0
	for p = res.Next; p != nil; p = p.Next {
		if cur == 0 {	//head（将原来的链表恢复，题目要求）
			tail1.Next = p
			tail1 = tail1.Next
			cur = 1
		}else{	//res
			tail2.Next = p
			tail2 = tail2.Next
			cur = 0
		}
	}
	tail1.Next = nil
	return res
}

func copyNode(head *Node) *Node {
	p := head
	for p != nil {
		node := &Node{
			Val: p.Val,
			Next: p.Next,
		}
		p.Next = node
		p = node.Next
	}
	return head
}
```

##### 剑指offer 36：二叉搜索树与双向链表

解法一：中序遍历非递归

```java
class Solution {
    //中序遍历非递归法
    public Node treeToDoublyList(Node root) {
        if (root == null) {
            return null;
        }
        Node head = null, tail = null;
        Stack<Node> stack = new Stack();
        while (root != null || !stack.isEmpty()) {
            while(root != null) {	//一直左走并入栈
                stack.push(root);
                root = root.left;
            }
            //访问当前节点
            root = stack.pop();
            if (tail != null) {
                tail.right = root;
                root.left = tail;
                tail = root;
            }else {		//头节点
                head = tail = root;
            }
            //往右转
            root = root.right;
        }
        //最后的首尾指针
        tail.right = head;
        head.left = tail;
        return head;
    }
}
```

解法二：中序遍历递归

```java
class Solution {
    Node head, tail;    //链表头尾指针
    public Node treeToDoublyList(Node root) {
        if (root == null) {	return null; }
        dfs(root);
        //最后的首尾指针
        head.left = tail;
        tail.right = head;
        return head;
    }
    void dfs(Node cur) {
        if (cur == null) { return; }
        dfs(cur.left);	//左
        if (tail != null) {	//链表为空
            tail.right = cur;
            cur.left = tail;
            tail = cur;
        }else{	//链表不为空
            head = tail = cur;
        }
        dfs(cur.right);	//右
    }
}
```

##### 剑指offer 37：序列化二叉树

```go 
type Codec struct {
}
func Constructor() Codec {
	return Codec{}
}
//通过先序遍历序列化二叉树
func (this *Codec) serialize(root *TreeNode) string {
	res := []byte{}
	var dfs func(*TreeNode)		//函数内定义函数，可以省去很多参数调用，res就可以当“全局变量”用
	dfs = func(cur *TreeNode) {
        //根
		if cur == nil {	//为空则填充"$"
			res = append(res, '$', ',')
			return
		}
		res = append(res, append([]byte{}, strconv.Itoa(cur.Val)...)...)	//关键步骤，哈哈
		res = append(res, ',')
		dfs(cur.Left)	//左子树
		dfs(cur.Right)	//右子树
	}
	dfs(root)	//函数调用
	return string(res)	//生成的字符串每个节点值用逗号分开，末尾还多了一个逗号
}
// 通过先序遍历反序列化二叉树
func (this *Codec) deserialize(data string) *TreeNode {
	s := strings.Split(data[:len(data)-1], ",")	//去除data末尾对于的逗号，然后根据逗号分割，得到每个节点值
	cur := 0	//用于控制当前访问的位置
	var dfs func() *TreeNode
	dfs = func() *TreeNode {
		if cur >= len(s) || s[cur] == "$" {
			cur++
			return nil
		}
		val, _ := strconv.Atoi(s[cur])
		cur++
		return &TreeNode{
			Val:   val,
			Left:  dfs(),
			Right: dfs(),
		}
	}
	return dfs()
}
```

##### 剑指offer 38：字符串的排列

题解：每次确定一个位置，将当前位置cur与其之后的元素依次交换，可确定当前位置有几种情况

```go
func permutation(s string) []string {
	res := []string{}
	length, bytes := len(s), []byte(s)
	var recursion func(int)
	recursion = func(pos int) {
		if pos == length {
			res = append(res, string(bytes))
			return
		}
		m := make(map[byte]bool)
        //该循环用于确定第一个位置元素
		for i := pos; i < length; i++ {
			if _, ok := m[bytes[i]]; ok {	//剪枝
				continue
			}
			m[bytes[i]] = true
			bytes[pos], bytes[i] = bytes[i], bytes[pos]	//依次交换
			recursion(pos+1)	//确定下一个位置
			bytes[pos], bytes[i] = bytes[i], bytes[pos]	//恢复原样（回溯）
		}

	}
	recursion(0)
	return res
}
```

扩展：如果题目是按照一定要求摆放若干个数字（字符），可以先求出这些数字（字符）的所有排列（全排列），然后剔除其中不符合要		   求的排列。

例如八皇后问题可用全排列解决：

   - 先定义一个长度为8的数组queen[]，数组中第 i 个数字代表第 i 行的皇后的列号，这样保证了皇后不在同一行
   - 对数组做全排列，用0-7填充，但要用不同数字填充，这样保证了皇后不在同一列
   - 对上述全排列进行筛选，剔除掉那些处于同一对角线的排列，这样保证了皇后不在同一对角线上，剩下的排列即为结果 

以下为全排列的例子（每一位由'0'-'9'组成）：

```go
//打印长度为n的字符串，字符串中每一位是由'0'-'9'组成
func print(length int) {
	res := make([]byte, length)	//存放每一次结果
	var recursion func(int)
    //pos用于控制当前访问位置
	recursion = func(pos int) {
		if pos == length {
			fmt.Println(string(res[:]))	//如果要将结果存入二维切片中，每次都要重新生成地址再append，否则会覆盖掉
			return	//return千万不要忘记！！！
		}
        //循环用于确定pos位置上的多种情况（注意：此循环只是确定一个位置）
		for i := 0; i < 10; i++ {
			res[pos] = byte('0' + i)
			recursion(pos + 1)	//确定下一个位置
		}
	}
	recursion(0)	//从第0位开始
}
```

##### 剑指offer 39：数组中出现次数超过一半的数字

解法一：若一个数字出现次数超过数据长度的一半，则数据排序后该数字为中位数，即N/2处。使用快排中的Partition()函数以及二分思想，可确定出该数字。时间为O(n)，虽然是O(n)，但执行速度挺慢，可以将此种思想掌握

```go
func majorityElement(nums []int) int {
	length := len(nums)
	low, middle, high := 0, length>>1, length-1
	index := partition(nums, low, high)
	for index != middle {
		if index > middle {
			high = index - 1
			index = partition(nums, low, high)
		}else{
			low = index + 1
			index = partition(nums, low, high)
		}
	}
	return nums[index]
}
func partition(nums []int, low, high int) int {
	part := nums[low]
	for low < high {
		for low < high && nums[high] >= part { high-- }
		nums[low] = nums[high]
		for low < high && nums[low] <= part { low++ }
		nums[high] = nums[low]
	}
	nums[low] = part
	return low
}
```

解法二：记录数字以及该数字出现次数，动态调整这两个值。时间为O(n)

本人将这种方法称之为打擂台法，例如[1,2,3,2,2]，1号上台，1号被2号打败了，2号被3号打败了，3号被2号打败了，2号又赢了一场，最后台上只剩下2号，如果给定数组中一定存在超过半数的数字，那此时2号即为结果，如果给定数组中可能不存在超过半数的数字，那就再查一查2号胜了几场，是否过半数

```go
func majorityElement(nums []int) int {
	cur, times := nums[0], 1	//第一位选手入场，规定每位选手入场时胜场数为1
	for i := 1; i < len(nums); i++ {
		if nums[i] != cur {	//被打败
			times--		//胜场数减一
			if times == 0 {	//没有机会了，换下一个人
				cur = nums[i]
				times = 1
			}
		} else {	//打赢了，胜场数加1
			times++
		}
	}
	return cur
}
```

##### 剑指offer 40：最小的k个数

解法一：还是利用快排中的Partition()函数和二分思想，将数组在k处划分。时间O(n)，改变了原数组

```go
func getLeastNumbers(arr []int, k int) []int {
    length := len(arr)
    if length == 0 || k == 0 { return []int{} }
    if k >= length { return arr }
    low, high := 0, length-1
    index := partition(arr, low, high)	//partition()函数见上一题
	for index != k-1 {
		if index > k-1 {
			high = index - 1
			index = partition(arr, low, high)
		}else{
			low = index + 1
			index = partition(arr, low, high)
		}
	}
    return arr[:k]
}
```

解法二：最大堆，时间为O(logk)，相对于解法一，该方法更适合海量数据的处理（因为不可能将所有数据都读入内存，该方法只需要在内存中维护一个大小为k的容器，再依次将数据读入即可）
思路：先读入k个数建成最大堆，根节点root为最大值，然后依次读入k之后的元素，若该元素大于root，则跳过，否则将该元素替换掉root，再调整为最大堆，依次。。。

```go
func getLeastNumbers(arr []int, k int) []int {
	length := len(arr)
	if length == 0 || k == 0{ return []int{} }
    if k >= length { return arr }
	maxHeap := make([]int, k+1)	//多申请一个空间，方便计算，左孩子节点下标=2*父节点下标
	for i := 0; i < k; i++ {	//将前k个数装入最大堆中
		maxHeap[i+1] = arr[i]
	}
	for i := len(maxHeap) >> 1; i > 0; i-- {	//将数组调整为最大堆，从最后一个非叶子节点往前开始调整
		adjust(maxHeap, i)	//将根节点为i的树调整为最大堆
	}
	for i := k; i < length; i++ {
		if arr[i] < maxHeap[1] {
			maxHeap[1] = arr[i]	  //替换掉大根堆的根节点
			adjust(maxHeap, 1)	//大根堆调整
		}
	}
	return maxHeap[1:]
}
//将nums数组中以pos为根节点的树调整为最大堆（从左右孩子中挑一个大的替换上来）
func adjust(nums []int, pos int) {
	parent, child, root, length := 0, 0, nums[pos], len(nums)
	for parent = pos; parent*2 < length; parent = child {
		child = parent * 2	//左孩子
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

##### 剑指offer 41：数据流中的中位数

```go
//困难题，暂时略过
```

##### 剑指offer 42：连续子数组的最大和

```go
//动态规划三部曲：
//1、分解子问题：以第i位结尾的数组的最值
//2、求得每个子问题的最优解：max(fn+nums[i], nums[i])
//3、取所有子问题最优解中的最值：max(fn, res)	
func maxSubArray(nums []int) int {
    res, fn := nums[0], nums[0]		//初始值
    for i := 1; i < len(nums); i++ {
        fn = max(fn+nums[i], nums[i])	//求取子问题最优解
        res = max(fn, res)				//取子问题最优解中的最值
    }	
    return res
}
```

##### 剑指offer 43：1~n整数中1的出现次数

```go
//困难题，暂时略过
```

##### 剑指offer 44：数字序列中某一位的数字

```go
func findNthDigit(n int) int {
	//找到n处数字是几位数（记为m）：将n逐渐减去第1、2、3...位数字的数目
	//得到m位数的起始数字大小，以及n处在此基础上的偏移量
	//得到n处是哪一个数字的哪一位
	if n < 10 {		//n处为1位数
		return n
	}
	digit, count := 2, 0	
	n -= 10	  //直接从2位数开始，先减去10个1位数，因为大于1的位数数量都有规律
	for {
		count = 9 * digit * int(math.Pow10(digit-1))	//计算digit位数的数字数量
		if n < count {	//已经确定是几位数
			pos := n%digit + 1	//此处加1是经过计算验证得知
			num := int(math.Pow10(digit-1)) + n/digit
			//求num的第pos位
			for i := 0; i < digit - pos; i++ {
				num /= 10
			}
			return num%10
		}
		n -= count
		digit++
	}
}
```

##### 剑指offer 45：把数组排成最小的数

```go
func minNumber(nums []int) string {
    sort.Slice(nums, func(i, j int) bool {	//排序，比较 xy 和 yx 的大小
        x, y := nums[i], nums[j]
        sx, sy := 10, 10
        for sx <= y {
            sx *= 10
        }
        for sy <= x {
            sy *= 10
        }
        return x*sx+y < y*sy+x
    })
    res := []byte{}
    for _, v := range nums {
        res = append(res, strconv.Itoa(v)...)
    }
    return string(res)
}
```

##### 剑指offer 46：把数字翻译成字符串

解法一：数字翻译过程类似于青蛙跳，只需计算每次的台阶数即可，最后将所有的跳数相乘

```go
func translateNum(num int) int {
	str := strconv.Itoa(num)
	res, step := 1, 1
	pre := str[0]
	for i := 1; i < len(str); i++ {
		if pre == '1' || (pre == '2' && str[i] < '6'){	//表示前置数、当前数结合后小于26，继续加1前进
			step++
		} else {	//本次台阶数已确定，计算跳数，且置step为1
			res *= steps(step)
			step = 1
		}
		pre = str[i]	//更新前置
	}
	res *= steps(step)	//计算最后一次
	return res
}
//青蛙一阶二阶跳
func steps(n int) int {
	if n <= 3 {
		return n
	}
	a, b, res := 2, 3, 0
	for i := 4; i <= n; i++ {
		res = a + b
		a, b = b, res
	}
	return res
}
```

解法二：递归：f(i) = f(i+1) + g(i, i+1) * f(i+2)，其中g(i, i+1)表示当前位以及下一位组成的数字是否小于26，若是则g(i, i+1) = 1，否则为0

```go
func translateNum(num int) int {
	str := strconv.Itoa(num)
	var recursion func(string) int
	recursion = func(s string) int {
		if len(s) < 2 {
			return 1
		}
        count := recursion(s[1:])	//f(i+1)
		if s[0] == '1' || (s[0] == '2' && s[1] < '6'){	//判断两位数是否小于26
            count += recursion(s[2:])	//f(i+2)
		}
		return count
	}
	return recursion(str)
}
```

解法三：将解法二中的递归转成动态规划，因为从左往右遍历存在重复子问题（12258拆一下就知道了），所以选择从右往左遍历

```go
func translateNum(num int) int {
	str := strconv.Itoa(num)
	if len(str) == 1 {
		return 1
	}
    pre1, pre2 := 1, 1	//pre1代表f(i+1), pre2代表f(i+2)
	for i := len(str)-2; i >= 0; i-- {
		if str[i] == '1' || (str[i] == '2' && str[i+1] < '6'){
            pre1, pre2 = pre1 + pre2, pre1 	//f(i) = f(i+1) + f(i+2)，pre1求和，pre2前移，等于原来的pre1
		}else{
            pre2 = pre1		//f(i) = f(i+1)，pre1不变，pre2前移，等于原来的pre1
		}
	}
	return pre1
}
```

##### 剑指offer 47：礼物的最大值

解法一：dp数组用二维存

```go
func maxValue(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}
	m, n := len(grid), len(grid[0])
	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			left, up := 0, 0
			if i > 0 {
				up = dp[i-1][j]
			}
			if j > 0 {
				left = dp[i][j-1]
			}
            dp[i][j] = maxNum(left, up) + grid[i][j]	//maxNum()为自定义函数
		}
	}
	return dp[m-1][n-1]
}
```

解法二：dp数组用一维存，长度为grid的列数，dp数组更新细节举例子画一画即可得，大体来讲，dp[i]代表上边，dp[i-1]代表左边

```go
func maxValue(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}
	m, n := len(grid), len(grid[0])
	dp := make([]int, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			left, up := 0, 0
			if i > 0 {
				up = dp[j]	
			}
			if j > 0 {
				left = dp[j-1]
			}
			dp[j] = max(left, up) + grid[i][j]
		}
	}
	return dp[n-1]
}
```

##### 剑指offer 48：最长不含重复字符的子字符串

滑动窗口+map

解法一：右指针一步步向前走，然后带动左指针的移动

```go
//right先前跑，同时拉着left，拉犁型
func lengthOfLongestSubstring2(s string) int {
    res := 0
    hash := map[byte]int{}	//int存的是下标
    left, right := 0, 0
    for right < len(s) {
        if v, ok := hash[s[right]]; ok {	//有重复元素
            res = max(right-left, res)	
            for left <= v {		//将v及之前的全部删除，同时左边界右移
                delete(hash, s[left])
                left++
            }
            hash[s[right]] = right
        }else{
            hash[s[right]] = right
        }
        right++		
    }
    res = max(right-left, res)		//最后一次
    return res
}
```

解法二：左指针一步步向前走，同时控制右指针的移动

```go
//相当于固定left，让right跑，看能跑多远， 发射型
func lengthOfLongestSubstring(s string) int {
    res := 0
    hash := map[byte]bool{}
    left, right := 0, -1
    n := len(s)
    for ; left < n; left++ {
        if left != 0 {
            delete(hash, s[left-1])
        }
        for right+1 < n && !hash[s[right+1]] {
            hash[s[right+1]] = true
            right++
        }
        res = max(res, right-left+1)
    }
    return res
}
```

##### 剑指offer 49：丑数

题解：本位丑数是由之前的丑数乘以2、3、5得到的，用 t2控制乘2，t3控制乘3，t5控制乘5，三个指针指向有时并不会指向同一个位置，三个指针是交叉前进

```go
func nthUglyNumber(n int) int {
    t2, t3, t5 := 0, 0, 0
	dp := make([]int, n)
	dp[0] = 1
	for i := 1; i < n; i++ {
		dp[i] = minNum(dp[t2]*2, dp[t3]*3, dp[t5]*5)
		if dp[t2] * 2 == dp[i] { t2++ }
		if dp[t3] * 3 == dp[i] { t3++ }
		if dp[t5] * 5 == dp[i] { t5++ }
	}
	return dp[n-1]
}

func minNum (a, b, c int) int {
	min := a
	if a > b {
		min = b
	}
	if min > c {
		min = c
	}
	return min
}
```

##### 剑指offer 50：第一个只出现一次的字符

###### 题目一：字符串中第一个只出现一次的字符

注意：这里的字符只包括小写字母，故哈希表大小为26即可，其他情况再做讨论

```go
func firstUniqChar(s string) byte {
	if len(s) == 0 { return ' '}
	arr := [26]int{}
	for _, v := range s {
		arr[v-'a']++
	}
	for _, v := range s {
		if arr[v-'a'] == 1 {
			return byte(v)
		}
	}
	return ' '
}
```

###### 题目二：字符流中第一个只出现一次的字符

```go
//无
```

##### 剑指offer 51：数组中的逆序对

```go
//困难，略过
```

##### 剑指offer 52：两个链表的第一个公共节点

解法一：先求出两个链表的长度，然后让较长的链表先走几步，然后两个链表同行找公共节点

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	lenA, lenB := length(headA), length(headB)
    //较长的链表先行
	if lenA > lenB {
		for ; lenA > lenB; lenA-- {
			headA = headA.Next
		}
	} else {
		for ; lenB > lenA; lenB-- {
			headB = headB.Next
		}
	}
    //两者同行
	for i := 0; i < lenA; i++ {
		if headA == headB {
			return headA
		} else {
			headA = headA.Next
			headB = headB.Next
		}
	}
	return nil
}

func length(head *ListNode) int {
	res := 0
	for head != nil {
		head = head.Next
		res++
	}
	return res
}
```

解法二：该解法不用先求出链表长度，而是通过交换跑道巧妙地让两者同行，交换跑道只会各自进行1次

链表A长度为 L1+C，链表B长度为 L2+C，其中C为公共部分，通过交换跑道，可以让两者均跑 L1+L2+C，这时已经相遇

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	pA, pB := headA, headB
	for pA != pB {
		if pA == nil {	//pA走到头了，换到B跑道上
			pA = headB	
		}else {
			pA = pA.Next
		}
		if pB == nil {
			pB = headA	//pB走到头了，换到A跑道上
		}else{
			pB = pB.Next
		}
	}
	return pA
}
```

##### 剑指offer 53：在排序数组中查找数字

###### 题目一：数字在排序数组中出现的次数

```go
func search(nums []int, target int) int {
    low, high, mid := 0, len(nums)-1, 0
	for low <= high {
		mid = (high-low) >> 1 + low
		if nums[mid] == target { 
			break 
		} else if nums[mid] < target {
			low = mid + 1
		}else {
			high = mid - 1
		}
	}
	if low > high { return 0 }
	res := 0
	for i := mid; i >= 0; i-- {	//往前找相同的
		if nums[i] == target {
			res++
		}else{
			break
		}
	}
	for i := mid+1; i < len(nums); i++ {	//往后找相同的
		if nums[i] == target {
			res++
		}else{
			break
		}
	}
	return res
}
```

###### 题目二：0~n-1中缺失的次数

题解：对于不缺失的地方，下标和值是相等的，从缺失处开始及之后，下标和值就不相等了，所以问题转化为求数组中第一个下标和值不相等的元素，采用二分查找

```go
func missingNumber(nums []int) int {
	low, high, mid := 0, len(nums)-1, 0
	for low <= high {
		mid = (high-low) >> 1 + low
		if nums[mid] == mid {	//相等表示缺失值在右边
			low = mid + 1
		}else {	//缺失值在左边或当前，所以需要判断当前是否是第一个不相等的地方
			if mid == 0 || nums[mid-1] == mid - 1 {	//判断mid前一个元素下标和值是否相等
				return mid	//若相等则mid处即为第一个下标和值不相等的元素
			}
			high = mid - 1	//不相等说明还在左边
		}
	}
    return low	//这种情况下low == len(nums)，即缺失值为n-1
}
```

##### 剑指offer 54：二叉搜索树的第K大节点

```go
func kthLargest(root *TreeNode, k int) int {
	ans := 0
	var inOrder func(*TreeNode) 
	inOrder = func(tn *TreeNode) {
		if tn == nil || k < 1{
			return 
		}
		inOrder(tn.Right)	//右			   
		if k == 1 {		    //中
			ans = tn.Val
			return
		}	
         k--	//就算已经得出ans，k--还是要做，否则递归回退时，ans会被覆盖
		inOrder(tn.Left)	//左
	}
	inOrder(root)
	return ans
}
```

##### 剑指offer 55：二叉树的深度

###### 题目一：二叉树的深度

题解一：递归

```go
func maxDepth(root *TreeNode) int {
    if root == nil {
		return 0
	}
	left, right := maxDepth(root.Left), maxDepth(root.Right)
    if left > right {
        return left + 1
    }else {
        return right + 1
    }
}
```

题解二：层序遍历

```go
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	ans := 0
	curLast, nextLast := root, root		//curLast、nextLast分别代表当前层和下一层的最后一个节点
	deque := &myQueue{}
	deque.push(root)
	if deque.head != nil {
		root = deque.head.Val
		if root.Left != nil {
			deque.push(root.Left)
			nextLast = root.Left
		}
		if root.Right != nil {
			deque.push(root.Right)
			nextLast = root.Right
		}
        deque.pop()
		if curLast == root {
			ans++
			curLast = nextLast
		}
	}
	return ans
}
```

###### 题目二：判断是否是平衡二叉树

题解一：自顶向下，对于同一个节点，函数depth会被重复调用

```go
func isBalanced(root *TreeNode) bool {	//先序遍历
    if root == nil { return true }
    left, right := depth(root.Left), depth(root.Right)
    if left == right || left - right == 1 || left - right == -1 { 
        return isBalanced(root.Left) && isBalanced(root.Right) 
    }
    return false
}
func depth(root *TreeNode) int {	//后序遍历
    if root == nil {
        return 0
    }
    left, right := depth(root.Left), depth(root.Right)
    if left > right {
        return left + 1
    }
    return right + 1
}
```

题解二：自底向上，即将题解一中的先序遍历改为后序遍历

```go
func isBalanced(root *TreeNode) bool {
	var recursion func(*TreeNode) int		//后序遍历
	recursion = func(tn *TreeNode) int {	//如果不平衡，返回-1退出，如果平衡，返回树的深度
        if tn == nil { return 0 }
        //左
        left := recursion(tn.Left)
        if left == -1 { return -1 }
        //右
        right := recursion(tn.Right)
        if right == -1 { return -1 }
        //中
	    if left - right > 1 || left - right < -1 { return -1 }
        return maxNum(left,right) + 1
	}
	return recursion(root) != -1
}
```

##### 剑指offer 56：数组中只出现一次的数字

###### 题目一：数组中只出现一次的两个数字（其余均出现两次）

题解：1、若是数组中只有一个出现一次的数字，那整个数组直接异或即可求出。2、本题中有两个只出现一次的数字，那就想办法将这两个数字分隔到两个数组中。3、分隔依据：将整个数组异或，得到是这两个数字的异或结果，记为num，则num肯定不为0，找出num的二进制形式中为1的最低位数，以此位数是否为1作为依据，将原数组分隔成两个数组，且每个数组中有一个只出现一次的数字，分别将这两个数组异或，即可求出两个结果

```go
func singleNumbers(nums []int) []int {
	one := 0
	for _, v := range nums {	//求出两个数字的异或
		one ^= v
	}
	n := one & (-one)	//得出为1的最低位数，这里求出的不是位数n，而是2^n，画一画就知道了
	num1, num2 := 0, 0
	for _, v := range nums {	//分隔为两个数组并各自异或
		if v & n == 0 {		//注意此处的判断标准，将v与2^n相与可判断其n位处是否是1
			num1 ^= v
		}else{
			num2 ^= v
		}
	}
	return []int{num1, num2}
}
```

###### 题目二：数组中只出现一次的数字（其余数字均出现三次）

题解：将数组中所有数字的二进制形式中每一位加起来，如果某一位的和能被3整除，说明那个只出现一次的数字在此位上为0，否则为1

```go
func singleNumber(nums []int) int {
	bitNum := [32]int{}
	for _, num := range nums {	//二进制每一位相加
		bitMask := 1
		for i := 31; i >= 0; i-- {
			if num & bitMask != 0 {
                bitNum[i] += 1
             }
			bitMask <<= 1
		}
	}
	res := 0
	for i := 0; i < 32; i++ {	
		if bitNum[i] % 3 == 1 {
			res += 1 << (31-i)
		}
	}
	return res
}
```

##### 剑指offer 57：和为s的数字

###### 题目一：和为s的两个数字 

```go
func twoSum(nums []int, target int) []int {
	low, high := 0, len(nums) - 1	//对撞指针
	for {
		if nums[low] + nums[high] == target {
			return []int{nums[low], nums[high]}
		}else if nums[low] + nums[high] > target {
			high--
		}else{
			low++
		}
	}
	return nil
}
```

###### 题目二：和为s的连续正数序列

```go
//滑动窗口
func findContinuousSequence(target int) [][]int {
	res := [][]int{}
	left, right, sum := 1, 2, 3
	for left < right {
		if sum < target {	//比target小，窗口右侧往外扩
			right++
			sum += right
		}else if sum > target{ 	//比target大，窗口左侧往里缩
			sum -= left
			left++
		}else{	//相等则保存结果，之后窗口左侧往里缩
             nums := make([]int, right-left+1)
             nums[0] = left
             for i := 1; i < len(nums); i++ {
                 nums[i] = nums[i-1] + 1
             }
             res = append(res, nums)
             sum -= left	
            left++	//窗口左侧往里缩（或者外扩）
		}
	}
	return res
}
```

##### 剑指offer 58：翻转字符串

###### 题目一：翻转单词顺序

题解：先整体翻转一遍，然后单词间再翻转一遍

```go
func reverseWords(s string) string {
	bytes := trim(s)	//去除字符串中多余的空格
	if len(bytes) == 0 { return "" }
	left, right := 0, 0
	reverse(bytes, 0, len(bytes)-1)	 //整体翻转
	for right <= len(bytes){
		if right == len(bytes) || bytes[right] == ' ' {
			reverse(bytes, left, right-1)	//各个单词翻转
			left = right + 1
		} 
        right++
	}
	return string(bytes)
}
func reverse(bytes []byte, left, right int) {
	for ; left < right; left, right = left+1, right-1 {
		bytes[left], bytes[right] = bytes[right], bytes[left]
	}
}
//去除字符串中多余的空格
func trim(s string) []byte {
	bytes := []byte{}
	for _, v := range s {
		if v != ' ' || (len(bytes) > 0 && bytes[len(bytes)-1] != ' '){	//当v不为空，或者v为空但bytes最后一位不为空
			bytes = append(bytes, byte(v))
		}
	}
	if len(bytes) > 0 && bytes[len(bytes)-1] == ' '{	//去除最后一位多出的空格
		return bytes[:len(bytes)-1]
	}
	return bytes
}
```

###### 题目二：左旋转字符串

```go
func reverseLeftWords(s string, n int) string {
	bytes := []byte(s)
	reverse(bytes, 0, n-1)
	reverse(bytes, n, len(bytes)-1)
	reverse(bytes, 0, len(bytes)-1)
	return string(bytes)
}
```

##### 剑指offer 59：队列的最大值

###### 题目一：滑动窗口的最大值

```go
func maxSlidingWindow(nums []int, k int) []int {
	if len(nums) == 0 || k == 0 { return nil }
	res := []int{}
	queue := []int{}	//双端队列当滑动窗口
	//当i < k时，为滑动窗口建立期，之后窗口开始右移
	for i := 0; i < len(nums); i++ {
		for len(queue) > 0 && queue[len(queue)-1] < nums[i] {	//排除队列中较小的值
			queue = queue[:len(queue)-1]	//删除末尾元素
		}
		queue = append(queue, nums[i])
		if i >= k && queue[0] == nums[i-k] {	//窗口右移
			queue = queue[1:]
		}
		if i >= k-1 {
			res = append(res, queue[0])
		}
	}
	return res
}
```

###### 题目二：队列的最大值

```go
type MaxQueue struct {
	dataQueue, maxQueue *list.List
}

func Constructor() MaxQueue {
	return MaxQueue{
		dataQueue: list.New(),
		maxQueue: list.New(),
	}
}
//队列首部记为最大值
func (this *MaxQueue) Max_value() int {
	if this.maxQueue.Len() == 0 {
		return -1
	}
	return this.maxQueue.Front().Value.(int)
}

func (this *MaxQueue) Push_back(value int)  {
	this.dataQueue.PushBack(value)
    //队列中比value小的全部出队
	for this.maxQueue.Len() != 0 && this.maxQueue.Back().Value.(int) < value {
		this.maxQueue.Remove(this.maxQueue.Back())
	}
	this.maxQueue.PushBack(value)
}
//若最大值同时出队
func (this *MaxQueue) Pop_front() int {
    if this.dataQueue.Len() == 0 { return -1 }
	res := this.dataQueue.Remove(this.dataQueue.Front()).(int)
	if res == this.maxQueue.Front().Value.(int){
		this.maxQueue.Remove(this.maxQueue.Front())
	}
	return res
}
```

##### 剑指offer 60：n个骰子的点数

题解一：递归全排列，超时

```go
func dicesProbability(n int) []float64 {
	nums := allSequence(n)
	res := make([]float64, len(nums))
	all := math.Pow(float64(6), float64(n))
	for i := 0; i < len(nums); i++ {
		res[i] = float64(nums[i]) / all
	}
	return res
}
//全排列
func allSequence(n int) []int {
	res := make([]int, 5*n+1)
	var recursion func(int, int)
	recursion = func(p int, sum int) {
		if p == n {
			res[sum-n]++
			return
		}
		for i := 0; i < 6; i++ {
			sum += i + 1
			recursion(p+1, sum)
			sum -= i + 1	//回溯
		}
	}
	recursion(0, 0)
	return res
}
```

题解二：动态规划

```go
func dicesProbability(n int) []float64 {
	pre := make([]float64, 6)
	for i := 0; i < 6; i++ {
		pre[i] = 1.0 / 6.0
	}
	for i := 2; i <= n; i++ {
		cur := make([]float64, 5*i+1)
		for j := 0; j < len(pre); j++ {		//由上一层得到下一层
			for k := 0; k < 6; k++ {
				cur[j+k] += pre[j] / 6.0
			}
		}
		pre = cur
	}
	return pre
}
```

##### 剑指offer 61：扑克牌的顺子

题解一：hash存储数组元素，找到数组中的最小值，然后开始往上枚举，没有则用0代替，如果没有0返回false

```go
func isStraight(nums []int) bool {
	hash := make(map[int]int)
	min := 14
	for _, v := range nums {
		if v > 0 &&  v < min {
			min = v
		}
		hash[v]++
	}
	res := 1
	for hash[min+1] > 0 || hash[0] > 0 {
		res++
		if hash[min+1] > 0 {
			hash[min+1]--
		}else{
			hash[0]--
		}
		min++
	}
	return res == 5
}
```

题解二：若组成顺子，需满足两个条件：数组中除了0之外没有重复元素，数组中 max - min < 5

```go
func isStraight(nums []int) bool {
	hash := make(map[int]bool)
	max, min := 0, 14
	for _, v := range nums {
		if v == 0 { continue }	//有0则跳过
		if hash[v] { return false }		//有重复元素，直接返回false
		if v > max { max = v }
		if v < min { min = v }
         hash[v] = true
	}
	return max - min < 5
}
```

##### 剑指offer 62：圆圈中最后剩下的数字

题解一：环状链表模拟

```go
//写不出来，略
```

题解二：数学公式

```go
func lastRemaining(n int, m int) int {
    last := 0
    for i := 2; i <= n; i++ {
        last = (last + m) % i
    }
    return last
}
```

##### 剑指offer 63：股票的最大利润

```go
//只有一次交易
func maxProfit(prices []int) int {
	if len(prices) == 0 { return 0}
	min, profit := prices[0], 0
	for _, price := range prices[1:] {
		if price < min {
			min = price
		}else if price - min > profit {
			profit = price - min
		}
	}
	return profit
}
```

##### 剑指offer 64：1+2+...+n

```go
//没得意思
```

##### 剑指offer 65：不用加减乘除做加法

```go
func add(a int, b int) int {
    sum, carry := 0, 0
    for {
        sum = a ^ b		//用异或模拟加法，但不处理进位
        carry = (a & b) << 1  //只有1 & 1 = 1，也只有1 + 1时才能进位，此处用 与 且左移一位（进位肯定往前进一位）模拟进位
        a, b = sum, carry	//继续循环计算
        if b == 0 {			//无进位时退出
            return a
        }
    }
}
```

##### 剑指offer 66：构建乘积数组

```go
func constructArr(a []int) []int {
	size := len(a)
	if size == 0 { return nil }
	res := make([]int, size)
	res[0] = 1
	for i := 1; i < size; i++ {	 //乘左半边
		res[i] = res[i-1] * a[i-1]
	}
	temp := 1
	for i := size-2; i >= 0; i-- {	//乘右半边
		temp *= a[i+1]	//记录右半边乘积
		res[i] *= temp
	}
	return res
}
```

##### 剑指offer 67：把字符串转换成整数

```go
func strToInt(str string) int {
	bytes := []byte(str)
	size := len(bytes)
	//寻找第一个非空字符
	index := 0
	for index < size && bytes[index] == ' ' {
		index++
	}
	if index < size {
		sign := 1
		if bytes[index] == '-' {	//负
			sign = -1
			index++
		}else if bytes[index] == '+' {	//正
			index++
		}
		res := 0
		for i := index; i < size; i++ {
			if bytes[i] >= '0' && bytes[i] <= '9' {
				res = res * 10 + int(bytes[i] - '0') //go中int位数与CPU位数一致，此处为64位，其余语言需考虑越界问题
			}else{
				break
			}
			if res > math.MaxInt32 {
				if sign == -1 {
					return math.MinInt32
				}
				return math.MaxInt32
			}
		}
		return sign*res
	}
	return 0
}
```

##### 剑指offer 68：二叉树的最近公共祖先

###### 题目一：该二叉树为二叉搜索树

题解一：两次遍历且用到了额外空间

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	pPath, qPath := path(root, p), path(root, q)
	var res *TreeNode
	for i := 0; i < len(pPath) && i < len(qPath) && pPath[i] == qPath[i]; i++ {	//寻找路径中最后一个相同的节点
		res = pPath[i]
	}
	return res
}
//寻找路径
func path(head, p *TreeNode) []*TreeNode {
	res := []*TreeNode{}
	for head.Val != p.Val {
		res = append(res, head)
		if head.Val > p.Val {
			head = head.Left
		}else{
			head = head.Right
		}
	}
	res = append(res, head)
	return res
}
```

题解二：一次遍历且无需额外空间

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	for {
		if root.Val < p.Val && root.Val < q.Val {	//两节点都在root右边
			root = root.Right
		}else if root.Val > p.Val && root.Val > q.Val {		//两节点都在root左边
			root = root.Left
		}else{	//两节点在root一左一右或者其中一个就是root，此时root即为结果
			return root
		}
	}
}
```

###### 题目二：该二叉树为普通二叉树

题解一：超慢的递归算法

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	for {
		if isInTree(root.Left, p) && isInTree(root.Left, q) {	//是否都在左边
			root = root.Left
		} else if isInTree(root.Right, p) && isInTree(root.Right, q) {	//是否都在右边
			root = root.Right
		} else {	//一左一右或者其中一个就是root
			return root
		}
	}
}
//判断节点是否在树上
func isInTree(root, p *TreeNode) bool {
	if root == nil {
		return false
	}
	if root == p {
		return true
	}
	return isInTree(root.Left, p) || isInTree(root.Right, p)
}

```

题解二：快一点的递归算法

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil || root == p || root == q {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)	//找左边
	right := lowestCommonAncestor(root.Right, p, q) //找右边
	if left == nil { return right }	//都在右边
	if right == nil { return left }	//都在左边
	return root	//一左一右
}
```

题解三：用哈希表存储父节点

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	parents := make(map[int]*TreeNode)
	var dfs func(*TreeNode)
	dfs = func(tn *TreeNode) {	
		if tn != nil {
			if tn.Left != nil {
				parents[tn.Left.Val] = tn
				dfs(tn.Left)
			}
			if tn.Right != nil {
				parents[tn.Right.Val] = tn
				dfs(tn.Right)
			}
		}
	}
	//用哈希表存储各节点的父节点
	parents[root.Val] = nil
	dfs(root)	
	//遍历p的父节点
	pPath := make(map[*TreeNode]bool)
	for p != nil {
		pPath[p] = true
		p = parents[p.Val]
	}
	//遍历q的父节点，同时对比q的父节点
	for q != nil {
		if pPath[q] {
			return q
		}
		q = parents[q.Val]
	}
	return nil
}
```

